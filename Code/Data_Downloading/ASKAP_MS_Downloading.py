import numpy as np
import pandas as pd
import os
import time
import warnings
import keyring
from astropy.io.votable.exceptions import VOTableSpecWarning
from astropy.coordinates import SkyCoord
import astropy.units as un
from astropy.time import Time
from astroquery.casda import Casda
from astroquery.utils.tap.core import TapPlus
from tqdm import tqdm

# 屏蔽无关警告
warnings.filterwarnings('ignore', category=VOTableSpecWarning)
warnings.filterwarnings('ignore', module='astropy.io.votable')


# ————————————————— 1. 自动化环境与路径管理 —————————————————
def project_path(relative_path):
    current = os.path.abspath(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd())
    while not (
            os.path.isdir(os.path.join(current, 'Code')) and
            os.path.isdir(os.path.join(current, 'Processed_Data'))
    ):
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.join(os.getcwd(), relative_path)
        current = parent
    return os.path.join(current, relative_path)


# CASDA 账号配置
keyring.core.set_keyring(keyring.core.load_keyring('keyrings.cryptfile.cryptfile.CryptFileKeyring'))
OPAL_USER = "acentauri_huangst@163.com"

# 路径配置
CASDA_BASE_PATH = project_path('Downloading_Data/ms_data')
INPUT_CSV = project_path('Processed_Data/Catalogue/02.final_confirmed_stars_direct_1.csv')
FAILED_LIST_PATH = os.path.join(os.path.dirname(CASDA_BASE_PATH), '0.failed_ms_downloads_log.csv')

# ————————————————— 核心控制参数 —————————————————
TARGET_SOURCES = ['AU Mic', 'Proxima Cen', 'GJ 896 A', 'GJ 4274', 'COCONUTS-2 A', 'AF Lep', '2MASS J01033563-5515561 A', 'PZ Tel', 'AB Pic']

MAX_RETRIES = 3
BATCH_SIZE = 15  # 每次最多向服务器请求的文件数量，避免 414 URI Too Long

# ————————————————— 2. 初始化与数据预处理  —————————————————
os.makedirs(CASDA_BASE_PATH, exist_ok=True)

casda = Casda()
casda.login(username=OPAL_USER, store_password=True)
print("CASDA 登录成功，正在初始化...")
tap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")

try:
    df = pd.read_csv(INPUT_CSV)

    if 'hostname' not in df.columns:
        df['hostname'] = 'Target_' + df.index.astype(str)

    valid_df = df.drop_duplicates(subset=['hostname']).copy()

    if TARGET_SOURCES:
        valid_df = valid_df[valid_df['hostname'].isin(TARGET_SOURCES)].reset_index(drop=True)
        print(f"\n[筛选激活] 仅处理列表中的特定源: {TARGET_SOURCES}")
        if valid_df.empty:
            print(" 在 CSV 文件中未找到您指定的特定源，请检查名称是否匹配。程序已退出。")
            exit()

    for col in ['sy_pmra', 'sy_pmdec']:
        if col not in valid_df.columns:
            valid_df[col] = 0.0
        valid_df[col] = valid_df[col].fillna(0.0)

    source_list = valid_df.to_dict('records')
    print(f"成功解析 CSV。共提取 {len(source_list)} 个独立源，准备执行全历元检索。")
except Exception as e:
    print(f"读取或解析 CSV 失败: {e}")
    exit()


# ————————————————— 3. 核心逻辑函数 (以“源”为驱动) —————————————————
def process_source(src):
    clean_hostname = str(src['hostname']).replace(' ', '_')
    source_dir = os.path.join(CASDA_BASE_PATH, clean_hostname)
    os.makedirs(source_dir, exist_ok=True)

    # 建立星表的基础坐标基准 (固定为 J2015.5)
    source_coords = SkyCoord(
        ra=src['ra'] * un.deg,
        dec=src['dec'] * un.deg,
        pm_ra_cosdec=src['sy_pmra'] * un.mas / un.yr,
        pm_dec=src['sy_pmdec'] * un.mas / un.yr,
        frame='icrs',
        obstime=Time('J2015.5'),
        distance=100 * un.pc  # 设置虚拟距离支持自行计算
    )

    for attempt in range(MAX_RETRIES):
        try:
            # 2角秒筛选
            query = (
                f"SELECT * FROM ivoa.obscore "
                f"WHERE dataproduct_type = 'visibility' "
                f"AND t_exptime > 360 "
                f"AND quality_level != 'BAD' "
                f"AND obs_id LIKE 'ASKAP-%' "
                f"AND obs_collection NOT LIKE '%BETA%' "
                f"AND 1 = CONTAINS(POINT('ICRS', {source_coords.ra.deg}, {source_coords.dec.deg}), CIRCLE('ICRS', s_ra, s_dec, 2.0))"
            )

            job = tap.launch_job_async(query)
            results = job.get_results()

            if len(results) == 0:
                return True, f" [源 {clean_hostname}] 历元上未被任何合格的 ASKAP MS 数据覆盖", []

            results = Casda.filter_out_unreleased(results)
            if len(results) == 0:
                return True, f" [源 {clean_hostname}] 历元 MS 数据存在但尚未公开释放", []

            df_res = results.to_pandas()
            unique_history_sbs = df_res['obs_id'].unique()

            files_to_download_indices = []
            local_errors = []  # 用于收集因 t_min 缺失而跳过的局部错误

            for sb in unique_history_sbs:
                sb_df = df_res[df_res['obs_id'] == sb]

                # 儒略历检查与丢弃机制
                mjd_val = sb_df['t_min'].iloc[0]
                if pd.isna(mjd_val):
                    warn_msg = f"{sb} 缺失儒略历时间数据"
                    tqdm.write(f"源 {clean_hostname} 的 {sb} 缺失儒略历数据，无法进行历元推演。已记录。")
                    # 将缺失记录添加到错误列表中，最终会汇总到 CSV
                    local_errors.append({'Source': clean_hostname, 'Error': warn_msg})
                    continue  # 直接跳过这个 SB，处理下一个

                epoch = Time(mjd_val, format='mjd')

                # 根据该历元时间，计算恒星真实坐标
                pm_coords = source_coords.apply_space_motion(epoch)

                # 提取这一批次 36 个波束的中心点
                beam_coords = SkyCoord(sb_df['s_ra'].values, sb_df['s_dec'].values, unit=(un.deg, un.deg))

                # 用修正后的 pm_coords 寻找距离最近的波束
                separations = pm_coords.separation(beam_coords)
                best_idx_in_sb = np.argmin(separations)

                best_filename = sb_df['filename'].iloc[best_idx_in_sb]
                global_idx = df_res.index[df_res['filename'] == best_filename].tolist()[0]

                # 拼接文件名
                sb_id_num = sb.replace('ASKAP-', '')
                safe_orig_name = best_filename.split('/')[-1]
                expected_local_name = f"{sb_id_num}_{safe_orig_name}"
                file_path = os.path.join(source_dir, expected_local_name)

                is_complete = os.path.exists(file_path) and os.path.getsize(file_path) > 100 * 1024 * 1024

                if not is_complete:
                    if os.path.exists(file_path): os.remove(file_path)
                    files_to_download_indices.append(global_idx)

            if not files_to_download_indices:
                if local_errors:
                    return True, f" [源 {clean_hostname}] 部分波束因缺失儒略历数据跳过，其余就绪。", local_errors
                return True, f" [源 {clean_hostname}] 历元上的 {len(unique_history_sbs)} 个最佳波束已在源文件夹中就绪，跳过", []

            # 分批切割下载列表，防止请求 URL 过长 (414 Error)
            total_files = len(files_to_download_indices)
            downloaded_count = 0

            for i in range(0, total_files, BATCH_SIZE):
                batch_indices = files_to_download_indices[i: i + BATCH_SIZE]
                indices_array = np.array(batch_indices)
                download_table = results[indices_array]

                url_list = casda.stage_data(download_table)

                if url_list:
                    filelist = casda.download_files(url_list, savedir=CASDA_BASE_PATH)

                    if filelist:
                        for downloaded_file in filelist:
                            orig_basename = os.path.basename(downloaded_file)
                            for idx in batch_indices:
                                row = results[idx]
                                if orig_basename in row['filename']:
                                    sb_id_num = row['obs_id'].replace('ASKAP-', '')
                                    final_path = os.path.join(source_dir, f"{sb_id_num}_{orig_basename}")

                                    if os.path.exists(final_path): os.remove(final_path)
                                    os.rename(downloaded_file, final_path)
                                    downloaded_count += 1
                                    break
                else:
                    tqdm.write(f" [警告] 源 {clean_hostname} 第 {i // BATCH_SIZE + 1} 批次 Staging 失败。")

            if downloaded_count > 0:
                # 即使下载成功，也要把之前收集到的跳过错误一起返回保存
                return True, f" [源 {clean_hostname}] 成功分批下载 {downloaded_count} 份数据。", local_errors
            else:
                return False, f" [源 {clean_hostname}] 无法获取下载链接 ", [{'Source': clean_hostname,
                                                                             'Error': 'Staging failed across all batches'}] + local_errors

        except Exception as e:
            err_msg = str(e)
            if any(k in err_msg for k in ["IncompleteRead", "Connection broken", "Timeout", "EOFError", "time out"]):
                if attempt < MAX_RETRIES - 1:
                    time.sleep(15)
                    continue
            return False, f" [源 {clean_hostname}] 错误: {err_msg}", [{'Source': clean_hostname, 'Error': err_msg}]

    return False, f" [源 {clean_hostname}] 连续 {MAX_RETRIES} 次重试失败", [
        {'Source': clean_hostname, 'Error': 'Max retries reached'}]


# ————————————————— 4. 执行单线程安全循环 —————————————————
failed_records = []

print(f"\n目标主目录: {CASDA_BASE_PATH}")
print("-" * 60)

for src in tqdm(source_list, desc="历元修正 MS 检索进度"):
    success, message, errors = process_source(src)
    tqdm.write(message)

    if errors:
        failed_records.extend(errors)

# ————————————————— 5. 报告总结与日志输出 —————————————————
print("\n" + "=" * 60)
print(f"  全历元精准 MS 检索统计报告:")
print(f" - 处理天体源总数: {len(source_list)}")
print(f" - 失败/跳过记录数: {len(failed_records)}")

if failed_records:
    log_df = pd.DataFrame(failed_records)
    log_df.to_csv(FAILED_LIST_PATH, index=False, encoding='utf-8-sig')
    print(f"\n 下为异常名单 (已保存为标准化 CSV 至: {FAILED_LIST_PATH})")
    print(log_df.head())
else:
    print(" 所有天体源的全历元精确 MS 数据已检索并下载完毕。")

print("=" * 60)