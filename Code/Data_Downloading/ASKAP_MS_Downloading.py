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
CASDA_BASE_PATH = '/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Data/Ms_Data'
INPUT_CSV = project_path('Processed_Data/Catalogue/02.final_confirmed_stars_direct.csv')
FAILED_LIST_PATH = os.path.join(os.path.dirname(CASDA_BASE_PATH), '0.failed_ms_downloads_log.csv')

# ————————————————— 核心控制参数 —————————————————
TARGET_SOURCES = ['2MASS J01033563-5515561 A']
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
    pmra = src.get('sy_pmra') if 'sy_pmra' in src else src.get('pmra')
    pmdec = src.get('sy_pmdec') if 'sy_pmdec' in src else src.get('pmdec')
    missing = []
    if pd.isna(pmra): missing.append('sy_pmra'); pmra = 0.0
    if pd.isna(pmdec): missing.append('sy_pmdec'); pmdec = 0.0
    if missing:
        tqdm.write(f"⚠️ [NO PROPER MOTION] {src['hostname']}: missing {', '.join(missing)}; "
                   f"using pmra={float(pmra):.3f}, pmdec={float(pmdec):.3f} mas/yr. "
                   "Epoch propagation continues without a complete reliable PM correction.")

    source_coords = SkyCoord(
        ra=src['ra'] * un.deg,
        dec=src['dec'] * un.deg,
        pm_ra_cosdec=float(pmra) * un.mas / un.yr,
        pm_dec=float(pmdec) * un.mas / un.yr,
        frame='icrs',
        obstime=Time('J2015.5'),
        distance=100 * un.pc
    )

    for attempt in range(MAX_RETRIES):
        try:
            # === 1. 检索数据并锁定最佳波束 ===
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
            local_errors = []

            for sb in unique_history_sbs:
                sb_df = df_res[df_res['obs_id'] == sb]

                # 儒略历检查与丢弃机制
                mjd_val = sb_df['t_min'].iloc[0]
                if pd.isna(mjd_val):
                    warn_msg = f"{sb} 缺失儒略历时间数据"
                    tqdm.write(f"源 {clean_hostname} 的 {sb} 缺失儒略历数据，无法进行历元推演。已记录。")
                    # 【优化】补充 File 字段占位，保持字典结构一致
                    local_errors.append({'Source': clean_hostname, 'File': 'Unknown', 'Error': warn_msg})
                    continue

                epoch = Time(mjd_val, format='mjd')
                pm_coords = source_coords.apply_space_motion(epoch)
                beam_coords = SkyCoord(sb_df['s_ra'].values, sb_df['s_dec'].values, unit=(un.deg, un.deg))

                separations = pm_coords.separation(beam_coords)
                best_idx_in_sb = np.argmin(separations)

                best_filename = sb_df['filename'].iloc[best_idx_in_sb]
                global_idx = df_res.index[df_res['filename'] == best_filename].tolist()[0]

                sb_id_num = sb.replace('ASKAP-', '')
                safe_orig_name = best_filename.split('/')[-1]
                expected_local_name = f"{sb_id_num}_{safe_orig_name}"
                file_path = os.path.join(source_dir, expected_local_name)
                # 期望的 checksum 文件路径
                checksum_path = f"{file_path}.checksum"
                # 只有当主文件和其专属的 .checksum 文件同时存在时，才认为下载完整
                is_complete = os.path.exists(file_path) and os.path.exists(checksum_path)

                if not is_complete:
                    if os.path.exists(file_path): os.remove(file_path)
                    # 为了干净，如果残存了旧的校验文件也顺手删掉
                    if os.path.exists(checksum_path): os.remove(checksum_path)
                    files_to_download_indices.append(global_idx)

            if not files_to_download_indices:
                if local_errors:
                    return True, f" [源 {clean_hostname}] 部分波束因缺失数据跳过，其余就绪。", local_errors
                return True, f" [源 {clean_hostname}] 历元上的 {len(unique_history_sbs)} 个beams已就绪，跳过", []

            # === 2. 分批下载（核心修复区域） ===
            total_files = len(files_to_download_indices)
            downloaded_count = 0

            for i in range(0, total_files, BATCH_SIZE):
                batch_indices = files_to_download_indices[i: i + BATCH_SIZE]
                indices_array = np.array(batch_indices)
                download_table = results[indices_array]

                # 【核心修复 1】：预先提取当前批次的具体文件名，方便报错时精准记录
                current_batch_files = []
                for idx in batch_indices:
                    row = results[idx]
                    sb_num = row['obs_id'].replace('ASKAP-', '')
                    fname = os.path.basename(row['filename'])
                    current_batch_files.append(f"{sb_num}_{fname}")

                # 【核心修复 2】：为单个批次设置局部重试与异常捕获，防止波及整个源
                batch_success = False
                batch_err_msg = ""

                for inner_attempt in range(MAX_RETRIES):
                    try:
                        url_list = casda.stage_data(download_table)
                        if url_list:
                            filelist = casda.download_files(url_list, savedir=CASDA_BASE_PATH)
                            if filelist:
                                for downloaded_file in filelist:
                                    orig_basename = os.path.basename(downloaded_file)
                                    for idx in batch_indices:
                                        row = results[idx]
                                        main_file_basename = os.path.basename(row['filename'])

                                        # 如果下载的文件是主文件，或者主文件名加上 .checksum
                                        if orig_basename == main_file_basename or orig_basename == f"{main_file_basename}.checksum":
                                            sb_id_num = row['obs_id'].replace('ASKAP-', '')
                                            final_path = os.path.join(source_dir, f"{sb_id_num}_{orig_basename}")

                                            if os.path.exists(final_path): os.remove(final_path)
                                            os.rename(downloaded_file, final_path)

                                            # 只有在归类主文件时，才增加成功下载的计数
                                            if orig_basename == main_file_basename:
                                                downloaded_count += 1
                                            break
                                batch_success = True
                                break  # 批次下载成功，跳出局部重试
                            else:
                                raise Exception("文件下载返回为空列表")
                        else:
                            raise Exception("Staging 失败，未返回下载链接")

                    except Exception as inner_e:
                        batch_err_msg = str(inner_e)
                        if inner_attempt < MAX_RETRIES - 1:
                            time.sleep(10)  # 下载失败局部冷却
                        else:
                            pass  # 局部重试耗尽，跳出交由下方记录错误

                # 【核心修复 3】：记录具体文件名，且使用 continue 继续下一个批次
                if not batch_success:
                    tqdm.write(f" [错误] 源 {clean_hostname} 第 {i // BATCH_SIZE + 1} 批次下载彻底失败。")
                    for f_name in current_batch_files:
                        local_errors.append({'Source': clean_hostname, 'File': f_name, 'Error': batch_err_msg})
                    # 注意：这里没有 return，代码会进入下一次 for 循环，继续下载后续文件批次

            # 所有批次循环结束评估结果
            if downloaded_count > 0 or not local_errors:
                return True, f" [源 {clean_hostname}] 成功分批获取 {downloaded_count} 份数据。", local_errors
            else:
                return False, f" [源 {clean_hostname}] 尝试下载，但所有批次均失败。", local_errors

        except Exception as e:
            # 这里捕获的是 TAP 请求层面的全局灾难性异常
            err_msg = str(e)
            if any(k in err_msg for k in ["IncompleteRead", "Connection broken", "Timeout", "EOFError", "time out"]):
                if attempt < MAX_RETRIES - 1:
                    time.sleep(15)
                    continue
            return False, f" [源 {clean_hostname}] 全局网络错误: {err_msg}", [
                {'Source': clean_hostname, 'File': 'Global Error', 'Error': err_msg}]

    return False, f" [源 {clean_hostname}] 连续 {MAX_RETRIES} 次全局重试均失败", [
        {'Source': clean_hostname, 'File': 'Global Error', 'Error': 'Max retries reached'}]


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
