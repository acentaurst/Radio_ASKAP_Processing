"""Retrieve ASKAP measurement-set archives selected from the project catalogue."""

import numpy as np
import pandas as pd
import os
import sys
import tempfile
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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from science_utils import quiet_casda_download, safe_download_error  # noqa: E402

# ————————————————— 1. 自动化环境与路径管理 —————————————————



# CASDA 账号配置
OPAL_USER = "acentauri_huangst@163.com"

# 路径配置
CASDA_BASE_PATH = '/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Data/Ms_Data'
INPUT_CSV = PROJECT_ROOT / "Processed_Data" / "Catalogue" / "02.final_confirmed_stars_direct.csv"
FAILED_LIST_PATH = os.path.join(os.path.dirname(CASDA_BASE_PATH), '0.failed_ms_downloads_log.csv')

# ————————————————— 核心控制参数 —————————————————
TARGET_SOURCES = ['2MASS J01033563-5515561 A']
MAX_RETRIES = 3
BATCH_SIZE = 15  # 每次最多向服务器请求的文件数量，避免 414 URI Too Long
MIN_FILE_SIZE = 10 * 1024  # MS archive minimum; checksum needs only be nonempty


def _download_ms_product(casda, row_table, source_dir, sb_num, main_name):
    """Stage anew on every attempt and commit a validated MS/checksum pair."""
    error_message = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with tempfile.TemporaryDirectory(prefix=".ms_download_", dir=CASDA_BASE_PATH) as temp_dir:
                urls = casda.stage_data(row_table)
                if not urls:
                    raise ValueError("CASDA staging returned no URLs")
                with quiet_casda_download():
                    returned = casda.download_files(urls, savedir=temp_dir)
                if not returned:
                    raise ValueError("CASDA download returned no files")
                paths = {Path(path).name: Path(path) for path in returned}
                main_file = paths.get(main_name)
                checksum_file = paths.get(f"{main_name}.checksum")
                if main_file is None or checksum_file is None:
                    raise ValueError(f"Missing MS or checksum for {main_name}")
                for path, minimum in ((main_file, MIN_FILE_SIZE), (checksum_file, 0)):
                    if path.resolve().parent != Path(temp_dir).resolve():
                        raise ValueError("CASDA returned a file outside the staging directory")
                    if not path.is_file() or path.stat().st_size <= minimum:
                        raise ValueError(f"Incomplete CASDA file: {path.name}")

                # Commit the main archive last: a present main file never masks
                # a missing checksum on the next run.
                os.replace(checksum_file, Path(source_dir) / f"{sb_num}_{main_name}.checksum")
                os.replace(main_file, Path(source_dir) / f"{sb_num}_{main_name}")
            return True, ""
        except Exception as error:
            error_message = safe_download_error(error)
            tqdm.write(f"  [MS {main_name} {attempt}/{MAX_RETRIES}] {error_message}")
            if attempt < MAX_RETRIES:
                time.sleep(10)
    return False, error_message

# ————————————————— 2. 初始化与数据预处理  —————————————————

def main() -> None:
    keyring.core.set_keyring(keyring.core.load_keyring('keyrings.cryptfile.cryptfile.CryptFileKeyring'))
    os.makedirs(CASDA_BASE_PATH, exist_ok=True)

    casda = Casda()
    try:
        casda.login(username=OPAL_USER, store_password=True)
    except Exception as error:
        raise RuntimeError(f"CASDA 登录失败: {safe_download_error(error)}") from None
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


    # ————————————————— 3. 核心逻辑 (以“源”为驱动) —————————————————
    # 该流程只在下方主循环中调用一次，直接内联以保持下载状态和错误记录相邻。
    failed_records = []
    print(f"\n目标主目录: {CASDA_BASE_PATH}")
    print("-" * 60)
    for src in tqdm(source_list, desc="历元修正 MS 检索进度"):
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

        process_success = False
        process_message = ""
        process_errors = []
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

                # CASDA 返回 VOTable 时可能产生已知格式提示；只在这次
                # TAP 请求上下文内抑制，不影响后续坐标与下载逻辑的警告。
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=VOTableSpecWarning)
                    warnings.filterwarnings('ignore', module='astropy.io.votable')
                    job = tap.launch_job_async(query)
                    results = job.get_results()

                if len(results) == 0:
                    process_success = True
                    process_message = f" [源 {clean_hostname}] 历元上未被任何合格的 ASKAP MS 数据覆盖"
                    process_errors = []
                    break

                results = Casda.filter_out_unreleased(results)
                if len(results) == 0:
                    process_success = True
                    process_message = f" [源 {clean_hostname}] 历元 MS 数据存在但尚未公开释放"
                    process_errors = []
                    break

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
                    # 不完整文件保留至成功重下后原子替换，避免失败时丢失旧数据。
                    is_complete = (
                        os.path.isfile(file_path)
                        and os.path.getsize(file_path) > MIN_FILE_SIZE
                        and os.path.isfile(checksum_path)
                        and os.path.getsize(checksum_path) > 0
                    )

                    if not is_complete:
                        files_to_download_indices.append(global_idx)

                if not files_to_download_indices:
                    if local_errors:
                        process_success = True
                        process_message = f" [源 {clean_hostname}] 部分波束因缺失数据跳过，其余就绪。"
                        process_errors = local_errors
                    else:
                        process_success = True
                        process_message = f" [源 {clean_hostname}] 历元上的 {len(unique_history_sbs)} 个beams已就绪，跳过"
                        process_errors = []
                    break

                # === 2. 分批下载（核心修复区域） ===
                total_files = len(files_to_download_indices)
                downloaded_count = 0

                for i in range(0, total_files, BATCH_SIZE):
                    batch_indices = files_to_download_indices[i: i + BATCH_SIZE]
                    # 批次仍限制处理规模；按文件隔离 staging 与下载，坏链接
                    # 不会让同批其他 MS 一起失败。
                    for idx in batch_indices:
                        row = results[idx]
                        sb_num = row['obs_id'].replace('ASKAP-', '')
                        main_name = os.path.basename(row['filename'])
                        success, error_message = _download_ms_product(
                            casda,
                            results[np.array([idx])],
                            source_dir,
                            sb_num,
                            main_name,
                        )
                        if success:
                            downloaded_count += 1
                        else:
                            local_errors.append({
                                'Source': clean_hostname,
                                'File': f"{sb_num}_{main_name}",
                                'Error': error_message,
                            })

                # 所有批次循环结束评估结果
                if downloaded_count > 0 or not local_errors:
                    process_success = True
                    process_message = f" [源 {clean_hostname}] 成功分批获取 {downloaded_count} 份数据。"
                    process_errors = local_errors
                else:
                    process_success = False
                    process_message = f" [源 {clean_hostname}] 尝试下载，但所有批次均失败。"
                    process_errors = local_errors
                break

            except Exception as e:
                # 这里捕获的是 TAP 请求层面的全局灾难性异常
                err_msg = safe_download_error(e)
                if any(k in err_msg for k in ["IncompleteRead", "Connection broken", "Timeout", "EOFError", "time out"]):
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(15)
                        continue
                process_success = False
                process_message = f" [源 {clean_hostname}] 全局网络错误: {err_msg}"
                process_errors = [
                    {'Source': clean_hostname, 'File': 'Global Error', 'Error': err_msg}]
                break

        else:
            process_success = False
            process_message = f" [源 {clean_hostname}] 连续 {MAX_RETRIES} 次全局重试均失败"
            process_errors = [
                {'Source': clean_hostname, 'File': 'Global Error', 'Error': 'Max retries reached'}]

        tqdm.write(process_message)
        if process_errors:
            failed_records.extend(process_errors)


    # ————————————————— 4. 报告总结与日志输出 —————————————————
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

if __name__ == "__main__":
    main()
