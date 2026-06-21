import numpy as np
import pandas as pd
import os
import glob
import time
import keyring
from astropy.coordinates import SkyCoord
import astropy.units as un
from astropy.time import Time
from astroquery.casda import Casda
from astroquery.utils.tap.core import TapPlus
from astropy.table import Table

# 屏蔽Astropy pixel单位警告
import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)


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


# ————————————————— 核心控制参数 —————————————————
MAX_RETRIES = 3  # 全局/局部最大重试次数
MIN_FILE_SIZE = 10 * 1024  # 验证下载完整性的最小文件大小 (10KB)
CUTOUT_WIDTH = 60 * un.arcsec
# ————————————————————————————————————————————————

# 1. CASDA 账号配置
try:
    keyring.core.set_keyring(keyring.core.load_keyring('keyrings.cryptfile.cryptfile.CryptFileKeyring'))
except Exception as e:
    pass

OPAL_USER = "acentauri_huangst@163.com"
casda = Casda()
casda.login(username=OPAL_USER, store_password=True)

# 2. 路径配置与数据读取
time_info_file = project_path('Processed_Data/Catalogue/01.askap_catalogue.csv')
Time_info = pd.read_csv(time_info_file)

star_catalog_file = project_path('Processed_Data/Catalogue/02.final_confirmed_stars_direct_1.csv')
star_df = pd.read_csv(star_catalog_file)
star_df = star_df.drop_duplicates(subset=['hostname']).reset_index(drop=True)

print(f"共有 {len(star_df)} 个独立的恒星源准备进行切片下载。")

# 2.5 优先级排序区
priority_list = ['AB Pic','AF Lep','AU Mic', 'COCONUTS-2 A','GJ 229','GJ 896 A','GJ 3323','HD 41004 A']
if priority_list:
    star_df['priority'] = star_df['hostname'].apply(lambda x: 0 if x in priority_list else 1)
    star_df = star_df.sort_values('priority').drop(columns=['priority']).reset_index(drop=True)
    print(f"已调整优先级：将优先处理 {priority_list}，随后处理剩余源。")

download_base_dir = '/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Data/Fits_image'
failed_downloads = []

# 3. 下载循环
for index, Star in star_df.iterrows():
    hostname = Star['hostname']
    safe_hostname = str(hostname).replace(" ", "_")
    print(f"\n[{index + 1}/{len(star_df)}] 正在处理目标源: {hostname}")

    source_coords = SkyCoord(
        ra=Star['ra'] * un.deg,
        dec=Star['dec'] * un.deg,
        pm_ra_cosdec=Star['sy_pmra'] * un.mas / un.yr,
        pm_dec=Star['sy_pmdec'] * un.mas / un.yr,
        frame='icrs',
        obstime=Time('J2015.5'),
        distance=100 * un.pc
    )

    Stokes_list = ['I', 'V']
    for stokes_param in Stokes_list:
        cutout_path = os.path.join(download_base_dir, safe_hostname, f"Stokes{stokes_param}")
        os.makedirs(cutout_path, exist_ok=True)

        # === 带有全局重试的 TAP 检索 ===
        r = None
        for attempt in range(MAX_RETRIES):
            try:
                image_tap_qry = (
                    f"SELECT * FROM ivoa.obscore WHERE pol_states = '/{stokes_param}/' AND "
                    f"dataproduct_subtype = 'cont.restored.t0' AND "
                    f"1 = CONTAINS(POINT('ICRS',{source_coords.ra.deg},{source_coords.dec.deg}),s_region)"
                )
                tap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")
                job = tap.launch_job_async(image_tap_qry)
                r = job.get_results()
                break  # 检索成功，跳出重试
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(5)
                else:
                    print(f"  -> [网络错误] 连续检索 TAP 失败: {e}")

        if r is None or len(r) == 0:
            print(f"  -> [Stokes {stokes_param}] CASDA 中未找到历史观测，跳过。")
            continue

        r = Casda.filter_out_unreleased(r)
        image_list = r.to_pandas()

        # 过滤条件
        image_list = image_list[image_list['obs_id'].str.contains('ASKAP')]
        image_list = image_list[image_list['quality_level'] != 'BAD']
        image_list = image_list[~image_list['filename'].str.contains('raw|alt|highres|iqr')]
        image_list = image_list[~image_list['obs_collection'].str.contains('BETA')]

        if image_list.empty:
            continue

        # 挂载时间数据
        image_list = pd.merge(image_list, Time_info[['obs_id', 't_min']], on='obs_id', how='left',
                              suffixes=('', '_user'))
        if 't_min_user' in image_list.columns:
            image_list['t_min'] = image_list['t_min_user'].combine_first(image_list['t_min'])
            image_list.drop(columns=['t_min_user'], inplace=True)
        image_list.rename(columns={'t_min': 'Time'}, inplace=True)

        # 提取按历元去重后的 SBID 列表
        unique_history_sbs = image_list['obs_id'].unique()

        # === 历元循环：直接获取该历元的首条记录进行切割 ===
        for sb in unique_history_sbs:
            sb_df = image_list[image_list['obs_id'] == sb]
            mjd_val = sb_df['Time'].iloc[0]

            # 异常值拦截
            if pd.isna(mjd_val) or mjd_val == 0.0:
                print(f"  -> [跳过] {sb} 缺失时间数据，记录日志。")
                failed_downloads.append(
                    {'Target': hostname, 'SBID': sb, 'Stokes': stokes_param, 'Error': 'Missing t_min'})
                continue

            # 基于该波束观测时间推算新坐标
            epoch = Time(mjd_val, format='mjd')
            pm_coords = source_coords.apply_space_motion(epoch)

            # 1. 直接获取当前 SBID 的记录 (FITS 图像已拼接，无需波束择优)
            best_row = sb_df.iloc[0]
            sbid_full = best_row['obs_id']

            # 2. 本地完整性与查重机制 (采用体积判定代替 checksum)
            search_pattern = os.path.join(cutout_path, f"{safe_hostname}_{sbid_full}_Stokes{stokes_param}_*.fits")
            existing_files = glob.glob(search_pattern)
            is_complete = False

            for f in existing_files:
                if os.path.getsize(f) > MIN_FILE_SIZE:
                    is_complete = True
                    break

            if is_complete:
                print(f"  -> [跳过] {sbid_full} (Stokes {stokes_param}) 数据已存在且完整。")
                continue
            elif existing_files:
                # 删除不完整的残存文件
                for f in existing_files:
                    try:
                        os.remove(f)
                    except OSError:
                        pass

            for orphaned_file in glob.glob(os.path.join(cutout_path, "cutout-*.fits")):
                try:
                    os.remove(orphaned_file)
                except OSError:
                    pass

            # 3. 单次切片任务的局部重试循环
            url_info_df = Table.from_pandas(pd.DataFrame([best_row]))
            batch_success = False
            batch_err_msg = ""

            for inner_attempt in range(MAX_RETRIES):
                try:
                    # 每次循环都保证切割坐标基于当前的 PM 动态算好
                    url_list = casda.cutout(url_info_df, coordinates=pm_coords, radius=CUTOUT_WIDTH)
                    if not url_list:
                        raise Exception("Cutout 返回空链接")

                    filelist = casda.download_files(url_list, savedir=cutout_path)

                    if filelist:
                        for downloaded_file in filelist:
                            orig_basename = os.path.basename(downloaded_file)
                            new_basename = f"{safe_hostname}_{sbid_full}_Stokes{stokes_param}_{orig_basename}"
                            new_filepath = os.path.join(cutout_path, new_basename)

                            if os.path.exists(new_filepath):
                                os.remove(new_filepath)
                            os.rename(downloaded_file, new_filepath)

                            # 确认下载并非产生空壳文件
                            if os.path.getsize(new_filepath) > MIN_FILE_SIZE:
                                print(f"  -> [成功下载] 匹配 {sbid_full}, 保存为 {new_basename}")
                                batch_success = True
                            else:
                                os.remove(new_filepath)
                                raise Exception("文件大小不足 10KB，可能损坏或仅包含 header，触发重试")

                    if batch_success:
                        break  # 成功跳出局部重试

                except Exception as inner_e:
                    batch_err_msg = str(inner_e)
                    if inner_attempt < MAX_RETRIES - 1:
                        time.sleep(5)
                    else:
                        pass  # 耗尽次数，留给下方记录

            if not batch_success:
                print(f"  -> [下载失败!] {hostname} - {sbid_full} | 报错: {batch_err_msg}")
                failed_downloads.append({
                    'Target': hostname,
                    'SBID': sbid_full,
                    'Stokes': stokes_param,
                    'Error': batch_err_msg
                })

# 4. 生成错误日志文件
print("\n" + "=" * 50)
print("所有目标源处理完毕")
if len(failed_downloads) > 0:
    print(f"【注意】共有 {len(failed_downloads)} 个数据请求/提取失败！")
    log_df = pd.DataFrame(failed_downloads)
    fail_log_path = os.path.join(os.path.dirname(download_base_dir), 'failed_cutout_log.csv')
    log_df.to_csv(fail_log_path, index=False, encoding='utf-8-sig')
    print(f"【成功】失败名单已保存至: {fail_log_path}\n")
else:
    print("所有匹配到的cutout数据已全部成功下载完毕，无报错！")
print("=" * 50 + "\n")