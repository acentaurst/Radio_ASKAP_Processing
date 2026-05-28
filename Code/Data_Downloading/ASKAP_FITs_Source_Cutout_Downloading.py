import numpy as np
import pandas as pd
import os
import glob
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
    current = os.path.abspath(os.path.dirname(__file__))
    while not (
            os.path.isdir(os.path.join(current, 'Code')) and
            os.path.isdir(os.path.join(current, 'Processed_Data'))
    ):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.join(current, relative_path)


# 1. CASDA 账号配置
keyring.core.set_keyring(keyring.core.load_keyring('keyrings.cryptfile.cryptfile.CryptFileKeyring'))

OPAL_USER = "acentauri_huangst@163.com"
casda = Casda()
casda.login(username=OPAL_USER, store_password=True)

# 2. 路径配置与数据读取
time_info_file = project_path('Processed_Data/Catalogue/01.askap_catalogue.csv')
Time_info = pd.read_csv(time_info_file)

star_catalog_file = project_path('Processed_Data/Catalogue/02.final_confirmed_stars_direct_1.csv')
star_df = pd.read_csv(star_catalog_file)

# 根据hostname列去除重复行
star_df = star_df.drop_duplicates(subset=['hostname']).reset_index(drop=True)

print(f"共有 {len(star_df)} 个独立的恒星源准备进行切片下载。")

# 2.5 优先级排序区
priority_list = ['GJ 4274']

if 'priority_list' in locals() and priority_list:
    star_df['priority'] = star_df['hostname'].apply(lambda x: 0 if x in priority_list else 1)
    star_df = star_df.sort_values('priority').drop(columns=['priority']).reset_index(drop=True)
    print(f"已调整优先级：将优先处理 {priority_list}，随后处理剩余源。")

download_base_dir = project_path('Downloading_Data/fits_images')
cutout_width = 60 * un.arcsec

# 用于记录下载失败的任务
failed_downloads = []

# 3. 下载循环
for index, Star in star_df.iterrows():
    hostname = Star['hostname']

    # 生成安全的文件名
    safe_hostname = str(hostname).replace(" ", "_")

    print(f"\n[{index + 1}/{len(star_df)}] 正在处理目标源: {hostname} (保存路径: {safe_hostname})")

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

        image_tap_qry = (
            f"SELECT * FROM ivoa.obscore WHERE pol_states = '/{stokes_param}/' AND "
            f"dataproduct_subtype = 'cont.restored.t0' AND "
            f"1 = CONTAINS(POINT('ICRS',{source_coords.ra.deg},{source_coords.dec.deg}),s_region)"
        )

        tap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")
        job = tap.launch_job_async(image_tap_qry)
        r = job.get_results()

        if len(r) == 0:
            print(f"  -> [Stokes {stokes_param}] CASDA 中未找到历史观测，跳过。")
            continue

        r = Casda.filter_out_unreleased(r)
        image_list = r.to_pandas()
        initial_count = len(image_list)

        # 过滤条件
        image_list = image_list[image_list['obs_id'].str.contains('ASKAP')]
        image_list = image_list[image_list['quality_level'] != 'BAD']
        image_list = image_list[~image_list['filename'].str.contains('raw|alt|highres|iqr')]
        image_list = image_list[~image_list['obs_collection'].str.contains('BETA')]

        filtered_count = len(image_list)
        print(f"    [统计] CASDA 共搜到 {initial_count} 条记录，过滤后剩余 {filtered_count} 个有效 SBID 准备下载...")

        if image_list.empty:
            continue

        image_list = pd.merge(image_list, Time_info[['obs_id', 't_min']], on='obs_id', how='left',
                              suffixes=('', '_user'))
        if 't_min_user' in image_list.columns:
            image_list['t_min'] = image_list['t_min_user'].combine_first(image_list['t_min'])
            image_list.drop(columns=['t_min_user'], inplace=True)

        image_list.rename(columns={'t_min': 'Time'}, inplace=True)

        for _, row in image_list.iterrows():
            filename = row['filename']
            sbid_full = row['obs_id']
            mjd_val = row['Time']

            # t_min 缺失检查拦截
            if pd.isna(mjd_val) or mjd_val == 0.0:
                print(
                    f"  -> [跳过]  {hostname} 的 {sbid_full} (Stokes {stokes_param}) 缺失mjd时间数据，无法进行历元推演。已记录。")
                failed_downloads.append({
                    'Target': hostname,
                    'SBID': sbid_full,
                    'Stokes': stokes_param,
                    'Error': 'Missing t_min for proper epoch propagation'
                })
                continue

            # 通过检查后，再进行转换
            epoch = Time(mjd_val, format='mjd')

            # 使用 Astropy Table 切片获取 URL info，喂给 Casda
            url_info_df = Table.from_pandas(image_list[image_list['filename'] == filename])

            # 查重机制
            search_pattern = os.path.join(cutout_path, f"{safe_hostname}_{sbid_full}_Stokes{stokes_param}_*.fits")
            existing_files = glob.glob(search_pattern)

            if len(existing_files) > 0:
                print(f"  -> [跳过] 已存在 {hostname} 的 {sbid_full} (Stokes {stokes_param}) 数据，不再重复下载。")
                continue

            # 容错下载机制
            try:
                # 基于提取的 epoch 进行坐标历元自行推算
                pm_coords = source_coords.apply_space_motion(epoch)
                url_list = casda.cutout(url_info_df, coordinates=pm_coords, radius=cutout_width)

                if len(url_list) == 0:
                    continue

                filelist = casda.download_files(url_list, savedir=cutout_path)

                if filelist:
                    for downloaded_file in filelist:
                        orig_basename = os.path.basename(downloaded_file)
                        new_basename = f"{safe_hostname}_{sbid_full}_Stokes{stokes_param}_{orig_basename}"
                        new_filepath = os.path.join(cutout_path, new_basename)

                        os.rename(downloaded_file, new_filepath)

                        if new_basename.endswith('.fits'):
                            print(f"  -> [成功下载] 匹配 {sbid_full}, 保存为 {new_basename}")

            except Exception as e:
                print(f"  -> [下载失败!] {hostname} - {sbid_full} (Stokes {stokes_param}) | 报错信息: {e}")
                failed_downloads.append({
                    'Target': hostname,
                    'SBID': sbid_full,
                    'Stokes': stokes_param,
                    'Error': str(e)
                })

# 4. 生成错误日志文件 (CSV)
print("\n" + "=" * 50)
print("所有目标源处理完毕")
if len(failed_downloads) > 0:
    print(f"【注意】共有 {len(failed_downloads)} 个数据请求/提取失败！")

    log_df = pd.DataFrame(failed_downloads)
    save_dir = os.path.dirname(download_base_dir)
    fail_log_path = os.path.join(save_dir, 'failed_cutout_log.csv')

    log_df.to_csv(fail_log_path, index=False, encoding='utf-8-sig')
    print(f"【成功】失败名单已保存至: {fail_log_path}\n")

    for fail in failed_downloads:
        print(f" - 目标源: {fail['Target']} | SBID: {fail['SBID']} | Stokes: {fail['Stokes']}")
        print(f"   报错: {fail['Error']}")
else:
    print("所有匹配到的cutout数据已全部成功下载完毕，无报错！")
print("=" * 50 + "\n")