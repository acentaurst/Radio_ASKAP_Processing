import pandas as pd
import numpy as np
import re
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from tqdm import tqdm
import warnings
import os
import glob

warnings.filterwarnings('ignore')


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


# --- 1. 配置文件路径 ---
XML_DIR = project_path('Downloading_Data/ASKAP_Catalogue')
ASKAP_CATALOGUE_CSV = project_path('Processed_Data/Catalogue/01.askap_catalogue.csv')
FINAL_OUTPUT_CSV = project_path('Processed_Data/Catalogue/02.final_confirmed_stars_direct_3.csv')

# 你的原始恒星表
STAR_CATALOG_CSV = project_path('Processed_Data/Catalogue/PS-Simbad.csv')

PRECISION_THRESHOLD = 3.0  # 3角秒


def extract_sbid_robust(text):
    if pd.isna(text): return None
    text = str(text)
    sb_match = re.search(r'SB(\d+)', text, re.IGNORECASE)
    if sb_match: return str(int(sb_match.group(1)))
    digit_match = re.search(r'(\d+)', text)
    if digit_match: return str(int(digit_match.group(1)))
    return None


def run_final_pipeline():
    print(" 直接交叉 (Direct Crossmatch)...")

    # --- 2. 加载 ASKAP 观测元数据 ---
    obs_df = pd.read_csv(ASKAP_CATALOGUE_CSV)
    obs_df.columns = obs_df.columns.str.strip()
    obs_df['sbid_clean'] = obs_df['obs_id'].apply(extract_sbid_robust)
    # 建立映射字典
    sbid_to_mjd = obs_df.dropna(subset=['sbid_clean']).drop_duplicates(subset=['sbid_clean']).set_index('sbid_clean')[
        't_min'].to_dict()
    print(f" 成功提取 {len(sbid_to_mjd)} 个 ASKAP 时间基准。")

    # --- 3. 加载原始恒星目录 ---
    if not os.path.exists(STAR_CATALOG_CSV):
        print(f" 错误：找不到恒星目录文件 {STAR_CATALOG_CSV}。")
        return

    print(f" 读取恒星星表: {STAR_CATALOG_CSV}")
    stars_df = pd.read_csv(STAR_CATALOG_CSV)
    stars_df.columns = stars_df.columns.str.strip()

    # 优先保留 NASA 官方推荐的参数行
    if 'default_flag' in stars_df.columns:
        stars_df = stars_df[stars_df['default_flag'] == 1]

    # 剔除坐标为空的无效数据
    stars_df = stars_df.dropna(subset=['ra', 'dec']).reset_index(drop=True)

    # pmra = stars_df['sy_pmra'].fillna(0.0).to_numpy(copy=True)
    # pmdec = stars_df['sy_pmdec'].fillna(0.0).to_numpy(copy=True)
    pmra = stars_df['pmra'].fillna(0.0).to_numpy(copy=True)
    pmdec = stars_df['pmdec'].fillna(0.0).to_numpy(copy=True)
    plx = stars_df['sy_plx'].fillna(1.0).to_numpy(copy=True)
    plx[plx <= 0] = 1.0

    print(f" 正在解析恒星坐标 (J2015.5)，共 {len(stars_df)} 条记录...")
    stars_j2015 = SkyCoord(
        ra=stars_df['ra'].values * u.deg,
        dec=stars_df['dec'].values * u.deg,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        distance=(1000 / plx) * u.pc,
        obstime=Time('J2015.5')
    )

    # --- 4. 遍历 XML 并实时匹配 ---
    xml_files = glob.glob(os.path.join(XML_DIR, '*.xml'))
    final_candidates = []
    missing_time_files = 0
    read_error_count = 0

    print(f" 找到 {len(xml_files)} 个 XML 文件，开始交叉匹配...")

    for xml_file in tqdm(xml_files, desc="交叉匹配中"):
        basename = os.path.basename(xml_file)
        sbid = extract_sbid_robust(basename)
        # 进行映射
        if not sbid or sbid not in sbid_to_mjd:
            missing_time_files += 1
            continue

        mjd_val = sbid_to_mjd[sbid]
        try:
            obs_time = Time(mjd_val, format='mjd')
        except:
            continue

        try:
            temp_df = Table.read(xml_file, format='votable').to_pandas()
        except:
            read_error_count += 1
            continue

        if 'col_ra_deg_cont' not in temp_df.columns or 'col_dec_deg_cont' not in temp_df.columns:
            continue

        temp_df = temp_df.dropna(subset=['col_ra_deg_cont', 'col_dec_deg_cont']).reset_index(drop=True)
        if temp_df.empty:
            continue

        stars_at_obs = stars_j2015.apply_space_motion(new_obstime=obs_time)

        askap_sources = SkyCoord(
            ra=temp_df['col_ra_deg_cont'].values * u.deg,
            dec=temp_df['col_dec_deg_cont'].values * u.deg
        )

        idx_askap, d2d, _ = stars_at_obs.match_to_catalog_sky(askap_sources)
        valid_matches = d2d < (PRECISION_THRESHOLD * u.arcsec)
        matched_stars_indices = np.where(valid_matches)[0]

        for i_star in matched_stars_indices:
            i_askap = idx_askap[i_star]

            star_row = stars_df.iloc[i_star].to_dict()
            askap_row = temp_df.iloc[i_askap].to_dict()
            merged_info = {**star_row, **askap_row}

            for k, v in merged_info.items():
                if isinstance(v, bytes):
                    merged_info[k] = v.decode('utf-8', errors='ignore')

            merged_info['origin_xml'] = basename
            merged_info['sbid_clean'] = sbid
            merged_info['obs_date_formatted'] = obs_time.to_datetime().strftime('%Y.%m.%d')
            merged_info['corrected_ra_deg'] = round(stars_at_obs[i_star].ra.deg, 7)
            merged_info['corrected_dec_deg'] = round(stars_at_obs[i_star].dec.deg, 7)
            merged_info['obs_mjd'] = mjd_val
            merged_info['obs_decimalyear'] = round(obs_time.decimalyear, 4)
            merged_info['true_sep_arcsec'] = round(d2d[i_star].arcsec, 4)

            final_candidates.append(merged_info)

    # --- 5. 汇总保存 ---
    if final_candidates:
        final_df = pd.DataFrame(final_candidates)
        # 一颗宿主星在一次SBID观测中，只出现一行记录！
        if 'hostname' in final_df.columns and 'sbid_clean' in final_df.columns:
            final_df = final_df.drop_duplicates(subset=['hostname', 'sbid_clean'])
        all_cols = list(final_df.columns)

        # 更改坐标排列
        group1 = ['obs_date_formatted', 'corrected_ra_deg', 'corrected_dec_deg', 'true_sep_arcsec']
        group2 = ['rastr', 'ra', 'decstr', 'dec']

        for c in group1 + group2:
            if c in all_cols:
                all_cols.remove(c)

        # 插入到 index 2 (即第 3, 4, 5, 6 列)
        for i, col in enumerate(group1):
            if col in final_df.columns:
                all_cols.insert(2 + i, col)

        # 插入到 index 6 (即第 7, 8, 9, 10 列)
        for i, col in enumerate(group2):
            if col in final_df.columns:
                all_cols.insert(6 + i, col)

        final_df = final_df[all_cols]
        # ==========================================

        final_df = final_df.sort_values(by='true_sep_arcsec').reset_index(drop=True)
        final_df.to_csv(FINAL_OUTPUT_CSV, index=False)
        print(f"\n 完成！解析出 {len(final_df)} 个匹配坐标。")
        print(f" 结果已保存至: {FINAL_OUTPUT_CSV}")
    else:
        print("\n 未找到符合交叉阈值的目标。")


if __name__ == "__main__":
    run_final_pipeline()
