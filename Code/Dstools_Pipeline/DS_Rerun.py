"""重跑管线（跳过 create-model），变量命名与批量管线完全一致。"""
import os, re, sys, shutil, tarfile, subprocess, glob
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord

# ========== 配置 ==========
# True：批量处理目录下全部 .ms.tar；False：只处理 SINGLE_TAR_FILE。
BATCH_PROCESS = True
SINGLE_TAR_FILE = (
    "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Data/Ms_Data/"
    "2MASS_J01033563-5515561_A/"
    "59565_scienceData.EMU_0054-55.SB59565.EMU_0054-55.beam22_averaged_cal.leakage.ms.tar"
)

# 批量模式递归扫描目标目录下所有 .ms.tar 文件。
# 注意：这里匹配的是 .ms.tar 文件，不是已经解压的 .ms 文件夹。
TAR_ROOT = (
    "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/"
    "Result/Dynamic Spectrum/2MASS_J01033563-5515561_A"
)
BATCH_TAR_PATHS = sorted(
    glob.glob(os.path.join(TAR_ROOT, "**", "*.ms.tar"), recursive=True)
)
TAR_PATHS = BATCH_TAR_PATHS if BATCH_PROCESS else [SINGLE_TAR_FILE]
if not TAR_PATHS:
    target = TAR_ROOT if BATCH_PROCESS else SINGLE_TAR_FILE
    raise FileNotFoundError(f"未找到待重跑的 .ms.tar 文件: {target}")
mode_name = "批量" if BATCH_PROCESS else "单个"
print(f"{mode_name} rerun：共 {len(TAR_PATHS)} 个 MS tar 文件")
MASK_RADIUS: int = 15        # 掩模半径（角秒）
# ==========================


def project_path(relative_path: str) -> str:
    current = os.path.abspath(os.path.dirname(__file__))
    while not (os.path.isdir(os.path.join(current, 'Code')) and os.path.isdir(os.path.join(current, 'Processed_Data'))):
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.join(os.getcwd(), relative_path)
        current = parent
    return os.path.join(current, relative_path)


def extract_sbid_and_beam(filename: str):
    sb_match = re.search(r'SB(\d+)', filename, re.IGNORECASE)
    beam_match = re.search(r'beam(\d+)', filename, re.IGNORECASE)
    sbid = str(int(sb_match.group(1))) if sb_match else None
    beam = str(int(beam_match.group(1))) if beam_match else None
    return sbid, beam


def run_cmd(cmd_str: str, cwd: str) -> None:
    conda_bin_dir = os.path.dirname(sys.executable)
    conda_lib_dir = os.path.join(os.path.dirname(conda_bin_dir), "lib")
    custom_env = os.environ.copy()
    custom_env["PATH"] = conda_bin_dir + os.pathsep + custom_env.get("PATH", "")
    custom_env["LD_LIBRARY_PATH"] = conda_lib_dir + os.pathsep + custom_env.get("LD_LIBRARY_PATH", "")
    cmd_str = f"env LD_LIBRARY_PATH={custom_env['LD_LIBRARY_PATH']} {cmd_str}"
    subprocess.run(cmd_str, shell=True, check=True, cwd=cwd, executable='/bin/bash', env=custom_env)


ASKAP_CATALOGUE_CSV = project_path('Processed_Data/Catalogue/01.askap_catalogue.csv')
INPUT_CSV = project_path('Processed_Data/Catalogue/02.final_confirmed_stars_direct.csv')
PIPELINE_RESULTS_BASE = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Result/DS"


def process_tar(tar_path: str) -> None:
    """Rerun one existing tar using the WSClean model retained in its workspace."""
    if not os.path.isfile(tar_path):
        raise FileNotFoundError(f"找不到 MS tar: {tar_path}")

    sbid, beam = extract_sbid_and_beam(os.path.basename(tar_path))
    if not sbid or not beam:
        raise ValueError(f"无法从文件名解析 SBID/beam: {tar_path}")

    obs_df = pd.read_csv(ASKAP_CATALOGUE_CSV)
    obs_df.columns = obs_df.columns.str.strip()
    obs_df['sbid_clean'] = obs_df['obs_id'].apply(
        lambda x: str(int(re.search(r'(\d+)', str(x)).group(1))) if re.search(r'(\d+)', str(x)) else None)
    sbid_to_mjd = obs_df.dropna(subset=['sbid_clean']).drop_duplicates(subset=['sbid_clean']).set_index('sbid_clean')['t_min'].to_dict()
    if sbid not in sbid_to_mjd:
        raise KeyError(f"ASKAP catalogue 未找到 SB{sbid} 的 t_min")
    obs_mjd = sbid_to_mjd[sbid]

    stars_df = pd.read_csv(INPUT_CSV)
    stars_df.columns = stars_df.columns.str.strip()
    stars_df['hostname_clean'] = stars_df['hostname'].astype(str).str.strip().str.replace(' ', '_')
    star_catalog_dict = stars_df.drop_duplicates(subset=['hostname_clean']).set_index('hostname_clean').to_dict('index')

    dir_name = os.path.basename(os.path.dirname(tar_path))
    norm_dir = re.sub(r'[^a-zA-Z0-9]', '', dir_name).lower()
    matched_key = None
    for k in star_catalog_dict:
        norm_k = re.sub(r'[^a-zA-Z0-9]', '', k).lower()
        if norm_k in norm_dir or norm_dir in norm_k:
            matched_key = k
            break
    if matched_key is None:
        raise KeyError(f"无法由 tar 父目录匹配宿主星: {dir_name}")
    clean_hostname = matched_key
    star_meta = star_catalog_dict[clean_hostname]

    pmra = star_meta.get('sy_pmra') if 'sy_pmra' in star_meta else star_meta.get('pmra')
    pmdec = star_meta.get('sy_pmdec') if 'sy_pmdec' in star_meta else star_meta.get('pmdec')
    missing_pm = []
    if pd.isna(pmra): missing_pm.append('sy_pmra'); pmra = 0.0
    if pd.isna(pmdec): missing_pm.append('sy_pmdec'); pmdec = 0.0
    if missing_pm:
        print(f"⚠️ [NO PROPER MOTION] {clean_hostname}: missing {', '.join(missing_pm)}; "
              f"using pmra={float(pmra):.3f}, pmdec={float(pmdec):.3f} mas/yr. "
              "Epoch propagation continues without a complete reliable PM correction.")
    plx_val = star_meta.get('sy_plx', star_meta.get('plx', 10.0))
    plx = 10.0 if pd.isna(plx_val) or float(plx_val) <= 0 else float(plx_val)

    star_j2015 = SkyCoord(ra=star_meta['ra'] * u.deg, dec=star_meta['dec'] * u.deg,
                          pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                          distance=(1000 / plx) * u.pc, frame='icrs', obstime=Time('J2015.5'))
    obs_time = Time(obs_mjd, format='mjd')
    star_at_obs = star_j2015.apply_space_motion(new_obstime=obs_time)
    corr_ra, corr_dec = round(star_at_obs.ra.deg, 7), round(star_at_obs.dec.deg, 7)

    star_results_dir = os.path.join(PIPELINE_RESULTS_BASE, clean_hostname)
    ds_results_dir = os.path.join(star_results_dir, "DS_Results")
    workspace_name = f"{clean_hostname}_SB{sbid}_beam{beam}_workspace"
    workspace_dir = os.path.join(star_results_dir, workspace_name)
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(ds_results_dir, exist_ok=True)

    with tarfile.open(tar_path, 'r') as tar:
        top_dirs = {n.split('/')[0] for n in tar.getnames() if n.strip()}
    extracted_folder_name = min(top_dirs, key=len)
    name_parts = extracted_folder_name.split('.')
    field_name = name_parts[1] if len(name_parts) > 1 else "UnknownField"
    clean_ms_name = f"SB{sbid}.{field_name}.beam{beam}.ms"
    subtracted_ms_name = f"SB{sbid}.{field_name}.beam{beam}.subtracted.ms"
    final_ds_name = f"{clean_hostname}_SB{sbid}_beam{beam}.ds"
    wsclean_model_dir_name = f"wsclean_model_{clean_hostname}_SB{sbid}_beam{beam}"
    model_dir = os.path.join(workspace_dir, wsclean_model_dir_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"缺少已有 WSClean 模型目录: {model_dir}")

    # 清理旧产物，明确保留已有 wsclean_model 目录。
    for f in glob.glob(os.path.join(workspace_dir, "SB*.*.ms")):
        shutil.rmtree(f, ignore_errors=True)
    for f in glob.glob(os.path.join(workspace_dir, "*.ds")):
        os.remove(f)
    ds_path = os.path.join(ds_results_dir, final_ds_name)
    if os.path.exists(ds_path):
        os.remove(ds_path)

    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=workspace_dir)
    os.rename(os.path.join(workspace_dir, extracted_folder_name), os.path.join(workspace_dir, clean_ms_name))
    ms_path = os.path.join(workspace_dir, clean_ms_name)
    if os.path.exists(os.path.join(ms_path, "FIELD_OLD")):
        print("  [重置] tar 包内 MS 含旧版预处理残留，彻底重置后重跑...")
        run_cmd(f"fix_ms_dir --restore {clean_ms_name}", cwd=workspace_dir)
        shutil.rmtree(os.path.join(ms_path, "FIELD_OLD"), ignore_errors=True)
        shutil.rmtree(os.path.join(ms_path, "FEED_OLD"), ignore_errors=True)
        import casacore.tables as _pt
        _t = _pt.table(ms_path, readonly=False, ack=False)
        if "CORRECTED_DATA" in _t.colnames():
            _t.remove_cols("CORRECTED_DATA")
            print("  [重置] 已删除旧 CORRECTED_DATA 列")
        _t.close()
    run_cmd(f"dstools-askap-preprocess {clean_ms_name}", cwd=workspace_dir)
    run_cmd(
        f"dstools-insert-model -p {corr_ra} {corr_dec} -r {MASK_RADIUS} {wsclean_model_dir_name} {clean_ms_name}",
        cwd=workspace_dir)
    run_cmd(f"dstools-subtract-model -S {clean_ms_name}", cwd=workspace_dir)
    run_cmd(
        f"dstools-extract-ds -p {corr_ra} {corr_dec} -v -u 500 -B {subtracted_ms_name} {final_ds_name}",
        cwd=workspace_dir)

    shutil.move(os.path.join(workspace_dir, final_ds_name), os.path.join(ds_results_dir, final_ds_name))
    print(f"完成: {os.path.join(ds_results_dir, final_ds_name)}")
    print(f"dstools-plot-ds -d {os.path.join(ds_results_dir, final_ds_name)} -l -s IV -t 1 -f 1")


if __name__ == "__main__":
    for tar_path in TAR_PATHS:
        try:
            process_tar(tar_path)
        except Exception as exc:
            print(f"[失败] {tar_path}: {exc}")
