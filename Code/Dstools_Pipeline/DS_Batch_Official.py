import os
import re
import glob
import sys
import shutil
import tarfile
import subprocess
import logging
import warnings
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import casacore.tables as pt

warnings.filterwarnings('ignore')

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline_execution_official.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("ASKAP_Stellar_Pipeline_Official")


# --- 路径与参数 ---
def project_path(relative_path: str) -> str:
    """自适应项目根目录定位"""
    current = os.path.abspath(os.path.dirname(__file__))
    while not (os.path.isdir(os.path.join(current, 'Code')) and os.path.isdir(os.path.join(current, 'Processed_Data'))):
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.join(os.getcwd(), relative_path)
        current = parent
    return os.path.join(current, relative_path)


ASKAP_CATALOGUE_CSV: str = project_path('Processed_Data/Catalogue/01.askap_catalogue.csv')
INPUT_CSV: str = project_path('Processed_Data/Catalogue/02.final_confirmed_stars_direct_1.csv')

CASDA_BASE_PATH: str = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet/Downloading_Data/ms_data"
PIPELINE_RESULTS_BASE: str = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet/Pipeline_Results"

# --- 控制参数 ---
TARGET_SOURCES: List[str] = ['2MASS J01033563-5515561 A','GJ 4274','AU Mic']

# 掩模半径（角秒）
MASK_RADIUS: int = 15
WSCLEAN_THREADS: int = 36


# --- 辅助函数 ---
def extract_sbid_and_beam(filename: str) -> Tuple[Optional[str], Optional[str]]:
    sb_match = re.search(r'SB(\d+)', filename, re.IGNORECASE)
    beam_match = re.search(r'beam(\d+)', filename, re.IGNORECASE)
    sbid = str(int(sb_match.group(1))) if sb_match else None
    beam = str(int(beam_match.group(1))) if beam_match else None
    return sbid, beam


def run_cmd(cmd_str: str, cwd: str) -> None:
    conda_bin_dir = os.path.dirname(sys.executable)
    custom_env = os.environ.copy()
    custom_env["PATH"] = conda_bin_dir + os.pathsep + custom_env.get("PATH", "")

    try:
        # 使用 shell=True 以兼容 C++ 底层库文件流处理
        subprocess.run(cmd_str, shell=True, check=True, cwd=cwd, executable='/bin/bash', env=custom_env)
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {cmd_str}")
        raise e


# --- 管线主流程 ---
def process_single_tar(tar_path: str, clean_hostname: str, star_meta: Dict[str, Any], sbid: str, beam: str,
                       obs_mjd: float) -> None:
    tar_filename = os.path.basename(tar_path)

    star_results_dir = os.path.join(PIPELINE_RESULTS_BASE, clean_hostname)
    os.makedirs(star_results_dir, exist_ok=True)
    ds_results_dir = os.path.join(star_results_dir, "DS_Results")
    os.makedirs(ds_results_dir, exist_ok=True)

    workspace_name = f"{clean_hostname}_SB{sbid}_beam{beam}_workspace"
    workspace_dir = os.path.join(star_results_dir, workspace_name)

    with tarfile.open(tar_path, 'r') as tar:
        top_dirs = {n.split('/')[0] for n in tar.getnames() if n.strip()}
        if not top_dirs:
            raise ValueError(f"Tar 包结构异常: {tar_filename}")
        extracted_folder_name = min(top_dirs, key=len)

    name_parts = extracted_folder_name.split('.')
    field_name = name_parts[1] if len(name_parts) > 1 else "UnknownField"

    clean_ms_name = f"SB{sbid}.{field_name}.beam{beam}.ms"
    subtracted_ms_name = f"SB{sbid}.{field_name}.beam{beam}.subtracted.ms"
    subtracted_ms_path = os.path.join(workspace_dir, subtracted_ms_name)
    final_ds_name = f"{clean_hostname}_SB{sbid}_beam{beam}.ds"

    wsclean_model_dir_name = f"wsclean_model_{clean_hostname}_SB{sbid}_beam{beam}"
    wsclean_model_full_path = os.path.join(workspace_dir, wsclean_model_dir_name)

    wsclean_sentinel = os.path.join(workspace_dir, ".wsclean_done")
    subtraction_sentinel = os.path.join(workspace_dir, ".subtraction_done")

    logger.info(f"开始处理 -> 源: {clean_hostname} | SBID: {sbid} | Beam: {beam}")

    expected_ds_path = os.path.join(ds_results_dir, final_ds_name)
    if os.path.exists(expected_ds_path):
        logger.info(f" [跳过] {final_ds_name} 已存在。")
        return

    # ==========================================================================
    # 坐标计算逻辑
    # ==========================================================================
    pmra_val = star_meta.get('sy_pmra', star_meta.get('pmra', 0.0))
    pmdec_val = star_meta.get('sy_pmdec', star_meta.get('pmdec', 0.0))
    pmra = 0.0 if pd.isna(pmra_val) else float(pmra_val)
    pmdec = 0.0 if pd.isna(pmdec_val) else float(pmdec_val)

    plx_val = star_meta.get('sy_plx', star_meta.get('plx', 10.0))
    plx = 10.0 if pd.isna(plx_val) or float(plx_val) <= 0 else float(plx_val)

    star_j2015 = SkyCoord(
        ra=star_meta['ra'] * u.deg,
        dec=star_meta['dec'] * u.deg,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        distance=(1000 / plx) * u.pc,
        frame='icrs',
        obstime=Time('J2015.5')
    )
    obs_time = Time(obs_mjd, format='mjd')
    star_at_obs = star_j2015.apply_space_motion(new_obstime=obs_time)
    corr_ra = round(star_at_obs.ra.deg, 7)
    corr_dec = round(star_at_obs.dec.deg, 7)
    logger.info(
        f"坐标计算 J2015.5 -> {obs_time.datetime.date()}: RA {corr_ra}, DEC {corr_dec}")

    existing_mfs_images = glob.glob(os.path.join(workspace_dir, "*wsclean_model*", "*-MFS-image.fits"))
    wsclean_done = os.path.exists(wsclean_sentinel) and len(existing_mfs_images) > 0

    if not wsclean_done:
        logger.warning(f" WSClean 模型未就绪，开始建图 ({WSCLEAN_THREADS} 线程)...")
        if os.path.exists(workspace_dir):
            shutil.rmtree(workspace_dir)
        os.makedirs(workspace_dir, exist_ok=True)

        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=workspace_dir)

        t = pt.table(os.path.join(workspace_dir, extracted_folder_name))
        t.copy(os.path.join(workspace_dir, clean_ms_name), deep=True, valuecopy=True)
        t.close()

        logger.info("执行预处理 (dstools-askap-preprocess)...")
        run_cmd(f"dstools-askap-preprocess {clean_ms_name}", cwd=workspace_dir)

        logger.info(f"执行 dstools-create-model 建模...")
        os.makedirs(wsclean_model_full_path, exist_ok=True)
        dstools_cmd = (
            f"dstools-create-model -I 8192 -c 2.5 -N 1000000 -g 0.8 -r 0.5 "
            f"-t 5 -m 6 -S --multiscale-scale-bias 0.7 --multiscale-max-scales 8 "
            f"-f 8 --deconvolution-channels 8 -n 3 -j {WSCLEAN_THREADS} "
            f"-o {wsclean_model_dir_name} --name wsclean --temp-dir {wsclean_model_dir_name} {clean_ms_name}"
        )
        run_cmd(dstools_cmd, cwd=workspace_dir)

        with open(wsclean_sentinel, 'w', encoding='utf-8') as f:
            f.write("WSCLEAN_SUCCESS")
    else:
        detected_model_path = os.path.dirname(existing_mfs_images[0])
        wsclean_model_dir_name = os.path.basename(detected_model_path)
        logger.info(f" [恢复] 检测到已有模型 {wsclean_model_dir_name}，跳过建图。")

    subtraction_done = os.path.exists(subtracted_ms_path) and os.path.exists(subtraction_sentinel)

    if not subtraction_done:
        logger.info(f"--> [STEP 3] 插入模型 (-p {corr_ra} {corr_dec} -r {MASK_RADIUS})...")
        run_cmd(
            f"dstools-insert-model -p {corr_ra} {corr_dec} -r {MASK_RADIUS} {wsclean_model_dir_name} {clean_ms_name}",
            cwd=workspace_dir)

        logger.info(f"--> [STEP 4] 执行背景减除 (dstools-subtract-model)...")
        run_cmd(f"dstools-subtract-model -S {clean_ms_name}", cwd=workspace_dir)

        with open(subtraction_sentinel, 'w', encoding='utf-8') as f:
            f.write("SUBTRACTION_SUCCESS")
        logger.info(f"背景减除完成: {subtracted_ms_name}")
    else:
        logger.info(f" [恢复] 背景减除数据集已就绪，跳过 subtract。")

    logger.info(f"--> [STEP 5] 提取动态谱 (-u 500 -B)...")
    if os.path.exists(os.path.join(workspace_dir, final_ds_name)):
        os.remove(os.path.join(workspace_dir, final_ds_name))

    run_cmd(f"dstools-extract-ds -p {corr_ra} {corr_dec} -v -u 500 -B {subtracted_ms_name} {final_ds_name}",
            cwd=workspace_dir)

    shutil.move(os.path.join(workspace_dir, final_ds_name), os.path.join(ds_results_dir, final_ds_name))
    logger.info(f"✅ 完成，结果已保存。")


def process_entire_source(clean_hostname: str, tar_files: List[str], star_meta: Dict[str, Any],
                          sbid_to_mjd: Dict[str, float]) -> str:
    logger.info(f"==========================================")
    logger.info(f"处理源: {clean_hostname}，共 {len(tar_files)} 个包")
    logger.info(f"==========================================")
    for tar_path in tar_files:
        sbid, beam = extract_sbid_and_beam(os.path.basename(tar_path))
        if not sbid or not beam: continue
        if sbid not in sbid_to_mjd:
            logger.warning(f"无法在 01 表中找到 SBID {sbid} 的对应观测时间，跳过该ms数据文件。")
            continue
        obs_mjd = sbid_to_mjd[sbid]
        try:
            process_single_tar(tar_path, clean_hostname, star_meta, sbid, beam, obs_mjd)
        except KeyboardInterrupt:
            logger.warning("接收到中断信号 (Ctrl+C)，退出。")
            raise
        except Exception as e:
            logger.exception(f"  处理失败: {e}")
            continue
    return clean_hostname


def main() -> None:
    warnings.filterwarnings('ignore', category=UserWarning)
    logger.info("ASKAP Stellar Pipeline 启动")

    if not os.path.exists(INPUT_CSV) or not os.path.exists(ASKAP_CATALOGUE_CSV):
        logger.critical("项目元数据表丢失！")
        return

    obs_df = pd.read_csv(ASKAP_CATALOGUE_CSV)
    obs_df.columns = obs_df.columns.str.strip()
    obs_df['sbid_clean'] = obs_df['obs_id'].apply(
        lambda x: str(int(re.search(r'(\d+)', str(x)).group(1))) if re.search(r'(\d+)', str(x)) else None)
    sbid_to_mjd = obs_df.dropna(subset=['sbid_clean']).drop_duplicates(subset=['sbid_clean']).set_index('sbid_clean')[
        't_min'].to_dict()

    stars_df = pd.read_csv(INPUT_CSV)
    stars_df.columns = stars_df.columns.str.strip()
    stars_df['hostname_clean'] = stars_df['hostname'].astype(str).str.strip().str.replace(' ', '_')
    star_catalog_dict = stars_df.drop_duplicates(subset=['hostname_clean']).set_index('hostname_clean').to_dict('index')

    star_folders = glob.glob(os.path.join(CASDA_BASE_PATH, '*'))

    for folder in star_folders:
        if not os.path.isdir(folder): continue
        dir_name = os.path.basename(folder)

        if TARGET_SOURCES:
            is_matched = False
            for target in TARGET_SOURCES:
                norm_target = str(target).strip().replace(' ', '_').lower()
                norm_dir = str(dir_name).strip().replace(' ', '_').lower()
                if norm_target in norm_dir or norm_dir in norm_target:
                    is_matched = True
                    break
            if not is_matched: continue

        dir_name = os.path.basename(folder)
        # 用正则剔除文件夹名字里的所有下划线、空格、横杠，只保留纯字母和数字，并转小写
        norm_dir = re.sub(r'[^a-zA-Z0-9]', '', dir_name).lower()

        # 2. 目标源过滤
        if TARGET_SOURCES:
            is_matched = False
            for target in TARGET_SOURCES:
                norm_target = re.sub(r'[^a-zA-Z0-9]', '', target).lower()
                if norm_target in norm_dir or norm_dir in norm_target:
                    is_matched = True
                    break
            if not is_matched:
                continue

        # 3. 星表字典匹配
        matched_key = None
        for k in star_catalog_dict.keys():
            norm_k = re.sub(r'[^a-zA-Z0-9]', '', k).lower()
            if norm_k in norm_dir or norm_dir in norm_k:
                matched_key = k
                break

        if not matched_key:
            logger.warning(f"无法在星表中匹配到 {dir_name}，跳过。")
            continue

        clean_hostname = matched_key
        star_meta = star_catalog_dict[clean_hostname]
        tar_files = glob.glob(os.path.join(folder, '*.tar'))

        if not tar_files: continue

        process_entire_source(clean_hostname, tar_files, star_meta, sbid_to_mjd)


if __name__ == "__main__":
    main()