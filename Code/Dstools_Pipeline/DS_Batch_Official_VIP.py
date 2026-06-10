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
from concurrent.futures import ThreadPoolExecutor, as_completed
import casacore.tables as pt

warnings.filterwarnings('ignore')

# --- 独立的 "VIP" 日志配置 (防止与主程序写冲突) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [VIP-%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        # 独立的日志文件！
        logging.FileHandler("pipeline_execution_vip.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("ASKAP_Stellar_Pipeline_VIP")


# --- 路径与参数 ---
def project_path(relative_path: str) -> str:
    current = os.path.abspath(os.path.dirname(__file__))
    while not (os.path.isdir(os.path.join(current, 'Code')) and os.path.isdir(os.path.join(current, 'Processed_Data'))):
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.join(os.getcwd(), relative_path)
        current = parent
    return os.path.join(current, relative_path)


ASKAP_CATALOGUE_CSV: str = project_path('Processed_Data/Catalogue/01.askap_catalogue.csv')
INPUT_CSV: str = project_path('Processed_Data/Catalogue/02.final_confirmed_stars_direct_1.csv')
PIPELINE_RESULTS_BASE: str = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet/Pipeline_Results"


#  VIP 控制面板
# 在这里填入你想优先处理的高质量包的绝对路径
VIP_TAR_FILES: List[str] = ['/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet/Downloading_Data/ms_data/2MASS_J01033563-5515561_A/59565_scienceData.EMU_0054-55.SB59565.EMU_0054-55.beam22_averaged_cal.leakage.ms.tar',
                            '/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet/Downloading_Data/ms_data/AB_Pic/51853_scienceData.EMU_0610-60.SB51853.EMU_0610-60.beam34_averaged_cal.leakage.ms.tar',
                            '/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet/Downloading_Data/ms_data/AB_Pic/61949_scienceData.EMU_0618-55.SB61949.EMU_0618-55.beam03_averaged_cal.leakage.ms.tar',
                            '/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet/Downloading_Data/ms_data/Proxima_Cen/77270_scienceData.EMU_1424-60.SB77270.EMU_1424-60.beam03_averaged_cal.leakage.ms.tar',
                            '/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet/Downloading_Data/ms_data/PZ_Tel/79074_scienceData.FLASH_153.SB79074.FLASH_153.beam14_averaged_cal.leakage.ms.tar',
                            '/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet/Downloading_Data/ms_data/2MASS_J01033563-5515561_A/66827_scienceData.WALLABY_0051-53A.SB66827.WALLABY_0051-53A.beam10_averaged_cal.leakage.ms.tar',
                            '/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet/Downloading_Data/ms_data/2MASS_J01033563-5515561_A/68040_scienceData.WALLABY_0051-53B.SB68040.WALLABY_0051-53B.beam10_averaged_cal.leakage.ms.tar'
    # 示例: '/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet/Downloading_Data/ms_data/GJ_4274/scienceData.VAST...beam00.ms.tar',
]

MASK_RADIUS: int = 15  # 掩模半径（角秒）
MAX_CONCURRENT_MS: int = 1  # VIP 通道的并发数量
WSCLEAN_THREADS: int = 10  # VIP 专属分配的核心数 (注意不要和主程序加起来超过服务器总核)


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
        subprocess.run(cmd_str, shell=True, check=True, cwd=cwd, executable='/bin/bash', env=custom_env)
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {cmd_str}")
        raise e


# --- 核心数据处理逻辑 (与主程序完全一致) ---
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
        if not top_dirs: raise ValueError(f"Tar 包结构异常: {tar_filename}")
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

    logger.info(f"开启 VIP 处理 -> 源: {clean_hostname} | SBID: {sbid} | Beam: {beam}")

    if os.path.exists(os.path.join(ds_results_dir, final_ds_name)):
        logger.info(f" [跳过] 成果 {final_ds_name} 已存在。")
        return

    # 坐标计算逻辑
    pmra_val = star_meta.get('sy_pmra', star_meta.get('pmra', 0.0))
    pmdec_val = star_meta.get('sy_pmdec', star_meta.get('pmdec', 0.0))
    pmra = 0.0 if pd.isna(pmra_val) else float(pmra_val)
    pmdec = 0.0 if pd.isna(pmdec_val) else float(pmdec_val)
    plx_val = star_meta.get('sy_plx', star_meta.get('plx', 10.0))
    plx = 10.0 if pd.isna(plx_val) or float(plx_val) <= 0 else float(plx_val)

    star_j2015 = SkyCoord(ra=star_meta['ra'] * u.deg, dec=star_meta['dec'] * u.deg,
                          pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                          distance=(1000 / plx) * u.pc, frame='icrs', obstime=Time('J2015.5'))
    obs_time = Time(obs_mjd, format='mjd')
    star_at_obs = star_j2015.apply_space_motion(new_obstime=obs_time)
    corr_ra, corr_dec = round(star_at_obs.ra.deg, 7), round(star_at_obs.dec.deg, 7)
    logger.info(f"坐标推演 J2015.5 -> {obs_time.datetime.date()}: RA {corr_ra}, DEC {corr_dec}")

    existing_mfs_images = glob.glob(os.path.join(workspace_dir, "*wsclean_model*", "*-MFS-image.fits"))
    wsclean_done = os.path.exists(wsclean_sentinel) and len(existing_mfs_images) > 0

    if not wsclean_done:
        logger.warning(f" WSClean 模型未就绪，开启专属 {WSCLEAN_THREADS} 核建图...")
        if os.path.exists(workspace_dir): shutil.rmtree(workspace_dir)
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
    if os.path.exists(os.path.join(workspace_dir, final_ds_name)): os.remove(os.path.join(workspace_dir, final_ds_name))
    run_cmd(f"dstools-extract-ds -p {corr_ra} {corr_dec} -v -u 500 -B {subtracted_ms_name} {final_ds_name}",
            cwd=workspace_dir)
    shutil.move(os.path.join(workspace_dir, final_ds_name), os.path.join(ds_results_dir, final_ds_name))
    logger.info(f" VIP 任务完成，结果已保存！")


# --- 调度程序 ---
def main() -> None:
    warnings.filterwarnings('ignore', category=UserWarning)
    logger.info("==========================================")
    logger.info(" 启动 ASKAP Stellar [ VIP 专属插队通道]")
    logger.info("==========================================")

    if not VIP_TAR_FILES:
        logger.info("️ VIP 队列为空，没有指定任何数据文件。程序退出。")
        return

    if not os.path.exists(INPUT_CSV) or not os.path.exists(ASKAP_CATALOGUE_CSV):
        logger.critical("项目元数据表丢失！")
        return

    # 1. 准备 MJD 数据字典
    obs_df = pd.read_csv(ASKAP_CATALOGUE_CSV)
    obs_df.columns = obs_df.columns.str.strip()
    obs_df['sbid_clean'] = obs_df['obs_id'].apply(
        lambda x: str(int(re.search(r'(\d+)', str(x)).group(1))) if re.search(r'(\d+)', str(x)) else None)
    sbid_to_mjd = obs_df.dropna(subset=['sbid_clean']).drop_duplicates(subset=['sbid_clean']).set_index('sbid_clean')[
        't_min'].to_dict()

    # 2. 准备星表字典
    stars_df = pd.read_csv(INPUT_CSV)
    stars_df.columns = stars_df.columns.str.strip()
    stars_df['hostname_clean'] = stars_df['hostname'].astype(str).str.strip().str.replace(' ', '_')
    star_catalog_dict = stars_df.drop_duplicates(subset=['hostname_clean']).set_index('hostname_clean').to_dict('index')

    logger.info(f"已捕获 {len(VIP_TAR_FILES)} 个最高优先级数据包，准备强行空降处理...")

    # 3. 单包运行逻辑
    def _run_vip_single(tar_path):
        if not os.path.exists(tar_path):
            logger.error(f"找不到指定的 VIP 文件: {tar_path}，已跳过。")
            return

        # 解析该包属于哪个天体
        dir_name = os.path.basename(os.path.dirname(tar_path))
        norm_dir = re.sub(r'[^a-zA-Z0-9]', '', dir_name).lower()

        matched_key = None
        for k in star_catalog_dict.keys():
            norm_k = re.sub(r'[^a-zA-Z0-9]', '', k).lower()
            if norm_k in norm_dir or norm_dir in norm_k:
                matched_key = k
                break

        if not matched_key:
            logger.error(f"无法在星表中匹配到该包所在的目录 {dir_name}，无法推算坐标，已跳过。")
            return

        sbid, beam = extract_sbid_and_beam(os.path.basename(tar_path))
        if not sbid or not beam: return
        if sbid not in sbid_to_mjd: return

        clean_hostname = matched_key
        star_meta = star_catalog_dict[clean_hostname]
        obs_mjd = sbid_to_mjd[sbid]

        # 直接触发流转
        process_single_tar(tar_path, clean_hostname, star_meta, sbid, beam, obs_mjd)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_MS) as executor:
        future_to_tar = {executor.submit(_run_vip_single, tar_path): tar_path for tar_path in VIP_TAR_FILES}

        for future in as_completed(future_to_tar):
            tar_path = future_to_tar[future]
            try:
                future.result()
            except KeyboardInterrupt:
                logger.warning("接收到中断信号 (Ctrl+C)，强制关闭 VIP 通道。")
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as e:
                logger.exception(f"  处理 VIP 包 {os.path.basename(tar_path)} 失败: {e}")
                continue

    logger.info("全部 VIP 指定数据文件处理完毕！")


if __name__ == "__main__":
    main()