"""仅处理交叉匹配 SBID，使用对应 ASKAP 射电坐标创建服务器端动态谱。"""

import os
import re
import glob
import sys
import shutil
import tarfile
import logging
import math
import json
from typing import List
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from science_utils import (  # noqa: E402
    extract_sbid_and_beam,
    run_cmd,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


logger = logging.getLogger("ASKAP_Stellar_Pipeline_Matched_SBID")


# --- 路径与参数 ---


INPUT_CSV = PROJECT_ROOT / "Processed_Data" / "Catalogue" / "02.final_confirmed_stars_direct.csv"
DRY_RUN: bool = False  # True 只列出符合交叉匹配记录的本地任务，不解包、不运行 dstools。

CASDA_BASE_PATH: str = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Data/Ms_Data"
PIPELINE_RESULTS_BASE: str = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Result/DS"

# --- 控制参数 ---
TARGET_SOURCES: List[str] = ['2MASS J01033563-5515561 A','AB Pic','AF Lep','AU Mic', 'COCONUTS-2 A','GJ 896 A','GJ 4274','HD 180902','HD 95086','Proxima Cen','PZ Tel','GJ 229','GJ 3323','HD 41004 A','ROXs 42 B','TOI-2992','TOI-2458']
MASK_RADIUS: int = 15     # 掩模半径（角秒）
MAX_CONCURRENT_MS: int = 7     # 同时并行处理的 MS 压缩包数量
WSCLEAN_THREADS: int = 8     # 每个 WSClean 进程分配的线程数

# --- 管线主流程：数据准备、线程调度与单包处理 ---
def main() -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if not DRY_RUN:
        handlers.append(logging.FileHandler("pipeline_execution_matched_sbid.log", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=handlers)
    logger.info("ASKAP 匹配 SBID / 官方射电坐标管线启动；DRY_RUN=%s", DRY_RUN)

    if not os.path.isfile(INPUT_CSV):
        logger.critical("交叉匹配表不存在：%s", INPUT_CSV)
        return
    stars_df = pd.read_csv(INPUT_CSV, dtype={"sbid_clean": "string"})
    stars_df.columns = stars_df.columns.str.strip()
    required = {"hostname", "sbid_clean", "col_ra_deg_cont", "col_dec_deg_cont"}
    missing = sorted(required - set(stars_df.columns))
    if missing:
        raise ValueError(f"交叉匹配表缺少必需列：{', '.join(missing)}")

    # SBID 只接受完整合法表示；禁止从任意文本里提取第一个数字。
    valid_rows = []
    for index, row in stars_df.iterrows():
        hostname = row["hostname"]
        sb_text = str(row["sbid_clean"]).strip()
        sb_match = re.fullmatch(r"(?:SB|ASKAP-)?(\d+)(?:\.0+)?", sb_text, re.IGNORECASE)
        if pd.isna(hostname) or not str(hostname).strip() or not sb_match:
            logger.warning("[INVALID_MATCH] CSV 行 %s：源名或 SBID 无效", index + 2)
            continue
        try:
            ra_deg = float(row["col_ra_deg_cont"])
            dec_deg = float(row["col_dec_deg_cont"])
        except (TypeError, ValueError):
            logger.warning("[INVALID_MATCH] %s / %s：射电坐标不是数值", hostname, sb_text)
            continue
        if not (math.isfinite(ra_deg) and math.isfinite(dec_deg)
                and 0 <= ra_deg < 360 and -90 <= dec_deg <= 90):
            logger.warning("[INVALID_MATCH] %s / %s：射电坐标缺失或超范围", hostname, sb_text)
            continue
        sbid = str(int(sb_match.group(1)))
        if int(sbid) <= 0:
            logger.warning("[INVALID_MATCH] %s：SBID 必须为正整数", hostname)
            continue
        record = row.to_dict()
        record.update(hostname_clean=str(hostname).strip().replace(" ", "_"),
                      sbid_clean=sbid, col_ra_deg_cont=ra_deg, col_dec_deg_cont=dec_deg)
        valid_rows.append(record)

    # 每个宿主的不同 SBID 保留独立坐标；同一观测的多行行星记录只处理一次。
    matches_by_host = {}
    if valid_rows:
        validated = pd.DataFrame(valid_rows)
        identity_columns = ["col_ra_deg_cont", "col_dec_deg_cont"]
        if "col_component_id" in validated:
            identity_columns.append("col_component_id")
        for (host, sbid), group in validated.groupby(["hostname_clean", "sbid_clean"], sort=False):
            if len(group[identity_columns].drop_duplicates()) != 1:
                logger.error("[AMBIGUOUS_MATCH] %s / SB%s：多个射电分量或不同坐标，跳过此组合", host, sbid)
                continue
            matches_by_host.setdefault(host, {})[sbid] = group.iloc[0].to_dict()

    host_keys = {}
    for host in matches_by_host:
        normalized = re.sub(r"[^a-zA-Z0-9]", "", host).lower()
        if not normalized or normalized in host_keys:
            raise ValueError(f"源名规范化后存在歧义：{host}")
        host_keys[normalized] = host
    target_keys = {re.sub(r"[^a-zA-Z0-9]", "", str(name)).lower()
                   for name in TARGET_SOURCES}

    # 单包 worker：保留线程池所需的回调边界，处理逻辑直接内联于此。
    def _run_single(tar_path, clean_hostname, match_meta):
        sbid, beam = extract_sbid_and_beam(os.path.basename(tar_path))
        if sbid != match_meta["sbid_clean"] or beam is None:
            raise ValueError(f"压缩包与匹配记录不一致：{tar_path}")
        corr_ra = float(match_meta["col_ra_deg_cont"])
        corr_dec = float(match_meta["col_dec_deg_cont"])
        logger.info("[QUEUED] %s | SB%s | beam%s | ASKAP RA=%.12f DEC=%.12f deg | %s",
                    clean_hostname, sbid, beam, corr_ra, corr_dec, tar_path)
        if DRY_RUN:
            return
        tar_filename = os.path.basename(tar_path)

        star_results_dir = os.path.join(PIPELINE_RESULTS_BASE, clean_hostname)
        os.makedirs(star_results_dir, exist_ok=True)
        ds_results_dir = os.path.join(star_results_dir, "DS_Results")
        os.makedirs(ds_results_dir, exist_ok=True)

        workspace_name = f"{clean_hostname}_SB{sbid}_beam{beam}_askap_workspace"
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
        final_ds_name = f"{clean_hostname}_SB{sbid}_beam{beam}_askap.ds"

        wsclean_model_dir_name = f"wsclean_model_{clean_hostname}_SB{sbid}_beam{beam}"
        wsclean_model_full_path = os.path.join(workspace_dir, wsclean_model_dir_name)

        wsclean_sentinel = os.path.join(workspace_dir, ".wsclean_done")
        subtraction_sentinel = os.path.join(workspace_dir, ".subtraction_done")

        logger.info(f"开始处理 -> 源: {clean_hostname} | SBID: {sbid} | Beam: {beam}")

        expected_ds_path = os.path.join(ds_results_dir, final_ds_name)
        # 插模掩模和提取相位中心依赖坐标；只有相同坐标的产物才可恢复。
        coordinate_meta = {
            "coordinate_source": "ASKAP_catalogue",
            "coordinate_unit": "deg",
            "hostname": clean_hostname,
            "sbid": sbid,
            "beam": beam,
            "ra_deg": corr_ra,
            "dec_deg": corr_dec,
            "mask_radius_arcsec": MASK_RADIUS,
            "minimum_baseline_m": 500,
            "average_baselines": False,
            "input_archive": os.path.abspath(tar_path),
        }
        workspace_meta_path = os.path.join(workspace_dir, ".askap_coordinates.json")
        ds_meta_path = expected_ds_path + ".coordinates.json"
        if os.path.isfile(expected_ds_path) and os.path.getsize(expected_ds_path) > 0:
            recorded = None
            if os.path.isfile(ds_meta_path):
                with open(ds_meta_path, encoding="utf-8") as stream:
                    recorded = json.load(stream)
            if recorded != coordinate_meta:
                raise RuntimeError(
                    f"[COORDINATE_MISMATCH] {expected_ds_path}：旧坐标或缺少坐标记录；"
                    "请使用另一个 PIPELINE_RESULTS_BASE，保留旧成果。"
                )
            logger.info("[跳过] %s 已存在且 ASKAP 坐标记录一致", final_ds_name)
            return
        if os.path.isdir(workspace_dir) and os.listdir(workspace_dir):
            recorded = None
            if os.path.isfile(workspace_meta_path):
                with open(workspace_meta_path, encoding="utf-8") as stream:
                    recorded = json.load(stream)
            if recorded != coordinate_meta:
                raise RuntimeError(
                    f"[COORDINATE_MISMATCH] {workspace_dir}：工作空间坐标记录不一致；"
                    "请使用另一个 PIPELINE_RESULTS_BASE。"
                )

        existing_mfs_images = glob.glob(os.path.join(workspace_dir, "*wsclean_model*", "*-MFS-*"))
        wsclean_done = os.path.exists(wsclean_sentinel) and len(existing_mfs_images) > 0

        if not wsclean_done:
            logger.warning(f" WSClean 模型未就绪，开始建图 ({WSCLEAN_THREADS} 线程)...")
            if os.path.exists(workspace_dir):
                shutil.rmtree(workspace_dir)
            os.makedirs(workspace_dir, exist_ok=True)
            with open(workspace_meta_path, "w", encoding="utf-8") as stream:
                json.dump(coordinate_meta, stream, ensure_ascii=False, indent=2)

            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=workspace_dir)
            os.rename(
                os.path.join(workspace_dir, extracted_folder_name),
                os.path.join(workspace_dir, clean_ms_name)
            )

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
            logger.info(f"--> [STEP 3] 插入模型并写入 MODEL_DATA (-p {corr_ra} {corr_dec} -r {MASK_RADIUS})...")
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

        created_ds_path = os.path.join(workspace_dir, final_ds_name)
        if not os.path.isfile(created_ds_path) or os.path.getsize(created_ds_path) == 0:
            raise RuntimeError(f"dstools 未生成非空动态谱：{created_ds_path}")
        shutil.move(created_ds_path, expected_ds_path)
        with open(ds_meta_path + ".tmp", "w", encoding="utf-8") as stream:
            json.dump(coordinate_meta, stream, ensure_ascii=False, indent=2)
        os.replace(ds_meta_path + ".tmp", ds_meta_path)
        logger.info("完成，动态谱及 ASKAP 坐标记录已保存：%s", expected_ds_path)

    # 源目录按完整规范化名称匹配，再按该源 CSV 中的 SBID 筛选压缩包。
    folder_index = {}
    for folder in sorted(glob.glob(os.path.join(CASDA_BASE_PATH, "*"))):
        if os.path.isdir(folder):
            key = re.sub(r"[^a-zA-Z0-9]", "", os.path.basename(folder)).lower()
            folder_index.setdefault(key, []).append(folder)

    for norm_host, clean_hostname in host_keys.items():
        if TARGET_SOURCES and norm_host not in target_keys:
            continue
        match_rows = matches_by_host[clean_hostname]
        folders = folder_index.get(norm_host, [])
        if not folders:
            logger.warning("[NO_MS_FOLDER] %s：缺少源目录；期望 SBID=%s",
                           clean_hostname, ",".join(sorted(match_rows)))
            continue
        if len(folders) != 1:
            logger.error("[AMBIGUOUS_MS] %s：多个同名规范化源目录，跳过：%s", clean_hostname, folders)
            continue
        candidates = {}
        for tar_path in sorted(glob.glob(os.path.join(folders[0], "*.tar"))):
            sbid, beam = extract_sbid_and_beam(os.path.basename(tar_path))
            if sbid is None or beam is None:
                logger.warning("[INVALID_MS_NAME] 无法解析 SBID/beam：%s", tar_path)
                continue
            if sbid not in match_rows:
                logger.info("[UNMATCHED_SBID] %s / SB%s：不在该源匹配表中，跳过", clean_hostname, sbid)
                continue
            candidates.setdefault((sbid, beam), []).append(tar_path)
        jobs = []
        for (sbid, beam), paths in candidates.items():
            if len(paths) != 1:
                logger.error("[AMBIGUOUS_MS] %s / SB%s / beam%s：多个包，跳过：%s",
                             clean_hostname, sbid, beam, paths)
                continue
            jobs.append((paths[0], match_rows[sbid]))
        found_sbids = {meta["sbid_clean"] for _, meta in jobs}
        for sbid in sorted(set(match_rows) - found_sbids):
            logger.warning("[MISSING_MS] %s / SB%s：没有可唯一选择的本地 MS 压缩包",
                           clean_hostname, sbid)
        if not jobs:
            continue
        logger.info("处理源 %s：%s 个匹配 MS 包；并发=%s",
                    clean_hostname, len(jobs), MAX_CONCURRENT_MS)
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_MS) as executor:
            future_to_tar = {
                executor.submit(_run_single, path, clean_hostname, meta): path
                for path, meta in jobs
            }
            for future in as_completed(future_to_tar):
                tar_path = future_to_tar[future]
                try:
                    future.result()
                except KeyboardInterrupt:
                    logger.warning("接收到中断信号 (Ctrl+C)，退出。")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as error:
                    logger.exception("处理包 %s 失败：%s", os.path.basename(tar_path), error)

if __name__ == "__main__":
    main()
