"""Search, validate, and download TESS light-curve products for configured targets."""

import os
import re
import glob
import time
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import lightkurve as lk
from tqdm.auto import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from science_utils import (  # noqa: E402
    list_fits_files,
    retry_with_exponential_backoff,
)

# ========================
# 配置：修改这里的默认值
# ========================
DEFAULT_TARGETS = ["2MASS J01033563-5515561 A"]
DEFAULT_DOWNLOAD_DIR = "/TESS_Data"
# DEFAULT_DOWNLOAD_DIR = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Data/TESS_Data"
DEFAULT_TYPE = "all"

SUFFIX_MAP = {
    "tpf": "_tp.fits",
    "lc": "_lc.fits",
    "hlsp": "_hlsp.fits",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 MAST 下载 TESS 数据（TPF / Light Curve），按目标名分类存放"
    )
    parser.add_argument(
        "--target",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="目标名称，支持多个（空格分隔）",
    )
    parser.add_argument(
        "--download-dir",
        default=DEFAULT_DOWNLOAD_DIR,
        help="下载根目录（每个目标会自动创建子文件夹）",
    )
    parser.add_argument(
        "--no-organize",
        action="store_true",
        help="禁用按 Sector 分子目录，该目标的所有文件直接放入目标文件夹",
    )
    parser.add_argument(
        "--type",
        choices=["tpf", "lc", "hlsp", "both", "all"],
        default=DEFAULT_TYPE,
        help="下载数据类型：tpf(Target Pixel File), lc(SPOC Light Curve), hlsp(High-Level Science Product), both(TPF+LC), all(全部)。默认: all",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="删除已下载的非目标 TIC ID 文件（清理锥形搜索混入的邻近星数据）",
    )
    args = parser.parse_args()

    targets = args.target if isinstance(args.target, list) else [args.target]
    base_dir = args.download_dir
    organize = not args.no_organize
    data_type = args.type
    clean = args.clean

    print(f"共 {len(targets)} 个目标待处理: {', '.join(targets)}")
    print(f"数据类型: {data_type}")

    grand_succeeded = 0
    grand_total = 0
    grand_failed = []
    grand_search_failed = []   # MAST 检索彻底失败的目标，单独记账（与 FITS 下载失败分开）

    for target_name in targets:
        total_succeeded = 0
        total_obs = 0
        all_failed = []

        # 0. 解析 TIC ID（返回 CSV 中匹配到的名字，用于目录命名）。
        matched_name = target_name
        ra_csv = None
        dec_csv = None
        tic_id = None

        # Step 1: 从交叉证认目录取坐标，再用坐标 cone search TIC 星表。
        try:
            csv_path = (
                PROJECT_ROOT
                / "Processed_Data"
                / "Catalogue"
                / "02.final_confirmed_stars_direct_2.csv"
            )
            df = pd.read_csv(csv_path)
            hostnames = df["hostname"].str.strip()
            match = df[hostnames == target_name.strip()]
            if len(match) == 0:
                match = df[
                    hostnames.str.contains(
                        re.escape(target_name.strip()), na=False
                    )
                ]
            if len(match) > 0:
                matched_name = match.iloc[0]["hostname"].strip()
                ra_csv = match.iloc[0]["ra"]
                dec_csv = match.iloc[0]["dec"]

                from astroquery.mast import Catalogs

                coord = SkyCoord(ra=ra_csv, dec=dec_csv, unit="deg")
                catalog = retry_with_exponential_backoff(
                    lambda: Catalogs.query_region(
                        coord, catalog="TIC", radius=0.0014
                    ),
                    "TIC 坐标查询",
                )
                if catalog and len(catalog) > 0:
                    row = catalog[0]
                    for key in ("ID", "tic", "TIC", "TIC_ID"):
                        val = row.get(key)
                        if val is not None:
                            tic_id = str(int(val))
                            break
        except Exception as error:
            print(
                f"  [WARN] 坐标→TIC 解析失败: "
                f"{type(error).__name__}: {error}"
            )

        # Step 2: 名字查询 fallback。
        if tic_id is None:
            try:
                from astroquery.mast import Catalogs

                catalog = retry_with_exponential_backoff(
                    lambda: Catalogs.query_object(
                        target_name, catalog="TIC", radius=0.0003
                    ),
                    "名字→TIC 查询",
                )
                if catalog and len(catalog) > 0:
                    row = catalog[0]
                    for key in ("ID", "tic", "TIC", "TIC_ID"):
                        val = row.get(key)
                        if val is not None:
                            tic_id = str(int(val))
                            break
            except Exception as error:
                print(
                    f"  [WARN] 名字→TIC 解析失败: "
                    f"{type(error).__name__}: {error}"
                )

        # Step 3: lightkurve 搜索 fallback。
        if tic_id is None:
            try:
                search_result = retry_with_exponential_backoff(
                    lambda: lk.search_targetpixelfile(
                        target_name, mission="TESS"
                    ),
                    "lightkurve 名称搜索",
                )
                if len(search_result) > 0:
                    match = re.search(
                        r"(\d{16})",
                        str(search_result.table["obs_id"][0]),
                    )
                    if match:
                        tic_id = str(int(match.group(1)))
            except Exception as error:
                print(
                    f"  [WARN] lightkurve 名称搜索失败: "
                    f"{type(error).__name__}: {error}"
                )

        # sanitize_name：保留原来的目录命名规则，不改变输出路径。
        safe_name = matched_name.replace(" ", "_").replace("/", "_").replace(
            "\\", "_"
        )
        target_dir = os.path.join(base_dir, safe_name)
        if tic_id:
            print(f"\n{'=' * 60}")
            if matched_name != target_name:
                print(
                    f"目标: {target_name}  ->  {matched_name}  ->  TIC {tic_id}"
                )
            else:
                print(f"目标: {target_name}  ->  TIC {tic_id}")
            search_target = f"TIC {tic_id}"
        else:
            print(f"目标: {target_name}  (无法解析 TIC ID)")
            search_target = target_name

        # 0b. 清理旧的错误文件。
        if clean and tic_id:
            target_tic = str(int(tic_id))
            removed = 0
            for file_path in sorted(
                glob.glob(os.path.join(target_dir, "**", "*.fits"), recursive=True)
            ):
                basename = os.path.basename(file_path)
                match = re.search(r"(\d{16})", basename)
                if match:
                    obs_tic = str(int(match.group(1)))
                    if obs_tic != target_tic:
                        os.remove(file_path)
                        print(
                            f"  已删除: {os.path.relpath(file_path, target_dir)}  "
                            f"[TIC {obs_tic}]"
                        )
                        removed += 1
            if removed > 0:
                print(f"  共清理 {removed} 个非目标文件")
            else:
                print("  无需清理")

        # 1. 检索。
        print(f"\n正在 MAST 中检索: {search_target} (类型: {data_type}) ...")
        try:
            search_results = {}
            if data_type in ("tpf", "both", "all"):
                search_results["tpf"] = retry_with_exponential_backoff(
                    lambda: lk.search_targetpixelfile(
                        search_target, mission="TESS"
                    ),
                    "TPF 检索",
                )
            if data_type in ("lc", "both", "all"):
                search_results["lc"] = retry_with_exponential_backoff(
                    lambda: lk.search_lightcurve(search_target, mission="TESS"),
                    "LC 检索",
                )
            if data_type in ("hlsp", "all"):
                # 部分 lightkurve 版本不接受 author=None；兼容两种接口。
                try:
                    search_results["hlsp"] = retry_with_exponential_backoff(
                        lambda: lk.search_lightcurve(
                            search_target, mission="TESS", author=None
                        ),
                        "HLSP 检索",
                    )
                except TypeError:
                    search_results["hlsp"] = retry_with_exponential_backoff(
                        lambda: lk.search_lightcurve(
                            search_target, mission="TESS"
                        ),
                        "HLSP 检索",
                    )
        except Exception as error:
            err = f"{type(error).__name__}: {error}"
            print(
                f"  [ERROR] {search_target} 检索失败（已重试 3 次）: {err}"
            )
            grand_search_failed.append((search_target, err))
            continue

        total_raw = sum(len(result) for result in search_results.values())
        for dtype, result in search_results.items():
            print(f"  {dtype.upper()}: 找到 {len(result)} 组")
        print(f"  总计: {total_raw} 组")

        # 1b. 过滤邻近星。
        if tic_id:
            target_tic = str(int(tic_id))
            filtered_results = {}
            excluded = {}
            for dtype, result in search_results.items():
                keep = []
                drop = []
                for row in result:
                    obs_id = str(row.table["obs_id"][0])
                    match = re.search(r"(\d{16})", obs_id)
                    if match:
                        obs_tic = str(int(match.group(1)))
                    else:
                        match_fallback = re.search(
                            r"tic(\d+)", obs_id, re.IGNORECASE
                        )
                        obs_tic = (
                            str(int(match_fallback.group(1)))
                            if match_fallback
                            else None
                        )
                    if obs_tic == target_tic:
                        keep.append(True)
                        drop.append(False)
                    else:
                        keep.append(False)
                        drop.append(True)
                keep_mask = np.array(keep, dtype=bool)
                drop_mask = np.array(drop, dtype=bool)
                filtered_results[dtype] = result[keep_mask]
                excluded[dtype] = result[drop_mask]
            search_results = filtered_results

            n_excluded = sum(len(result) for result in excluded.values())
            if n_excluded > 0:
                print(
                    f"  已过滤 {n_excluded} 组非目标数据（目标 TIC {tic_id}）:"
                )
                for dtype, excluded_result in excluded.items():
                    for row in excluded_result:
                        obs_id = row.table["obs_id"][0]
                        match = re.search(r"(\d{16})", str(obs_id))
                        other_tic = str(int(match.group(1))) if match else "?"
                        print(
                            f"    - {dtype.upper()}: TIC {other_tic}  "
                            f"[{obs_id[:60]}]"
                        )
            else:
                print(f"  目标 TIC {tic_id}，所有数据均匹配")
        else:
            print("  无法解析 TIC ID，跳过邻近星过滤")

        total_found = sum(len(result) for result in search_results.values())
        if total_found == 0:
            print("未找到任何数据，跳过。")
            continue

        # 2. 扫描本地，按原后缀分类已下载 obs_id。
        if not os.path.isdir(target_dir):
            existing_ids = {suffix: set() for suffix in SUFFIX_MAP.values()}
            n_files = 0
        else:
            existing_files = glob.glob(
                os.path.join(target_dir, "**", "*.fits"), recursive=True
            )
            existing_ids = {suffix: set() for suffix in SUFFIX_MAP.values()}
            for file_path in existing_files:
                basename = os.path.basename(file_path)
                matched_suffix = False
                for suffix in SUFFIX_MAP.values():
                    if basename.endswith(suffix):
                        existing_ids[suffix].add(basename.replace(suffix, ""))
                        matched_suffix = True
                        break
                if not matched_suffix:
                    existing_ids["_tp.fits"].add(basename.replace(".fits", ""))
            n_files = len(existing_files)
        if n_files > 0:
            safe_target_name = target_name.replace(" ", "_").replace(
                "/", "_"
            ).replace("\\", "_")
            print(
                f"本地 [{safe_target_name}] 目录已有 {n_files} 个 FITS 文件。"
            )

        # 3. 找出缺失数据。
        to_download = []
        seen = set()
        for dtype, result in search_results.items():
            suffix = SUFFIX_MAP[dtype]
            for row in result:
                obs_id = row.table["obs_id"][0]
                if obs_id in seen:
                    continue
                seen.add(obs_id)
                if obs_id not in existing_ids[suffix]:
                    to_download.append((row, dtype))

        if not to_download:
            print(f"[{target_name}] 所有数据均已下载。")
            continue

        # 3b. 只下载最近的 20 组（按观测时间降序，低于 20 组则全下载）。
        download_with_times = []
        for item in to_download:
            obs_id = item[0].table["obs_id"][0]
            match = re.search(r"tess(\d{12})", obs_id)
            if match:
                obs_time = int(match.group(1))
            else:
                match = re.search(r"s(\d{4})", obs_id)
                obs_time = int(match.group(1)) * 1000000 if match else 0
            download_with_times.append((obs_time, item))
        download_with_times.sort(key=lambda item: item[0], reverse=True)
        to_download = [item for _, item in download_with_times[:20]]
        print(f"[{target_name}] 下载最新 {len(to_download)} 组...")

        # 4. 下载。
        total_obs = len(to_download)
        for row, dtype in tqdm(
            to_download, desc=f"下载 {target_name}", unit="组"
        ):
            obs_id = row.table["obs_id"][0]
            if organize:
                sector_parts = obs_id.split("-")
                try:
                    sector = int(sector_parts[1].lstrip("s"))
                except (IndexError, ValueError):
                    sector = 0
                dl_dir = os.path.join(target_dir, f"Sector_{sector:02d}")
            else:
                dl_dir = target_dir
            os.makedirs(dl_dir, exist_ok=True)

            # 内联 download_one；ThreadPoolExecutor 保持原有第三方回调。
            success = False
            obs_id_out = obs_id
            err = None
            for attempt in range(1, 3 + 1):
                before = list_fits_files(dl_dir)
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            row.download, download_dir=dl_dir
                        )
                        future.result(timeout=300)
                    success = True
                    break
                except FutureTimeout:
                    after = list_fits_files(dl_dir)
                    if after - before:
                        success = True
                        break
                    print(
                        f"      {obs_id} 下载超时 (300s)，尝试 "
                        f"{attempt}/3"
                    )
                    if attempt < 3:
                        time.sleep(2 ** attempt)
                    else:
                        err = "超时 (300s)"
                except Exception as error:
                    after = list_fits_files(dl_dir)
                    if after - before:
                        success = True
                        break
                    if attempt < 3:
                        time.sleep(2 ** attempt)
                    else:
                        err = str(error)
            if not success:
                err = err or "unknown"
                all_failed.append((obs_id_out, err))
                tqdm.write(f"  失败 {obs_id_out} [{dtype}]: {err}")
            else:
                total_succeeded += 1

        grand_succeeded += total_succeeded
        grand_total += total_obs
        grand_failed.extend(all_failed)

    # 汇总
    print(f"\n{'='*60}")
    print(f"全部下载完毕: 成功 {grand_succeeded}/{grand_total}")
    if grand_search_failed:
        print("\n以下目标 MAST 检索失败（已重试 3 次，可重新运行脚本重试）:")
        for label, err in grand_search_failed:
            print(f"  - {label}: {err}")
    if grand_failed:
        print("以下数据下载失败（可重新运行脚本重试）:")
        for obs_id, err in grand_failed:
            print(f"  - {obs_id}: {err}")


if __name__ == "__main__":
    main()
