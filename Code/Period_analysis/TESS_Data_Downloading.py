import os
import re
import glob
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import lightkurve as lk
from tqdm.auto import tqdm

# ========================
# 配置：修改这里的默认值
# ========================
DEFAULT_TARGETS = ["2MASS J01033563-5515561 A"]
# DEFAULT_DOWNLOAD_DIR = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/TESS_Data"
DEFAULT_DOWNLOAD_DIR = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Data/TESS_Data"
DEFAULT_TYPE = "all"

SUFFIX_MAP = {
    "tpf": "_tp.fits",
    "lc": "_lc.fits",
    "hlsp": "_hlsp.fits",
}


def project_path(relative_path):
    current = os.path.abspath(os.path.dirname(__file__))
    while not (os.path.isdir(os.path.join(current, 'Code')) and
               os.path.isdir(os.path.join(current, 'Processed_Data'))):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.join(current, relative_path)


def sanitize_name(name):
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def scan_local_files(folder):
    """扫描目录，返回已下载数据的 obs_id 集合（按后缀分类）。"""
    if not os.path.isdir(folder):
        return {suffix: set() for suffix in SUFFIX_MAP.values()}, 0

    pattern = os.path.join(folder, "**", "*.fits")
    existing_files = glob.glob(pattern, recursive=True)

    existing_ids = {suffix: set() for suffix in SUFFIX_MAP.values()}
    for f in existing_files:
        basename = os.path.basename(f)
        matched = False
        for suffix in SUFFIX_MAP.values():
            if basename.endswith(suffix):
                clean = basename.replace(suffix, "")
                existing_ids[suffix].add(clean)
                matched = True
                break
        if not matched:
            clean = basename.replace(".fits", "")
            existing_ids["_tp.fits"].add(clean)

    return existing_ids, len(existing_files)


def parse_sector(obs_id):
    try:
        sector_str = obs_id.split("-")[1]
        return int(sector_str.lstrip("s"))
    except (IndexError, ValueError):
        return 0


def parse_obs_time(obs_id):
    """从 TESS obs_id 提取观测日期，返回可排序的整数（YYYYDDDHHMMSS）。
    格式: tessYYYYDDDHHMMSS-...  例如 tess2018234235059-s0002"""
    m = re.search(r"tess(\d{12})", obs_id)
    if m:
        return int(m.group(1))
    m = re.search(r"s(\d{4})", obs_id)  # fallback: sector number
    if m:
        return int(m.group(1)) * 1000000
    return 0


def resolve_tic_id(target_name):
    """用 CSV 目录中的坐标查 TIC ID，失败则走名字搜索 fallback。
    返回 (tic_id, matched_name, ra, dec)"""

    matched_name = target_name
    ra_csv = None
    dec_csv = None

    # Step 1: 从交叉证认目录取坐标 → 用坐标 cone search TIC 星表
    try:
        csv_path = project_path('Processed_Data/Catalogue/02.final_confirmed_stars_direct_2.csv')
        df = pd.read_csv(csv_path)
        hostnames = df['hostname'].str.strip()

        # 精确匹配
        match = df[hostnames == target_name.strip()]
        # 模糊匹配：hostname 包含 target_name
        if len(match) == 0:
            match = df[hostnames.str.contains(re.escape(target_name.strip()), na=False)]
        if len(match) > 0:
            matched_name = match.iloc[0]['hostname'].strip()
            ra_csv = match.iloc[0]['ra']
            dec_csv = match.iloc[0]['dec']

            from astroquery.mast import Catalogs
            coord = SkyCoord(ra=ra_csv, dec=dec_csv, unit="deg")
            catalog = Catalogs.query_region(coord, catalog="TIC", radius=0.0014)
            if catalog and len(catalog) > 0:
                row = catalog[0]
                for key in ("ID", "tic", "TIC", "TIC_ID"):
                    val = row.get(key)
                    if val is not None:
                        return str(int(val)), matched_name, ra_csv, dec_csv
    except Exception:
        pass

    # Step 2: 名字查询 fallback
    try:
        from astroquery.mast import Catalogs
        catalog = Catalogs.query_object(target_name, catalog="TIC", radius=0.0003)
        if catalog and len(catalog) > 0:
            row = catalog[0]
            for key in ("ID", "tic", "TIC", "TIC_ID"):
                val = row.get(key)
                if val is not None:
                    return str(int(val)), matched_name, ra_csv, dec_csv
    except Exception:
        pass

    # Step 3: lightkurve 搜索 fallback
    try:
        sr = lk.search_targetpixelfile(target_name, mission="TESS")
        if len(sr) > 0:
            m = re.search(r"(\d{16})", str(sr.table["obs_id"][0]))
            if m:
                return str(int(m.group(1))), matched_name, ra_csv, dec_csv
    except Exception:
        pass

    return None, matched_name, ra_csv, dec_csv


def filter_search_results(results, tic_id_str):
    """从 MAST 搜索结果中过滤掉非目标 TIC ID 的数据"""
    if tic_id_str is None:
        return results, {}
    target_tic = str(int(tic_id_str))        # "206502540"

    filtered = {}
    excluded = {}
    for dtype, result in results.items():
        keep = []
        drop = []
        for row in result:
            obs_id = str(row.table["obs_id"][0])
            m = re.search(r"(\d{16})", obs_id)
            if m:
                obs_tic = str(int(m.group(1)))
            else:
                m2 = re.search(r"tic(\d+)", obs_id, re.IGNORECASE)
                obs_tic = str(int(m2.group(1))) if m2 else None
            if obs_tic == target_tic:
                keep.append(True)
                drop.append(False)
            else:
                keep.append(False)
                drop.append(True)
        keep_mask = np.array(keep, dtype=bool)
        drop_mask = np.array(drop, dtype=bool)
        filtered[dtype] = result[keep_mask]
        excluded[dtype] = result[drop_mask]
    return filtered, excluded


def clean_existing_files(target_dir, tic_id_str):
    """删除目标目录中不属于该 TIC ID 的已下载文件"""
    if tic_id_str is None:
        return 0
    target_tic = str(int(tic_id_str))
    pattern = os.path.join(target_dir, "**", "*.fits")
    removed = 0
    for f in sorted(glob.glob(pattern, recursive=True)):
        basename = os.path.basename(f)
        m = re.search(r"(\d{16})", basename)
        if m:
            obs_tic = str(int(m.group(1)))
            if obs_tic != target_tic:
                os.remove(f)
                print(f"  已删除: {os.path.relpath(f, target_dir)}  [TIC {obs_tic}]")
                removed += 1
    return removed


def _list_fits_files(directory):
    """递归列出目录下所有 .fits 文件"""
    if not os.path.isdir(directory):
        return set()
    return set(glob.glob(os.path.join(directory, "**", "*.fits"), recursive=True))


def download_one(row, download_dir, max_retries=3, timeout=300):
    obs_id = row.table["obs_id"][0]

    for attempt in range(1, max_retries + 1):
        before = _list_fits_files(download_dir)
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(row.download, download_dir=download_dir)
                future.result(timeout=timeout)
            return True, obs_id, None
        except FutureTimeout:
            after = _list_fits_files(download_dir)
            new_files = after - before
            if new_files:
                return True, obs_id, None
            print(f"      {obs_id} 下载超时 ({timeout}s)，尝试 {attempt}/{max_retries}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return False, obs_id, f"超时 ({timeout}s)"
        except Exception as e:
            after = _list_fits_files(download_dir)
            new_files = after - before
            if new_files:
                return True, obs_id, None
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return False, obs_id, str(e)

    return False, obs_id, "unknown"


def search_data(target, data_type):
    """根据类型检索 MAST：tpf / lc / hlsp / both / all。
    target 可以是名字字符串或 SkyCoord 对象。"""
    results = {}
    if data_type in ("tpf", "both", "all"):
        results["tpf"] = lk.search_targetpixelfile(target, mission="TESS")
    if data_type in ("lc", "both", "all"):
        results["lc"] = lk.search_lightcurve(target, mission="TESS")
    if data_type in ("hlsp", "all"):
        try:
            results["hlsp"] = lk.search_lightcurve(target, mission="TESS", author=None)
        except TypeError:
            results["hlsp"] = lk.search_lightcurve(target, mission="TESS")
    return results


def process_target(target_name, base_dir, organize_by_sector, data_type, clean=False):
    total_succeeded = 0
    total_obs = 0
    all_failed = []

    # 0. 解析 TIC ID（返回 CSV 中匹配到的名字，用于目录命名）
    tic_id, matched_name, _, _ = resolve_tic_id(target_name)
    target_dir = os.path.join(base_dir, sanitize_name(matched_name))
    if tic_id:
        print(f"\n{'='*60}")
        if matched_name != target_name:
            print(f"目标: {target_name}  ->  {matched_name}  ->  TIC {tic_id}")
        else:
            print(f"目标: {target_name}  ->  TIC {tic_id}")
        search_target = f"TIC {tic_id}"
    else:
        print(f"目标: {target_name}  (无法解析 TIC ID)")
        search_target = target_name

    # 0b. 清理旧的错误文件
    if clean and tic_id:
        n = clean_existing_files(target_dir, tic_id)
        if n > 0:
            print(f"  共清理 {n} 个非目标文件")
        else:
            print(f"  无需清理")

    # 1. 检索
    print(f"\n正在 MAST 中检索: {search_target} (类型: {data_type}) ...")
    search_results = search_data(search_target, data_type)

    total_raw = sum(len(r) for r in search_results.values())
    for dtype, result in search_results.items():
        print(f"  {dtype.upper()}: 找到 {len(result)} 组")
    print(f"  总计: {total_raw} 组")

    # 1b. 过滤邻近星
    if tic_id:
        search_results, excluded = filter_search_results(search_results, tic_id)
        n_excluded = sum(len(r) for r in excluded.values())
        if n_excluded > 0:
            print(f"  已过滤 {n_excluded} 组非目标数据（目标 TIC {tic_id}）:")
            for dtype, ex in excluded.items():
                for row in ex:
                    obs_id = row.table["obs_id"][0]
                    m = re.search(r"(\d{16})", str(obs_id))
                    other_tic = str(int(m.group(1))) if m else "?"
                    print(f"    - {dtype.upper()}: TIC {other_tic}  [{obs_id[:60]}]")
        else:
            print(f"  目标 TIC {tic_id}，所有数据均匹配")
    else:
        print("  无法解析 TIC ID，跳过邻近星过滤")

    total_found = sum(len(r) for r in search_results.values())

    if total_found == 0:
        print("未找到任何数据，跳过。")
        return 0, 0, []

    # 2. 扫描本地
    existing_ids, n_files = scan_local_files(target_dir)
    if n_files > 0:
        print(f"本地 [{sanitize_name(target_name)}] 目录已有 {n_files} 个 FITS 文件。")

    # 3. 找出缺失数据
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
        return 0, 0, []

    # 3b. 只下载最近的 20 组（按 Sector 降序，低于 20 组则全下载）
    MAX_DOWNLOAD = 20
    to_download.sort(key=lambda x: parse_obs_time(x[0].table["obs_id"][0]), reverse=True)
    to_download = to_download[:MAX_DOWNLOAD]
    print(f"[{target_name}] 下载最新 {len(to_download)} 组...")

    # 4. 下载
    total_obs = len(to_download)

    for row, dtype in tqdm(to_download, desc=f"下载 {target_name}", unit="组"):
        obs_id = row.table["obs_id"][0]

        if organize_by_sector:
            sector = parse_sector(obs_id)
            dl_dir = os.path.join(target_dir, f"Sector_{sector:02d}")
        else:
            dl_dir = target_dir

        os.makedirs(dl_dir, exist_ok=True)

        success, obs_id_out, err = download_one(row, download_dir=dl_dir)
        if success:
            total_succeeded += 1
        else:
            all_failed.append((obs_id_out, err))
            tqdm.write(f"  失败 {obs_id_out} [{dtype}]: {err}")

    return total_succeeded, total_obs, all_failed


def main():
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

    for target_name in targets:
        s, t, f = process_target(target_name, base_dir, organize, data_type, clean)
        grand_succeeded += s
        grand_total += t
        grand_failed.extend(f)

    # 汇总
    print(f"\n{'='*60}")
    print(f"全部下载完毕: 成功 {grand_succeeded}/{grand_total}")
    if grand_failed:
        print("以下数据下载失败（可重新运行脚本重试）:")
        for obs_id, err in grand_failed:
            print(f"  - {obs_id}: {err}")


if __name__ == "__main__":
    main()