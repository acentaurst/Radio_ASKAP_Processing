"""Measure ASKAP Stokes-I/V periods and produce epoch-aware Lomb--Scargle folds."""

import os
import sys
import glob
import re
import multiprocessing as mp
import warnings
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# 强制在无图形界面的服务器环境下运行
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import astropy.units as u
import h5py
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.timeseries import LombScargle
from astropy.time import Time
from astropy.utils import iers
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from science_utils import (  # noqa: E402
    add_panel_label,
    as_text,
    phase_bin_for_display,
    rebin,
    rebin2d,
    slice_open_end,
)

ASKAP_LOCATION = EarthLocation.from_geodetic(
    lon=116.631425 * u.deg,
    lat=-26.697000 * u.deg,
    height=361.0 * u.m,
)
iers.conf.auto_download = False
iers.conf.auto_max_age = None


# 自适应路径定位（优先从脚本位置向上找项目根，PyCharm中则用容器挂载目录）
_PROJECT_MOUNT = "/home/dev/projects/ASKAP_Stellar_with_Exoplanet"


# ==========================================
# 1. 全局核心配置
# ==========================================
TESS_PERIOD = 0.1664         # TESS 测定的恒星光学周期 (天)
TESS_PERIOD_ERR = 0.0008     # TESS 周期不确定度 (天)
TESS_PERIOD_SIGMA = 3        # LS 图上 TESS 周期 band 的半宽 (σ)

# 数据目录：包含待处理 .ds 文件的目录
DS_FILES_DIR = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/Ds/2MASS_J01033563-5515561_A/Flare"

# 待处理的 SBID 列表（空列表 = 目录下全部 SBID）
TARGET_SBIDS = ["59565"]
# "59565","66827", "68040"
# 处理模式开关：
#   True  = 将 TARGET_SBIDS 中多个 SBID 拼接合并，统一测量周期（激活窗口函数）
#   False = 批量逐个处理每个 SBID，各自单独测量周期
COMBINE_SBIDS = True

OUTPUT_BASE = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/Radio_LS_PhaseFolding"

# Lomb-Scargle 周期搜索范围 (天)
PERIOD_MIN = 0.01
PERIOD_MAX = 50.0

# FAP 总开关：True 计算并显示全部 FAP（analytic 阈值线、数值、bootstrap）；
# False 时 analytic 阈值线、数值、bootstrap 全部不计算不显示。
ENABLE_FAP = False

FAP_PERCENT = 0.01   # LS 图上 FAP 阈值线对应的虚警概率（0.01 = 1%）
BOOTSTRAP_ITERS = 10000   # Bootstrap FAP 迭代次数（显著性检验）
# Block bootstrap 块长 (分钟)。保持时间网格不变，把 flux 切成连续块随机重排，
# 保留块内 flare/相关结构、破坏跨块连续性；None 则退化为单点 permutation。
BOOTSTRAP_BLOCK_MIN = 10
# Bootstrap 性能控制：
#   BOOTSTRAP_MAX_NFREQ : 参与 bootstrap 的频率网格点数上限（长基线时 autopower
#                         网格可达几十万点，降采样到该上限可大幅提速，峰值比较在
#                         同一降采样网格上进行，保持自洽）
#   BOOTSTRAP_NPROC     : 并行核数（None = 自动用满 CPU）
BOOTSTRAP_MAX_NFREQ = 10000
BOOTSTRAP_NPROC = None
BOOTSTRAP_RANDOM_SEED = 42

# 相位折叠模式：
#   1 = 先折叠至 [0, 1)，再复制至 [1, 2)（规范的单周期折叠展示）；
#   2 = 原始相位直接 mod 2，显示相邻两个实际周期的差异（不复制数据）。
# 两种模式只影响相位图，不改变 LS、FAP、LOEO 或独立样本数。
PHASE_FOLD_CYCLES = 2
# 每个相位图面板的总分箱数。mod 1 的复制展示会自动均分给左右两半；
# mod 2 则将此数用于两个实际周期的完整 [0, 2) 范围。
PHASE_NBINS = 60
# 科研图输出分辨率；保持所有主结果图的像素质量一致。
FIGURE_DPI = 300


def _bootstrap_chunk(args):
    """多进程 worker：对一段 seeds 逐次打乱 flux 并统计 LS 峰值 >= 观测峰值的次数。"""
    t, flux, frequency, block_size, epoch_bounds, seeds, power_obs = args
    rng = np.random.default_rng(seeds[0])
    count = 0
    for _ in range(len(seeds)):
        if block_size <= 1:
            shuffled = flux[rng.permutation(len(flux))]
        else:
            shuffled_chunks = []
            for start, end in epoch_bounds:
                epoch_flux = flux[start:end]
                chunks = [epoch_flux[i:i + block_size]
                          for i in range(0, len(epoch_flux), block_size)]
                rng.shuffle(chunks)
                shuffled_chunks.append(np.concatenate(chunks))
            shuffled = np.concatenate(shuffled_chunks)
        if np.max(LombScargle(t, shuffled).power(frequency)) >= power_obs:
            count += 1
    return count


# ==========================================
# 3--4. 主流程：数据加载、周期分析与绘图
# ==========================================
def main() -> None:
    """入口：逐项执行数据加载、周期分析和绘图，保持合并/逐 SBID 两种模式。"""
    if COMBINE_SBIDS:
        processing_targets = [
            (TARGET_SBIDS or None,
             "目录下全部 SBID" if not TARGET_SBIDS else f"{TARGET_SBIDS} 合并")
        ]
    else:
        processing_targets = [([sbid], f"SB{sbid}") for sbid in TARGET_SBIDS]

    for sbid_list, target_label in processing_targets:
        if COMBINE_SBIDS:
            print(f"\n[INFO] === 合并模式：{target_label}，统一测量周期 ===")
        else:
            print(f"\n[INFO] === 逐个模式：处理 {target_label} ===")

        # ==========================================
        # 3. 数据加载、过滤与拼接
        # ==========================================
        files = sorted(glob.glob(os.path.join(DS_FILES_DIR, "*.ds")))

        if not files:
            print("[ERROR] 未找到匹配的 .ds 文件")
            if COMBINE_SBIDS:
                sys.exit(1)
            print(f"  [WARN] {target_label} 无可用数据，跳过")
            continue

        # 从第一个文件名提取 source_name
        basename0 = os.path.basename(files[0])
        source_match = re.search(r'(.+?)_SB\d+', basename0)
        SOURCE_NAME = source_match.group(1) if source_match else os.path.splitext(basename0)[0]
        MASTER_OUTPUT_DIR = os.path.join(OUTPUT_BASE, SOURCE_NAME)
        os.makedirs(MASTER_OUTPUT_DIR, exist_ok=True)

        print(f"[INFO] 源: {SOURCE_NAME} | 候选文件: {len(files)} 个")

        # 读取 catalogue CSV 以获取各 SBID 的绝对起始时间 (MJD)
        csv_path = PROJECT_ROOT / "Processed_Data" / "Catalogue" / "01.askap_catalogue.csv"
        sbid_mjd_map = {}
        if os.path.exists(csv_path):
            try:
                df_cat = pd.read_csv(csv_path)
                obs_col, mjd_col = 'obs_id', 't_min'
                if obs_col in df_cat.columns and mjd_col in df_cat.columns:
                    for _, row in df_cat.iterrows():
                        match = re.search(r'(\d+)', str(row[obs_col]))
                        if match:
                            sbid_mjd_map[f"SB{match.group(1)}"] = float(row[mjd_col])
            except Exception as e:
                print(f"[WARN] CSV 目录文件读取失败: {e}")

        all_mjd, all_flux_i, all_flux_v, all_sbid_id = [], [], [], []
        sbid_map, beam_map = {}, {}
        useful_count = 0

        for file in files:
            basename = os.path.basename(file)

            # 提取并校验 SBID
            sbid_num_match = re.search(r'SB(\d+)', basename, re.IGNORECASE)
            if not sbid_num_match: continue
            sbid_num = sbid_num_match.group(1)
            sbid = f"SB{sbid_num}"

            # 按 TARGET_SBIDS 过滤（空列表 = 全部）
            if sbid_list and sbid_num not in sbid_list:
                continue

            # 提取 beam 编号用于文件命名
            beam_match = re.search(r'beam(\d+)', basename, re.IGNORECASE)
            beam_num = beam_match.group(1) if beam_match else "X"

            try:
                # 读取时进行日心校正，使跨历元周期分析使用统一的 TDB 时间轴。
                # 该脚本只在这里读取一次 `.ds`；按低复用函数规则直接展开原
                # DynamicSpectrum 的 HDF5、校准扫描拼接、重采样和 Stokes 逻辑。
                tunit = u.hour
                tavg = favg = 1
                required = {"flux", "time", "frequency", "uvdist"}
                with h5py.File(file, "r") as data_file:
                    missing = required.difference(data_file.keys())
                    if missing:
                        raise ValueError(f"Missing .ds datasets: {sorted(missing)}")
                    header = dict(data_file.attrs)
                    uvdist = data_file["uvdist"][:]
                    time = np.asarray(data_file["time"][:], dtype=float)
                    freq = np.asarray(data_file["frequency"][:], dtype=float) / 1e6
                    flux = data_file["flux"][:] * 1e3
                if flux.ndim != 4 or flux.shape[-1] != 4:
                    raise ValueError(
                        "Expected flux shape (baseline, time, frequency, 4), "
                        f"got {flux.shape}"
                    )
                if len(time) != flux.shape[1] or len(freq) != flux.shape[2]:
                    raise ValueError(".ds time/frequency axes do not match flux shape")
                if len(uvdist) != flux.shape[0]:
                    raise ValueError(".ds uvdist axis does not match flux shape")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    flux = np.nanmean(flux, axis=0)
                xx, xy, yx, yy = (flux[:, :, index] for index in range(4))
                time_scale_factor = tunit.to(u.s)
                time /= time_scale_factor
                corr_dumptime = 10.1 / time_scale_factor
                if freq[0] > freq[-1]:
                    xx = np.flip(xx, axis=1)
                    xy = np.flip(xy, axis=1)
                    yx = np.flip(yx, axis=1)
                    yy = np.flip(yy, axis=1)
                    freq = np.flip(freq)

                # 原调用固定 trim=True、无频率/时间裁剪，因此这里保留同一
                # “有效频率范围 + 全时间范围”的边界计算。
                full = np.nansum(xx + xy + yx + yy, axis=0)
                full[full == 0.0 + 0.0j] = np.nan
                all_pols = np.isfinite(full)
                if not np.any(all_pols):
                    raise ValueError("No finite channels remain after trimming")
                min_channel = int(np.argmax(all_pols))
                max_channel = 0 if all_pols[-1] else int(-np.argmax(all_pols[::-1]) + 1)
                min_time = max_time = 0
                converted_time = Time((time * tunit).to(u.day), format="mjd", scale="utc")
                telescope = as_text(header.get("telescope", ""))
                if telescope != "ASKAP":
                    raise ValueError(f"No embedded observatory location for {telescope!r}")
                phasecentre = as_text(header.get("phasecentre", ""))
                try:
                    ra, dec = phasecentre.split()
                except ValueError as error:
                    raise ValueError(
                        f"Invalid phasecentre attribute: {phasecentre!r}"
                    ) from error
                target_coord = SkyCoord(
                    ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame="icrs"
                )
                at_site = Time(
                    converted_time, format="mjd", scale="utc", location=ASKAP_LOCATION
                )
                converted_time = at_site.tdb + at_site.light_travel_time(target_coord)
                start_time = getattr(converted_time[0], converted_time[0].scale)
                header["time_start"] = start_time.iso
                header["time_scale"] = converted_time[0].scale
                time = (converted_time - converted_time[0]).value * u.day.to(tunit)
                xx = slice_open_end(xx, min_time, max_time, min_channel, max_channel)
                xy = slice_open_end(xy, min_time, max_time, min_channel, max_channel)
                yx = slice_open_end(yx, min_time, max_time, min_channel, max_channel)
                yy = slice_open_end(yy, min_time, max_time, min_channel, max_channel)
                freq = slice_open_end(freq, min_channel, max_channel)
                time = slice_open_end(time, min_time, max_time)
                header.update({"integrations": len(time), "channels": len(freq)})

                # calscans=True：在扫描间隔中插入 NaN 时间步，保持旧有
                # 动态谱到周期分析的时间网格语义。
                if len(time) >= 2:
                    deltas = np.zeros(len(time))
                    deltas[1:] = np.diff(time)
                    scan_starts = np.where(np.abs(deltas) > corr_dumptime)[0]
                    scan_ends = scan_starts - 1
                    scan_starts = np.insert(scan_starts, 0, 0)
                    scan_ends = np.append(scan_ends, len(time) - 1)
                    time_end_break = time[scan_starts[1:]]
                    time_start_break = time[scan_ends[:-1]]
                    cadence = time[1] - time[0]
                    break_cycles = np.append((time_end_break - time_start_break), 0) / cadence
                    chunks_by_pol = [[], [], [], []]
                    time_chunks = []
                    for start, end, cycles in zip(scan_starts, scan_ends, break_cycles):
                        pol_chunks = [array[start:end + 1, :] for array in (xx, xy, yx, yy)]
                        time_chunk = time[start:end + 1]
                        if cycles > 0:
                            number_of_steps = int(round(cycles) - 1)
                            nan_chunk = np.full(
                                (number_of_steps, header["channels"]),
                                np.nan + np.nan * 1j,
                            )
                            pol_chunks = [
                                np.ma.vstack((chunk, nan_chunk)) for chunk in pol_chunks
                            ]
                            break_start = time[end] + cadence
                            break_times = break_start + np.arange(number_of_steps) * cadence
                            time_chunk = np.append(time_chunk, break_times)
                        for chunk_list, chunk in zip(chunks_by_pol, pol_chunks):
                            chunk_list.append(chunk)
                        time_chunks.append(time_chunk)
                    time = np.concatenate(time_chunks)
                    xx, xy, yx, yy = (
                        np.ma.vstack(chunks) for chunks in chunks_by_pol
                    )

                number_of_times, number_of_channels = xx.shape
                time_bins = number_of_times // tavg
                frequency_bins = number_of_channels // favg
                if time_bins < 1 or frequency_bins < 1:
                    raise ValueError("Averaging factor is larger than an available axis")
                xx = rebin2d(xx, (time_bins, frequency_bins))
                xy = rebin2d(xy, (time_bins, frequency_bins))
                yx = rebin2d(yx, (time_bins, frequency_bins))
                yy = rebin2d(yy, (time_bins, frequency_bins))
                time = rebin(number_of_times, time_bins, axis=0) @ time
                freq = freq @ rebin(number_of_channels, frequency_bins, axis=1)
                if len(time) < 2 or len(freq) < 2:
                    raise ValueError("Dynamic spectrum needs at least two time and frequency bins")
                header.update({
                    "time_resolution": f"{((time[1] - time[0]) * tunit).to(u.s):.3f}",
                    "freq_resolution": f"{((freq[1] - freq[0]) * u.MHz).to(u.MHz):.2f}",
                })

                feed_type = as_text(header.get("feeds", ""))
                if feed_type == "linear":
                    intensity = (xx + yy) / 2
                    q_stokes = (xx - yy) / 2
                    u_stokes = (xy + yx) / 2
                    v_stokes = 1j * (yx - xy) / 2
                elif feed_type == "circular":
                    intensity = (xx + yy) / 2
                    q_stokes = (xy + yx) / 2
                    u_stokes = 1j * (xy - yx) / 2
                    v_stokes = (xx - yy) / 2
                else:
                    raise ValueError(
                        f"Feed type {feed_type!r} is not 'linear' or 'circular'"
                    )
                data = {
                    "XX": xx,
                    "XY": xy,
                    "YX": yx,
                    "YY": yy,
                    "I": intensity,
                    "Q": q_stokes,
                    "U": u_stokes,
                    "V": v_stokes,
                    "L": q_stokes.real + 1j * u_stokes.real,
                }
                ds = SimpleNamespace(time=time, tunit=tunit, header=header,
                                     data=data, freq=freq)
                relative_time = getattr(ds, "time", None)
                if relative_time is None: continue
                relative_time = np.asarray(relative_time, dtype=float)

                # ds.time 的单位由 ds.tunit 决定，先显式换算成小时再做时长判断
                try:
                    t_hours = (relative_time * ds.tunit).to_value("hour")
                except Exception:
                    # 兼容少数旧 `.ds` 文件将 tunit 保存为普通字符串或数值、且 time 本身即小时的情况
                    t_hours = relative_time

                print(f"    ds.time range: [{t_hours[0]:.2f}, {t_hours[-1]:.2f}] h")

                duration_hours = np.ptp(t_hours)
                # 过滤掉观测时长过短的数据块
                if duration_hours < 2.0: continue

                if sbid not in sbid_mjd_map:
                    print(f"  [WARN] {sbid} 未在 CSV 目录中登记，跳过 MJD 校验；"
                          f"采样时间仍以 header time_start + ds.time 为准")

                if sbid not in sbid_map:
                    sbid_map[sbid] = len(sbid_map)
                    beam_map[sbid] = beam_num

                # 将 `.ds` 的相对时间和文件头起始时刻组合为 BJD_TDB。
                # 不能直接把 catalogue 的 UTC t_min 当作采样时间，否则会引入
                # 时间尺度/起始定义不一致，尤其会影响跨历元相位。
                relative_time = np.asarray(relative_time, dtype=float)
                time_scale = str(ds.header.get("time_scale", "utc")).lower()
                t_start = Time(ds.header["time_start"], scale=time_scale)
                try:
                    relative_days = (relative_time * ds.tunit).to_value("day")
                except AttributeError:
                    # 兼容旧 `.ds` 文件将 tunit 保存为普通字符串或数值的情况。
                    relative_days = relative_time / 24.0
                t_days = t_start.tdb.jd + relative_days

                stokes_i_data, stokes_v_data = ds.data.get("I"), ds.data.get("V")
                if stokes_i_data is None or stokes_v_data is None: continue

                # 沿频率轴求平均，获取宽频光变曲线
                t_len = len(t_hours)
                freq_axis = 1 if stokes_i_data.shape[0] == t_len else 0
                flux_i = np.nanmean(stokes_i_data, axis=freq_axis)
                flux_v = np.nanmean(stokes_v_data, axis=freq_axis)

                flux_i = np.real(flux_i)
                flux_v = np.real(flux_v)

                # 每 epoch 去基线偏移（保留 flare 结构，消除 calibration 尺度差异）
                flux_i = flux_i - np.nanmedian(flux_i)
                flux_v = flux_v - np.nanmedian(flux_v)

                # 确认数据形状、长度与有效值后，才计入 useful_count 并加入列表
                if not (len(flux_i) == len(flux_v) == t_len and t_len > 0):
                    print(f"  [WARN] 跳过 {sbid}: 光变曲线形状异常")
                    continue
                useful_count += 1

                all_mjd.extend(t_days)
                all_flux_i.extend(flux_i)
                all_flux_v.extend(flux_v)
                all_sbid_id.extend([sbid_map[sbid]] * t_len)

                print(f"  [OK] 加载 {sbid} | Beam: {beam_num} | 时长: {duration_hours:.2f}h | BJD_TDB: {t_days[0]:.4f}")

            except Exception as e:
                print(f"  [WARN] 读取 {basename} 失败: {e}")

        if useful_count == 0:
            print("[ERROR] 未找到匹配的目标 SBID 数据，请检查 TARGET_SBIDS 配置。")
            if COMBINE_SBIDS:
                sys.exit(1)
            print(f"  [WARN] {target_label} 无可用数据，跳过")
            continue

        all_mjd, all_flux_i, all_flux_v = np.array(all_mjd), np.array(all_flux_i), np.array(all_flux_v)
        all_sbid_id = np.array(all_sbid_id, dtype=int)

        # 按时间排序
        sort_idx = np.argsort(all_mjd)
        all_mjd, all_flux_i, all_flux_v, all_sbid_id = all_mjd[sort_idx], all_flux_i[sort_idx], all_flux_v[sort_idx], \
            all_sbid_id[sort_idx]

        # 剔除无效值 (NaN)
        valid = ~np.isnan(all_flux_i) & ~np.isnan(all_flux_v)
        # 与原 load_and_stitch_long_tracks() 的返回契约一致：后续周期分析
        # 只接收同时具有 I/V 有效值的积分点。
        mjd = all_mjd[valid]
        flux_i = all_flux_i[valid]
        flux_v = all_flux_v[valid]
        all_sbid_id = all_sbid_id[valid]

        print(f"\n[INFO] 数据拼接完成。共包含 {useful_count} 个观测块，有效积分点: {np.sum(valid)}")



        # ==================================
        # 4. 主程序流程：周期分析与绘图
        # ==================================

        # ==================================
        # 4. 主程序流程：周期分析与绘图
        # ==================================
        if PHASE_FOLD_CYCLES not in (1, 2):
            raise ValueError("PHASE_FOLD_CYCLES 必须为 1（mod 1 后复制）或 2（原始 mod 2）")
        if not isinstance(PHASE_NBINS, int) or PHASE_NBINS < 2:
            raise ValueError("PHASE_NBINS 必须是不小于 2 的正整数")
        if PHASE_FOLD_CYCLES == 1 and PHASE_NBINS % 2:
            raise ValueError("mod 1 复制展示要求 PHASE_NBINS 为偶数，以便左右两半均分")

        # 动态构建输出文件名后缀：单 SBID 沿用现有格式；
        # 多个 SBID 合并时在文件名中显式列出合并的 SBID。
        unique_sbids = list(sbid_map.keys())
        sbid_name = {v: k for k, v in sbid_map.items()}
        if len(unique_sbids) == 1:
            single_sbid = unique_sbids[0]
            single_beam = beam_map[single_sbid]
            file_suffix = f"_{single_sbid}_beam{single_beam}"
        else:
            file_suffix = "_" + "_".join(unique_sbids) + f"_Stitched_{len(unique_sbids)}obs"

        # --- Lomb-Scargle 周期图计算 ---
        t_centered = mjd - np.mean(mjd)
        baseline_days = mjd[-1] - mjd[0]

        # 动态周期搜索上限：单 epoch 基线只有几小时，搜到 50 天没有意义。
        # 上限取 min(PERIOD_MAX, baseline/2)，同时保证 TESS 目标周期
        # 及其 2 倍一定在搜索范围内。
        period_max_eff = min(PERIOD_MAX, max(TESS_PERIOD * 2, baseline_days / 2.0))
        if period_max_eff > baseline_days:
            print("[WARN] 搜索范围包含长于数据基线的周期；"
                  "这些结果仅用于目标周期检验，不能视为独立周期检测。")

        # 1. 统一尺度归一化：以 Stokes I 的 MAD 为共同尺度，
        #    保持 V 相对 I 的真实极化比例（修正 1）。
        med_i = np.nanmedian(flux_i)
        mad_i = np.nanmedian(np.abs(flux_i - med_i))
        if mad_i == 0:
            mad_i = 1.0
        flux_i_norm = (flux_i - med_i) / mad_i
        flux_v_norm = (flux_v - np.nanmedian(flux_v)) / mad_i
        ls_v = LombScargle(t_centered, flux_v_norm)
        frequency, power_v = ls_v.autopower(minimum_frequency=1 / period_max_eff, maximum_frequency=1 / PERIOD_MIN,
                                            samples_per_peak=15)
        periods = 1.0 / frequency

        # 2. 计算 Stokes I
        ls_i = LombScargle(t_centered, flux_i_norm)
        _, power_i = ls_i.autopower(minimum_frequency=1 / period_max_eff, maximum_frequency=1 / PERIOD_MIN,
                                    samples_per_peak=15)

        # 3. 计算窗函数 (Window Function)
        window_power = LombScargle(t_centered, np.ones_like(t_centered), fit_mean=False, center_data=False).power(frequency)

        # 分别提取最佳周期
        best_p_v = periods[np.argmax(power_v)]
        best_p_i = periods[np.argmax(power_i)]

        # FAP 计算与显示由 ENABLE_FAP 总开关控制（analytic 阈值线、数值、bootstrap）。
        # 注意：analytic FAP 基于"独立高斯白噪声"假设，对含窗函数、epoch 间隔、
        # calibration 漂移的射电数据通常过于乐观，不能作为检测周期的唯一证据。
        if ENABLE_FAP:
            fap_level_i_1pct = ls_i.false_alarm_level(FAP_PERCENT)
            fap_level_v_1pct = ls_v.false_alarm_level(FAP_PERCENT)
            try:
                fap_v = ls_v.false_alarm_probability(np.max(power_v))
                fap_i = ls_i.false_alarm_probability(np.max(power_i))
            except Exception:
                fap_v, fap_i = None, None
        else:
            fap_level_i_1pct = fap_level_v_1pct = None
            fap_v = fap_i = None

        # Bootstrap FAP：保持时间网格不变、随机重排 flux（block bootstrap），
        # 直接利用 ASKAP 真实采样结构估计"纯噪声随机产生该峰"的概率。
        compute_fap = ENABLE_FAP
        if compute_fap:
            boot_method = f"block={BOOTSTRAP_BLOCK_MIN}min" if BOOTSTRAP_BLOCK_MIN else "single-point permutation"
            n_proc = BOOTSTRAP_NPROC or (os.cpu_count() or 1)
            grid_n = min(len(frequency), BOOTSTRAP_MAX_NFREQ)
            print(f"[INFO] 正在进行 Bootstrap FAP 计算（{boot_method}，{BOOTSTRAP_ITERS} 次，"
                  f"频率网格 {grid_n} 点，{n_proc} 进程）...")
            bootstrap_results = {}
            for bootstrap_label, bootstrap_flux in (
                ("V", flux_v_norm),
                ("I", flux_i_norm),
            ):
                # LS 对点顺序不敏感；排序后按真实时间连续性切 block。
                bootstrap_t = np.asarray(t_centered, dtype=float)
                bootstrap_flux = np.asarray(bootstrap_flux, dtype=float)
                bootstrap_frequency = np.asarray(frequency, dtype=float)
                order = np.argsort(bootstrap_t)
                bootstrap_t = bootstrap_t[order]
                bootstrap_flux = bootstrap_flux[order]
                bootstrap_epoch_ids = np.asarray(all_sbid_id)[order]

                if len(bootstrap_frequency) > BOOTSTRAP_MAX_NFREQ:
                    step = max(1, len(bootstrap_frequency) // BOOTSTRAP_MAX_NFREQ)
                    bootstrap_frequency = bootstrap_frequency[::step]
                power_obs = np.max(
                    LombScargle(bootstrap_t, bootstrap_flux).power(bootstrap_frequency)
                )
                n_bootstrap = len(bootstrap_flux)
                n_iters = int(BOOTSTRAP_ITERS)

                epoch_bounds = []
                max_epoch_len = 0
                for epoch in np.unique(bootstrap_epoch_ids):
                    indices = np.nonzero(bootstrap_epoch_ids == epoch)[0]
                    if len(indices):
                        epoch_bounds.append((int(indices[0]), int(indices[-1]) + 1))
                        max_epoch_len = max(max_epoch_len, len(indices))

                if BOOTSTRAP_BLOCK_MIN is None:
                    block_size = 1
                else:
                    dt_days = (
                        float(np.median(np.diff(bootstrap_t)))
                        if n_bootstrap >= 2 else 1.0
                    )
                    block_size = max(
                        1,
                        int(round((BOOTSTRAP_BLOCK_MIN / 1440.0) / dt_days)),
                    )
                    if block_size >= max_epoch_len:
                        block_size = 1

                if n_proc > 1 and n_iters >= n_proc:
                    base = n_iters // n_proc
                    remainder = n_iters - base * n_proc
                    seed_chunks = []
                    seed_index = 0
                    for worker_index in range(n_proc):
                        chunk_size = base + (1 if worker_index < remainder else 0)
                        seed_chunks.append([
                            BOOTSTRAP_RANDOM_SEED + seed_index + offset
                            for offset in range(chunk_size)
                        ])
                        seed_index += chunk_size
                    tasks = [
                        (
                            bootstrap_t,
                            bootstrap_flux,
                            bootstrap_frequency,
                            block_size,
                            epoch_bounds,
                            seeds,
                            power_obs,
                        )
                        for seeds in seed_chunks
                    ]
                    try:
                        try:
                            context = mp.get_context("fork")
                        except ValueError:
                            context = mp.get_context()
                        with context.Pool(n_proc) as pool:
                            counts = pool.map(_bootstrap_chunk, tasks)
                        n_exceed = sum(counts)
                    except Exception as error:
                        print(f"  [WARN] 多进程 bootstrap 失败（{error}），退回单进程计算")
                        n_exceed = 0
                        for seeds in seed_chunks:
                            n_exceed += _bootstrap_chunk(
                                (
                                    bootstrap_t,
                                    bootstrap_flux,
                                    bootstrap_frequency,
                                    block_size,
                                    epoch_bounds,
                                    seeds,
                                    power_obs,
                                )
                            )
                else:
                    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
                    n_exceed = 0
                    for _ in range(n_iters):
                        if block_size <= 1:
                            shuffled = bootstrap_flux[rng.permutation(n_bootstrap)]
                        else:
                            shuffled_chunks = []
                            for start, end in epoch_bounds:
                                epoch_flux = bootstrap_flux[start:end]
                                chunks = [
                                    epoch_flux[index:index + block_size]
                                    for index in range(0, len(epoch_flux), block_size)
                                ]
                                rng.shuffle(chunks)
                                shuffled_chunks.append(np.concatenate(chunks))
                            shuffled = np.concatenate(shuffled_chunks)
                        if np.max(
                            LombScargle(bootstrap_t, shuffled).power(bootstrap_frequency)
                        ) >= power_obs:
                            n_exceed += 1
                bootstrap_results[bootstrap_label] = (
                    (n_exceed + 1) / (n_iters + 1),
                    power_obs,
                )
            boot_fap_v, peak_power_v = bootstrap_results["V"]
            boot_fap_i, peak_power_i = bootstrap_results["I"]
        else:
            print("[INFO] 已关闭 FAP 计算与显示（如需开启：ENABLE_FAP = True）")
            boot_fap_v = boot_fap_i = None

        print(f"\n[INFO] === 射电周期拟合结果 ===")
        print(f"   TESS 周期: {TESS_PERIOD:.5f} d")
        print(f"   TESS 半周期  : {TESS_PERIOD / 2.0:.5f} d")
        print(f"   Stokes V: {best_p_v:.5f} d,  FAP = {fap_v:.2e}" if fap_v is not None else f"   Stokes V: {best_p_v:.5f} d")
        print(f"   Stokes I: {best_p_i:.5f} d,  FAP = {fap_i:.2e}" if fap_i is not None else f"   Stokes I: {best_p_i:.5f} d")
        print(f"   数据基线: {baseline_days:.2f} d, V 覆盖 {baseline_days/best_p_v:.1f} 个周期")
        if compute_fap:
            print(f"   Bootstrap FAP:  Stokes V = {boot_fap_v:.4f},  Stokes I = {boot_fap_i:.4f}")
        print(f"=====================================\n")

        # Leave-One-Epoch-Out 一致性检验：判断检测是否被单一 epoch 驱动
        if len(unique_sbids) > 1:
            print("[INFO] === Leave-One-Epoch-Out 一致性检验 ===")
            loeo_rows = []
            for epoch in np.unique(all_sbid_id):
                mask = all_sbid_id != epoch
                if mask.sum() < 5:
                    continue
                t_sub = t_centered[mask]
                flux_i_sub = flux_i_norm[mask]
                flux_v_sub = flux_v_norm[mask]
                power_i_sub = LombScargle(t_sub, flux_i_sub).power(frequency)
                power_v_sub = LombScargle(t_sub, flux_v_sub).power(frequency)
                loeo_rows.append(
                    (
                        int(epoch),
                        1.0 / frequency[np.argmax(power_i_sub)],
                        np.max(power_i_sub),
                        1.0 / frequency[np.argmax(power_v_sub)],
                        np.max(power_v_sub),
                    )
                )
            if loeo_rows:
                print(f"   {'剔除 epoch':<12} {'I 最佳周期(d)':>14} {'I 峰值':>10} {'V 最佳周期(d)':>14} {'V 峰值':>10}")
                for epoch, p_i, pow_i, p_v, pow_v in loeo_rows:
                    print(f"   {sbid_name.get(epoch, epoch):<12} {p_i:>14.5f} {pow_i:>10.4f} {p_v:>14.5f} {pow_v:>10.4f}")
                loeo_csv = os.path.join(MASTER_OUTPUT_DIR, f"{SOURCE_NAME}{file_suffix}_LOEO.csv")
                pd.DataFrame(loeo_rows, columns=["dropped_epoch", "best_P_I_d", "power_I", "best_P_V_d", "power_V"]).to_csv(
                    loeo_csv, index=False, encoding="utf-8")
                print(f"   已保存: {loeo_csv}")
            else:
                print("   [WARN] 每个 epoch 样本过少，跳过 Leave-One-Epoch-Out 检验。")
            print("=====================================\n")
        else:
            print("[INFO] 单 epoch 数据，跳过 Leave-One-Epoch-Out 检验。\n")

        # ---------------------------------------------
        # 绘图 1：Lomb-Scargle 周期图（Stokes I / Stokes V 上下双栏）
        # ---------------------------------------------
        fig, (ax_i, ax_v) = plt.subplots(
            2, 1, figsize=(11, 8.5), dpi=FIGURE_DPI, sharex=True,
            gridspec_kw={"hspace": 0},
        )
        add_panel_label(ax_i, "Stokes I | Lomb–Scargle", fontsize=11)
        add_panel_label(ax_v, "Stokes V | Lomb–Scargle", fontsize=11)

        # 两个面板共用的数据：窗函数、TESS 参考周期线、X 轴范围与网格
        for ax in (ax_i, ax_v):
            ax.set_zorder(10)
            ax.patch.set_visible(False)

            # 多 epoch 时才画窗函数
            if len(unique_sbids) > 1:
                axw = ax.twinx()
                axw.fill_between(periods, 0, window_power, color="gray", alpha=0.12)
                axw.plot(periods, window_power, color="gray", linewidth=0.6, alpha=0.3)
                axw.set_ylabel("Window Function Power", color="gray", fontsize=9)
                axw.set_ylim(0, max(1.1, np.max(window_power) * 1.2))
                axw.tick_params(axis='y', labelcolor="gray", labelsize=8)
                axw.yaxis.set_minor_locator(AutoMinorLocator(2))

            # 标记参考周期线（含 TESS 周期不确定度 band，判断 LS 峰是否偏离）
            ax.axvspan(TESS_PERIOD - TESS_PERIOD_SIGMA * TESS_PERIOD_ERR,
                       TESS_PERIOD + TESS_PERIOD_SIGMA * TESS_PERIOD_ERR,
                       color="#cc3333", alpha=0.15,
                       label=f"TESS Period ± {TESS_PERIOD_SIGMA}σ = "
                             f"{TESS_PERIOD:.4f} ± {TESS_PERIOD_SIGMA * TESS_PERIOD_ERR:.4f} d")
            ax.axvline(x=TESS_PERIOD, color="#cc3333", linestyle="-.", linewidth=1.8,
                       label=f"TESS Period = {TESS_PERIOD:.4f} d")
            ax.axvspan(TESS_PERIOD / 2.0 - TESS_PERIOD_SIGMA * TESS_PERIOD_ERR / 2.0,
                       TESS_PERIOD / 2.0 + TESS_PERIOD_SIGMA * TESS_PERIOD_ERR / 2.0,
                       color="#cc3333", alpha=0.12, label="TESS Half-Period ± 3σ")
            ax.axvline(x=TESS_PERIOD / 2.0, color="#cc3333", linestyle=":", linewidth=1.5,
                       label=f"TESS Half Period = {TESS_PERIOD / 2.0:.4f} d")

            # 动态自适应 X 轴范围，保证最佳周期竖线一定在画面内。
            plot_x_max = max(0.5, best_p_i * 1.5, best_p_v * 1.5)
            x_min = PERIOD_MIN
            x_max = min(plot_x_max, period_max_eff)
            x_scale = "log" if plot_x_max > 5.0 else "linear"
            ax.set_xlim(x_min, x_max)
            if x_scale == "log":
                ax.set_xscale('log')
                ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))

            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(True, which='major', color='gray', linestyle='-', alpha=0.3)
            ax.grid(False, which='minor')

        # 上图：Stokes I（偏振专属数据）
        if compute_fap:
            ax_i.axhline(fap_level_i_1pct, color='steelblue', linestyle=':', linewidth=1.2, alpha=0.8,
                         label=f"I {FAP_PERCENT * 100:g}% FAP")
        ax_i.plot(periods, power_i, color="steelblue", linewidth=1.0, alpha=0.6, label="Radio Stokes I")
        ax_i.axvline(x=best_p_i, color="steelblue", linestyle="--", linewidth=1.5,
                     label=f"Stokes I LS Peak = {best_p_i:.4f} d")
        ax_i.set_ylabel("Lomb-Scargle Power (Stokes I)", fontweight='bold')
        ax_i.legend(fontsize=9, loc="upper right")
        if compute_fap:
            ax_i.text(0.02, 0.95, f"Bootstrap FAP = {boot_fap_i:.3g}", transform=ax_i.transAxes, fontsize=9, va="top",
                      bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        # 下图：Stokes V（偏振专属数据）
        if compute_fap:
            ax_v.axhline(fap_level_v_1pct, color='darkorange', linestyle=':', linewidth=1.2, alpha=0.8,
                         label=f"V {FAP_PERCENT * 100:g}% FAP")
        ax_v.plot(periods, power_v, color="darkorange", linewidth=1.5, label="Radio Stokes V")
        ax_v.axvline(x=best_p_v, color="darkorange", linestyle="--", linewidth=1.5,
                     label=f"Stokes V LS Peak = {best_p_v:.4f} d")
        ax_v.set_ylabel("Lomb-Scargle Power (Stokes V)", fontweight='bold')
        if x_scale == "log":
            ax_v.set_xlabel("Period (Days) [Log Scale]", fontweight='bold')
        else:
            ax_v.set_xlabel("Period (Days)", fontweight='bold')
            ax_v.xaxis.set_major_locator(MultipleLocator(0.05))
            ax_v.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax_v.legend(fontsize=9, loc="upper right")
        if compute_fap:
            ax_v.text(0.02, 0.95, f"Bootstrap FAP = {boot_fap_v:.3g}", transform=ax_v.transAxes, fontsize=9, va="top",
                      bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        out_ls = os.path.join(MASTER_OUTPUT_DIR, f"{SOURCE_NAME}{file_suffix}_LS.png")
        fig.subplots_adjust(top=0.96, bottom=0.07, hspace=0)
        fig.savefig(out_ls, dpi=FIGURE_DPI)
        plt.close(fig)

        # ---------------------------------------------
        # 绘图 2：相位折叠图 (Phase Folding)
        # ---------------------------------------------
        # 将折叠目标增加为 4 个，分别验证 I 和 V 的拟合结果
        fold_targets = [
            ("TESS Period", TESS_PERIOD, "#cc3333"),
            ("TESS Half-Period", TESS_PERIOD / 2.0, "#cc3333"),
            ("Stokes I LS Peak", best_p_i, "steelblue"),
            ("Stokes V LS Peak", best_p_v, "darkorange")
        ]

        fig, axes = plt.subplots(
            len(fold_targets), 2, figsize=(14, 4.2 * len(fold_targets)), dpi=FIGURE_DPI,
            gridspec_kw={"hspace": 0, "wspace": 0.05},
        )

        for row, (label, p_val, row_color) in enumerate(fold_targets):
            for col, (flux_data, name, alpha_val) in enumerate(
                    [(flux_i, "Stokes I", 0.6), (flux_v, "Stokes V", 1.0)]):
                ax = axes[row, col]

                if p_val <= 0:
                    ax.text(0.5, 0.5, "Invalid Period", ha='center', va='center')
                    continue

                t_ref = np.min(mjd)
                if PHASE_FOLD_CYCLES == 1:
                    # 规范展示：主相位 [0, 1) 折叠后原样复制到右半区。
                    phase_base = np.mod((mjd - t_ref) / p_val, 1.0)
                    phase_full = np.concatenate((phase_base, phase_base + 1.0))
                    flux_plot = np.concatenate((flux_data, flux_data))
                    phase_epoch_ids = np.concatenate((all_sbid_id, all_sbid_id))
                else:
                    # 诊断展示：保留相邻两个实际周期，不进行复制。
                    phase_full = np.mod((mjd - t_ref) / p_val, 2.0)
                    flux_plot = flux_data
                    phase_epoch_ids = all_sbid_id

                ax.scatter(phase_full, flux_plot, c=row_color, s=6, alpha=0.25 * alpha_val,
                           edgecolors="none", rasterized=True)

                # 与 Radio_MultipleEpoch_PhaseFolding.py 保持同一折叠语义：
                # mod 1 先在 [0, 1) 用一半的总分箱数分箱，再复制到 [1, 2)；
                # mod 2 直接在 [0, 2) 分箱。这样两种模式的每个展示周期点密度一致。
                epochs = np.unique(phase_epoch_ids)
                if PHASE_FOLD_CYCLES == 1:
                    display_bins = PHASE_NBINS // 2
                    display_phase_max = 1.0
                else:
                    display_bins = PHASE_NBINS
                    display_phase_max = 2.0

                combined_phase, combined_values, combined_errors = phase_bin_for_display(
                    phase_full,
                    flux_plot,
                    PHASE_NBINS,
                    min_count=3,
                    phase_max=2.0,
                )
                bin_matrix = []
                err_matrix = []
                for epoch_index, epoch in enumerate(epochs):
                    if PHASE_FOLD_CYCLES == 1:
                        # 每个真实 epoch 只用一次 [0, 1) 数据分箱；
                        # 之后复制分箱结果，避免复制原始点导致误差被人为缩小。
                        epoch_mask = all_sbid_id == epoch
                        epoch_phase = phase_base[epoch_mask]
                        epoch_flux = flux_data[epoch_mask]
                    else:
                        epoch_mask = phase_epoch_ids == epoch
                        epoch_phase = phase_full[epoch_mask]
                        epoch_flux = flux_plot[epoch_mask]
                    epoch_phase, epoch_values, epoch_errors = phase_bin_for_display(
                        epoch_phase,
                        epoch_flux,
                        display_bins,
                        min_count=3,
                        phase_max=display_phase_max,
                    )
                    if PHASE_FOLD_CYCLES == 1:
                        epoch_phase = np.concatenate((epoch_phase, epoch_phase + 1.0))
                        epoch_values = np.concatenate((epoch_values, epoch_values))
                        epoch_errors = np.concatenate((epoch_errors, epoch_errors))
                    bin_matrix.append((epoch_phase, epoch_values))
                    err_matrix.append(epoch_errors)

                if len(combined_phase):
                    ax.errorbar(
                        combined_phase,
                        combined_values,
                        yerr=combined_errors,
                        fmt="o",
                        color="black",
                        markersize=4.5,
                        linewidth=1.1,
                        capsize=2,
                        zorder=10,
                        label="Binned Avg",
                    )
                for epoch_index, epoch in enumerate(epochs):
                    epoch_phase, epoch_values = bin_matrix[epoch_index]
                    epoch_errors = err_matrix[epoch_index]
                    if not len(epoch_phase):
                        continue
                    epoch_label = sbid_name.get(int(epoch), f"Epoch {int(epoch)}")
                    epoch_color = plt.cm.tab10(epoch_index % 10)
                    ax.errorbar(
                        epoch_phase,
                        epoch_values,
                        yerr=epoch_errors,
                        fmt="s",
                        color=epoch_color,
                        markersize=3.5,
                        linewidth=0.9,
                        capsize=2,
                        label=epoch_label,
                    )

                ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
                phase_xlabel = ("Folded phase (mod 1; repeated)" if PHASE_FOLD_CYCLES == 1
                               else "Cycle phase (mod 2)")
                if row == len(fold_targets) - 1:
                    ax.set_xlabel(phase_xlabel)
                    ax.tick_params(axis="x", labelbottom=True)
                else:
                    ax.set_xlabel("")
                    ax.tick_params(axis="x", labelbottom=False)
                ax.set_ylabel("Flux (mJy)")
                add_panel_label(ax, f"{name} | {label}", fontsize=10)
                ax.grid(True, alpha=0.2)
                ax.legend(fontsize=8, loc="upper right")

                # 根据数据分布自动调整 Y 轴范围，剔除极端离群值
                lo, hi = np.percentile(flux_data[np.isfinite(flux_data)], [0.5, 99.5])
                span = hi - lo
                ax.set_ylim(lo - 0.2 * span, hi + 0.2 * span)

        # 统一所有面板的 X 轴范围
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlim(0, 2)

        if PHASE_FOLD_CYCLES == 1:
            phase_display = "mod 1 folded; [0, 1) repeated in [1, 2)"
            phase_bins_note = f"{PHASE_NBINS // 2} bins per repeated half"
            fold_tag = "mod1_repeated"
        else:
            phase_display = "raw phase mod 2; two actual consecutive cycles"
            phase_bins_note = f"{PHASE_NBINS} bins across [0, 2)"
            fold_tag = "mod2"
        fig.suptitle(
            f"{SOURCE_NAME} {file_suffix} | phase: mod {PHASE_FOLD_CYCLES}",
            fontsize=18, fontweight="bold", y=0.99,
        )
        fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0, wspace=0.05)
        out_fold = os.path.join(
            MASTER_OUTPUT_DIR,
            f"{SOURCE_NAME}{file_suffix}_Folding_{fold_tag}_bins{PHASE_NBINS}.png",
        )
        fig.savefig(out_fold, dpi=FIGURE_DPI)
        plt.close(fig)

        print(f"\n[INFO] 绘图完成。\n 周期图: {out_ls}\n "
              f"折叠图（mod {PHASE_FOLD_CYCLES}; {phase_display}; "
              f"{phase_bins_note}）: {out_fold}")


if __name__ == "__main__":
    main()
