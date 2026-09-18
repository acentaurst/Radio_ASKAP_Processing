"""Align selected ASKAP epochs with a TESS ephemeris and render phase-folded comparisons."""

import os
import re
import sys
import glob
import warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
# macOS 优先使用系统中文字体；其他平台回退至 Unicode/DejaVu 字体。
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import astropy.units as u
import h5py
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from science_utils import (  # noqa: E402
    add_panel_label,
    as_text,
    phase_bin_for_display,
    propagated_phase_uncertainty,
    rebin,
    rebin2d,
    slice_open_end,
)

# 当前脚本只读取已生成的 TESS light curve，不调用 PRF 模型；屏蔽该可选 PRF
# 依赖（oktopus）的导入提示，避免与实际数据加载失败混淆。
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"Warning: the tpfmodel submodule is not available.*",
        category=UserWarning,
        module=r"lightkurve\.prf",
    )
    import lightkurve as lk
ASKAP_LOCATION = EarthLocation.from_geodetic(
    lon=116.631425 * u.deg,
    lat=-26.697000 * u.deg,
    height=361.0 * u.m,
)
iers.conf.auto_download = False
iers.conf.auto_max_age = None


# ============================================================
# 配置区
# ============================================================
FIGURE_LAYOUT = "phase_aligned"   # "phase_aligned" 或 "epoch_stack"
# 与 Radio_LS_PhaseFolding.py 保持一致：
#   1 = [0,1) 折叠后复制到 [1,2)，用于规范的两周期展示；
#   2 = 原始相位直接 mod 2，不复制数据，用于查看相邻实际周期差异。
PHASE_FOLD_CYCLES = 2
PHASE_DISPLAY_MODE = ("mod1" if PHASE_FOLD_CYCLES == 1 else "mod2")  # 只读兼容标签

# 周期结果表与 Bootstrap 样本必须来自同一次 TESS_Period 运行。
EPHEMERIS_RESULT_DIR = Path(
    "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/"
    "TESS_Period/s29+69"
)
PERIOD_RESULT_FILE = EPHEMERIS_RESULT_DIR / "TESS_Period_Result.md"
# 与中心星历配对的 T0/P bootstrap 样本；用于传播光学极小值的相位置信带。
EPHEMERIS_SAMPLES_FILE = EPHEMERIS_RESULT_DIR / "TESS_Period_Bootstrap.txt"
EPHEMERIS_ALIAS_RANK = 1  # 只传播与中心星历相同的 alias，避免把竞争周期混入置信带
BAND_CONFIDENCE_LEVEL = 0.95
# "per_epoch"：每个射电历元从自身起始时刻定义相位零点，适合周期不确定时比较形状。
# "absolute_ephemeris"：所有历元共用同一绝对零点，仅在已有足够精确的外推星历时使用。
PHASE_REFERENCE_MODE = "absolute_ephemeris"
ABSOLUTE_PHASE_WARN_CYCLES = 0.25  # 相位外推误差超过此值时给出警告
BIN_SEC = 210              # 射电时间分箱（3.5 min）
TESS_PHASE_BINS = 160      # [0,2) 展示范围的总相位分箱数；mod 1 每周期 50 箱后复制
MIN_I_SNR = 3.0            # |V|/I 质量筛选：I/I_err 的最低值
MAX_POLARIZATION_ERROR_PERCENT = 100.0
XLO, XHI = 0.0, 2.0        # epoch_stack 旧布局的双周期显示范围

# epoch_stack 旧布局仍使用真实相位 mod 2（点不重复，x 轴显示两个连续周期）；
# phase_aligned 新布局使用共同星历 mod 1，并在内存结果中保留 cycle_index。

# ds.time 是否为绝对时间（决定 read_ds 是否还要加 header time_start）：
#   None = 自动探测（按量级判定 JD/MJD/相对，自动时打印 [WARN] 提示核对并锁定）
#   True = ds.time 已是绝对时间（不再加 time_start）
#   False = ds.time 是相对时间（用 t0 + ds.time*tunit）
# 已用 SB66827、SB68040 验证：ds.time 从 0 h 起算，需与 header time_start 相加。
DS_TIME_ABSOLUTE = False

# ── TESS ──
# TESS_FILE = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Data/TESS_Data/2MASS_J01033563-5515561_A/hlsp_qlp_tess_ffi_s0069-0000000616014335_tess_v01_llc/hlsp_qlp_tess_ffi_s0069-0000000616014335_tess_v01_llc.fits"
TESS_FILE = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/TESS_Data/2MASS_J01033563-5515561_A/hlsp_qlp_tess_ffi_s0069-0000000616014335_tess_v01_llc/hlsp_qlp_tess_ffi_s0069-0000000616014335_tess_v01_llc.fits"
INCLUDE_TESS = True         # 是否在顶部加入 TESS 相位折叠面板

# ── 射电 DS ──
# 三个绘图槽位固定标记为 Blue、Red、Orange；path 留空时该槽位不读取、不绘制。
# label 只表示颜色角色；图例文字和输出文件名均从实际 .ds 文件名自动提取。
DS_FILES = [
    {
        "path": "",
        # /Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/Ds/2MASS_J01033563-5515561_A/Flare/2MASS_J01033563-5515561_A_SB66827_beam10.ds
        "label": "Blue",
    },
    {
        "path": "",
        # /Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/Ds/2MASS_J01033563-5515561_A/Flare/2MASS_J01033563-5515561_A_SB68040_beam10.ds
        "label": "Red",
    },
    {
        "path": "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/Ds/2MASS_J01033563-5515561_A/Flare/2MASS_J01033563-5515561_A_SB59565_beam22.ds",
        # /Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/Ds/2MASS_J01033563-5515561_A/Flare/2MASS_J01033563-5515561_A_SB59565_beam22.ds
        "label": "Orange",
    },
]
BATCH_PROCESS = False
DS_FILES_DIR = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/Ds/2MASS_J01033563-5515561_A/Flare"
SOURCE_FILTER = ""

# OUTPUT_BASE = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Result/Radio_MultipleEpoch_PhaseFolding"
OUTPUT_BASE = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/Radio_MultipleEpoch_PhaseFolding"
ASKAP_CATALOGUE_CSV = PROJECT_ROOT / "Processed_Data" / "Catalogue" / "01.askap_catalogue.csv"

# 绘图尺寸与输出质量（不改变科学计算）
FIG_W = 12
PANEL_H = 3.3
LINE_LW = 1.5
ERR_MS = 3
ERR_ELW = 0.8
DPI = 300
SHOW_OPTICAL_FEATURE_BAND = False
OPTICAL_FEATURE_PHASE_INTERVALS = []
SHOW_OPTICAL_MINIMUM = True       # 标出当前 TESS 模板的最小值相位
SHOW_OPTICAL_MINIMUM_BAND = True  # 有 paired T0/P 样本时绘制相位置信带
# ============================================================


def main() -> None:
    if PHASE_REFERENCE_MODE not in {"per_epoch", "absolute_ephemeris"}:
        raise ValueError("PHASE_REFERENCE_MODE 必须为 'per_epoch' 或 'absolute_ephemeris'")

    # 从结果表读取所选 alias 的中心周期、参考时刻和 68% 周期误差，避免
    # 手工填写的旧星历与 Bootstrap 样本不匹配。
    if not PERIOD_RESULT_FILE.is_file():
        raise FileNotFoundError(f"周期结果表不存在：{PERIOD_RESULT_FILE}")
    if not EPHEMERIS_SAMPLES_FILE.is_file():
        raise FileNotFoundError(f"星历 Bootstrap 样本不存在：{EPHEMERIS_SAMPLES_FILE}")
    result_lines = PERIOD_RESULT_FILE.read_text(encoding="utf-8").splitlines()
    header_index = next(
        (index for index, line in enumerate(result_lines)
         if line.strip().startswith("| Alias rank |")),
        None,
    )
    if header_index is None or header_index + 2 >= len(result_lines):
        raise ValueError(f"周期结果表缺少 alias 结果表：{PERIOD_RESULT_FILE}")
    headers = [cell.strip() for cell in result_lines[header_index].strip().strip("|").split("|")]
    selected_ephemeris = None
    for line in result_lines[header_index + 2:]:
        if not line.strip().startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        row = dict(zip(headers, cells, strict=True))
        if row.get("Alias rank") == str(EPHEMERIS_ALIAS_RANK):
            selected_ephemeris = row
            break
    if selected_ephemeris is None:
        raise ValueError(
            f"周期结果表中没有 alias rank={EPHEMERIS_ALIAS_RANK}：{PERIOD_RESULT_FILE}"
        )

    P = float(selected_ephemeris["Period (d)"])
    ephemeris_t0_bjd_tdb = float(selected_ephemeris["T0 (BJD_TDB)"])
    period_error_seconds = max(
        float(selected_ephemeris["68% error - (s)"]),
        float(selected_ephemeris["68% error + (s)"]),
    )
    period_error_days = period_error_seconds / 86400.0
    selected_delta_bic = float(selected_ephemeris["Delta BIC"])
    ephemeris_label = (
        f"{EPHEMERIS_RESULT_DIR.name}, alias rank {EPHEMERIS_ALIAS_RANK} (conditional)"
    )

    print("=" * 60)
    print(" Multi-Epoch Phase Folding (TESS + Radio)")
    print(f" Period: {P:.6f} d ({P*24:.4f} h)")
    print(f" T0: {ephemeris_t0_bjd_tdb:.9f} BJD_TDB; alias rank={EPHEMERIS_ALIAS_RANK}; "
          f"Delta BIC={selected_delta_bic:.3f}")
    print(f" Phase reference: {PHASE_REFERENCE_MODE}")
    if FIGURE_LAYOUT == "phase_aligned":
        print(f" Phase display: {'mod 1 repeated to [0,2)' if PHASE_FOLD_CYCLES == 1 else 'raw mod 2 [0,2)'}")
    print("=" * 60)

    # ── 加载 TESS ──
    tess_folded = None
    tess_jd = tess_flux = t0_tess = tess_sector = None
    if INCLUDE_TESS and os.path.exists(TESS_FILE):
        try:
            # 读取、清洗并归一化 TESS 光变曲线；该逻辑仅在本脚本中使用一次，直接保留在主流程。
            lc = lk.read(TESS_FILE)
            metadata = getattr(lc, "meta", {}) or {}
            tess_sector = metadata.get("SECTOR", metadata.get("SECTOR_NUMBER"))
            if tess_sector is None:
                sector_match = re.search(
                    r"s(\d{4})", os.path.basename(TESS_FILE), re.IGNORECASE
                )
                tess_sector = int(sector_match.group(1)) if sector_match else "unknown"
            if hasattr(lc, "to_lightcurve"):
                lc = lc.to_lightcurve(aperture_mask="pipeline")
            lc = lc.remove_nans().remove_outliers(sigma=5)
            try:
                lc = lc.flatten(window_length=401)
            except Exception:
                pass
            flux_median = float(np.nanmedian(lc.flux.value))
            lc_norm = lc / flux_median
            # 射电时间也转换为 BJD_TDB；两类数据必须使用同一时间标准。
            tess_jd = lc_norm.time.tdb.jd
            tess_flux = lc_norm.flux.value
            t0_tess = float(lc_norm.time[0].tdb.jd)
            print(f"TESS S{tess_sector}: {len(tess_jd)} pts, JD_TDB range "
                  f"[{tess_jd[0]:.4f}, {tess_jd[-1]:.4f}]")
        except Exception as e:
            print(f"[WARN] TESS 加载失败: {e}")
            tess_folded = None
    elif not INCLUDE_TESS:
        print("[INFO] 跳过 TESS")
    else:
        print(f"[WARN] TESS 文件不存在，未绘制 TESS 面板：{TESS_FILE}")

    # ── 加载射电 ──
    if BATCH_PROCESS:
        discovered_paths = sorted(glob.glob(os.path.join(DS_FILES_DIR, "*.ds")))
        if SOURCE_FILTER:
            discovered_paths = [
                path for path in discovered_paths
                if SOURCE_FILTER in os.path.basename(path)
            ]
        if len(discovered_paths) > 3:
            raise ValueError("三色叠加图最多接受 3 个 .ds 文件；请用 SOURCE_FILTER 缩小范围")
        configured_ds = [
            {"path": path, "color_label": ("Blue", "Red", "Orange")[slot],
             "plot_slot": slot}
            for slot, path in enumerate(discovered_paths)
        ]
    else:
        if len(DS_FILES) != 3:
            raise ValueError("DS_FILES 必须保留 3 个 {path, label} 配置槽位")
        configured_ds = []
        for plot_slot, item in enumerate(DS_FILES):
            if not isinstance(item, dict):
                raise TypeError("DS_FILES 的每一项必须包含 path 和 label")
            ds_path = str(item.get("path", "")).strip()
            if not ds_path:
                continue
            color_label = str(item.get("label", "")).strip()
            expected_color_label = ("Blue", "Red", "Orange")[plot_slot]
            if not color_label:
                color_label = expected_color_label
            if color_label != expected_color_label:
                raise ValueError(
                    f"DS_FILES 第 {plot_slot + 1} 个槽位 label 必须为 "
                    f"{expected_color_label}，当前为 {color_label}"
                )
            configured_ds.append({
                "path": ds_path,
                "color_label": color_label,
                "plot_slot": plot_slot,
            })

    if not configured_ds:
        print("[ERROR] 三个 .ds 路径均为空"); sys.exit(1)

    print(f"找到 {len(configured_ds)} 个已配置的 .ds 文件")

    epochs = []
    for item in configured_ds:
        ds_file = item["path"]
        basename = os.path.basename(ds_file)
        sb_match = re.search(r'SB(\d+)', basename, re.IGNORECASE)
        if sb_match is None:
            print(f"  [WARN] 文件名无法识别 SBID，跳过：{basename}")
            continue
        sbid = sb_match.group(1)
        # 图例文字严格使用文件名自动提取的 SBID，不使用手写颜色 label。
        plot_label = f"SB{sbid}"
        try:
            # `.ds` 读取和时间/频率积分逻辑只服务于此处的 epoch 装载，
            # 直接展开原 DynamicSpectrum 的 HDF5、校准扫描拼接、重采样
            # 和 Stokes 组合逻辑；不保留低复用类方法。
            tunit = u.hour
            tavg = favg = 1
            required = {"flux", "time", "frequency", "uvdist"}
            with h5py.File(ds_file, "r") as data_file:
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
                raise ValueError(f"Invalid phasecentre attribute: {phasecentre!r}") from error
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
                        pol_chunks = [np.ma.vstack((chunk, nan_chunk)) for chunk in pol_chunks]
                        break_start = time[end] + cadence
                        break_times = break_start + np.arange(number_of_steps) * cadence
                        time_chunk = np.append(time_chunk, break_times)
                    for chunk_list, chunk in zip(chunks_by_pol, pol_chunks):
                        chunk_list.append(chunk)
                    time_chunks.append(time_chunk)
                time = np.concatenate(time_chunks)
                xx, xy, yx, yy = (np.ma.vstack(chunks) for chunks in chunks_by_pol)

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
                raise ValueError(f"Feed type {feed_type!r} is not 'linear' or 'circular'")
            ds = SimpleNamespace(
                time=time,
                tunit=tunit,
                header=header,
                freq=freq,
                data={
                    "XX": xx,
                    "XY": xy,
                    "YX": yx,
                    "YY": yy,
                    "I": intensity,
                    "Q": q_stokes,
                    "U": u_stokes,
                    "V": v_stokes,
                    "L": q_stokes.real + 1j * u_stokes.real,
                },
            )
            i_data = np.asarray(ds.data["I"].real)
            v_data = np.asarray(ds.data["V"].real)
            t_len = len(ds.time)

            # 诊断 ds.time 与文件头起始时刻的相对/绝对解释，便于复核时间标准。
            time_vals = np.asarray(ds.time, dtype=float)
            try:
                rel_days = (time_vals * ds.tunit).to_value("day")
            except Exception:
                # 兼容旧 `.ds` 文件将 tunit 保存为普通字符串/数值且 time 为小时的情况。
                rel_days = time_vals / 24.0
            print(
                f"    [DBG] ds.time=[{time_vals[0]:.4f}, {time_vals[-1]:.4f}]  "
                f"tunit={ds.tunit}  header.time_start={ds.header.get('time_start')}"
            )

            # 动态识别时间轴，不假设数据一定为 (time, frequency)。
            if i_data.shape[0] == t_len:
                freq_axis = 1
            elif i_data.shape[1] == t_len:
                freq_axis = 0
            else:
                raise ValueError(
                    f"无法识别数据轴：I shape={i_data.shape}, time length={t_len}"
                )
            valid_i = np.any(np.isfinite(i_data), axis=freq_axis)
            valid_v = np.any(np.isfinite(v_data), axis=freq_axis)
            valid_time = valid_i & valid_v
            n_invalid = int(np.count_nonzero(~valid_time))
            if n_invalid:
                print(
                    f"    [WARN] 跳过 {n_invalid}/{t_len} 个全频段无有效 I/V 数据的时间积分。"
                )
            I_t = np.full(t_len, np.nan)
            V_t = np.full(t_len, np.nan)
            # 仅抑制全频段为空时由 nanmean 产生的已知 RuntimeWarning；
            # 数据有效性筛选和结果数组仍保持原样。
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if freq_axis == 1:
                    I_t[valid_time] = np.nanmean(i_data[valid_time, :], axis=1)
                    V_t[valid_time] = np.nanmean(v_data[valid_time, :], axis=1)
                else:
                    I_t[valid_time] = np.nanmean(i_data[:, valid_time], axis=0)
                    V_t[valid_time] = np.nanmean(v_data[:, valid_time], axis=0)

            t0 = Time(
                ds.header["time_start"],
                scale=str(ds.header.get("time_scale", "utc")).lower(),
            )
            if DS_TIME_ABSOLUTE is None:
                max_days = float(np.nanmax(np.abs(rel_days))) if rel_days.size else 0.0
                if max_days > 2.4e6:
                    abs_mode, fmt = True, "jd"
                elif max_days > 3.0e4:
                    abs_mode, fmt = True, "mjd"
                else:
                    abs_mode, fmt = False, None
                print(
                    f"    [WARN] DS_TIME_ABSOLUTE 未显式设置，自动判定为 "
                    f"{'绝对时间 (' + fmt + ')' if abs_mode else '相对时间'}；"
                    "请对照上方 [DBG] 输出核对并在配置中锁定 DS_TIME_ABSOLUTE。"
                )
            elif DS_TIME_ABSOLUTE:
                max_days = float(np.nanmax(np.abs(rel_days))) if rel_days.size else 0.0
                fmt = "jd" if max_days > 2.4e6 else "mjd"
                abs_mode = True
            else:
                abs_mode, fmt = False, None

            if abs_mode:
                bjd = Time(
                    rel_days,
                    format=fmt,
                    scale=str(ds.header.get("time_scale", "utc")).lower(),
                ).tdb.jd
            else:
                try:
                    bjd = (t0 + (ds.time * ds.tunit)).tdb.jd
                except Exception:
                    bjd = t0.tdb.jd + rel_days
            valid_data = np.isfinite(bjd) & np.isfinite(I_t) & np.isfinite(V_t)
            bjd = bjd[valid_data]
            I_t = I_t[valid_data]
            V_t = V_t[valid_data]
            T0 = float(t0.tdb.jd)
            freq_min_mhz = float(np.nanmin(ds.freq))
            freq_max_mhz = float(np.nanmax(ds.freq))
            dur_h = (bjd[-1] - bjd[0]) * 24
            print(f"  [OK] {basename} | color={item['color_label']} | SB{sbid} | {bjd.size} pts "
                  f"| duration={dur_h:.2f}h | freq={freq_min_mhz:.1f}-{freq_max_mhz:.1f} MHz")
            epochs.append({'sbid': sbid, 'plot_label': plot_label,
                           'color_label': item['color_label'],
                           'plot_slot': item['plot_slot'], 'bjd': bjd,
                           'I': I_t, 'V': V_t, 'T0': T0,
                           'freq_min_mhz': freq_min_mhz, 'freq_max_mhz': freq_max_mhz,
                           'path': ds_file})
        except Exception as e:
            print(f"  [WARN] 读取 {basename} 失败: {e}")

    if len(epochs) < 1:
        print("[ERROR] 没有有效数据"); sys.exit(1)

    # ── 共同绝对星历、可切换相位显示、三面板输出 ──
    # 该分支只复用上面的读取与时间分箱结果，旧的 epoch_stack 绘图流程保持在下方。
    if FIGURE_LAYOUT == "phase_aligned":
        if PHASE_FOLD_CYCLES not in (1, 2):
            raise ValueError("PHASE_FOLD_CYCLES 必须为 1（mod 1 后复制）或 2（原始 mod 2）")
        if PHASE_FOLD_CYCLES == 1 and (
                not isinstance(TESS_PHASE_BINS, int) or TESS_PHASE_BINS < 2
                or TESS_PHASE_BINS % 2 != 0):
            raise ValueError("mod 1 复制展示要求 TESS_PHASE_BINS 为不小于 2 的偶数")
        if PHASE_REFERENCE_MODE != "absolute_ephemeris":
            raise ValueError("phase_aligned 要求 PHASE_REFERENCE_MODE='absolute_ephemeris'")
        if not np.isfinite(ephemeris_t0_bjd_tdb) or P <= 0:
            raise ValueError("结果表中的 T0 必须有限，周期必须为正")
        if len({ep['sbid'] for ep in epochs}) != len(epochs):
            raise ValueError("DS_FILES 中 SBID 必须唯一")
        if INCLUDE_TESS and tess_jd is None:
            raise RuntimeError("phase_aligned 需要可读取的 TESS S69 文件")

        PHASE_ZERO_TIME = ephemeris_t0_bjd_tdb
        folded_radio = []
        radio_rows = []
        for ep in epochs:
            # 按固定秒数对单个 epoch 的 I/V 时间序列分箱；该逻辑在本脚本只有两处用途，
            # 直接展开以保留输入时间单位和 std/sqrt(n) 误差定义。
            t_jd = np.asarray(ep['bjd'], float)
            y1 = np.asarray(ep['I'], float)
            y2 = np.asarray(ep['V'], float)
            finite = np.isfinite(t_jd) & np.isfinite(y1) & np.isfinite(y2)
            t_jd, y1, y2 = t_jd[finite], y1[finite], y2[finite]
            if t_jd.size == 0:
                tb, Ib, Ibe, Vb, Vbe = (np.array([]),) * 5
            else:
                dt = BIN_SEC / 86400.0
                edges = np.arange(np.nanmin(t_jd), np.nanmax(t_jd) + dt, dt)
                bin_index = np.digitize(t_jd, edges) - 1
                tb_list, y1_list, e1_list, y2_list, e2_list = [], [], [], [], []
                for bin_number in range(len(edges) - 1):
                    in_bin = bin_index == bin_number
                    count = int(np.sum(in_bin))
                    if count <= 0:
                        continue
                    tb_list.append(np.nanmean(t_jd[in_bin]))
                    y1_list.append(np.nanmean(y1[in_bin]))
                    y2_list.append(np.nanmean(y2[in_bin]))
                    if count > 1:
                        e1_list.append(np.nanstd(y1[in_bin]) / np.sqrt(count))
                        e2_list.append(np.nanstd(y2[in_bin]) / np.sqrt(count))
                    else:
                        e1_list.append(np.nan)
                        e2_list.append(np.nan)
                tb = np.asarray(tb_list)
                Ib = np.asarray(y1_list)
                Ibe = np.asarray(e1_list)
                Vb = np.asarray(y2_list)
                Vbe = np.asarray(e2_list)
            absolute_cycle = (tb - PHASE_ZERO_TIME) / P
            phase_0_1 = np.mod(absolute_cycle, 1.0)
            phase_display = (phase_0_1 if PHASE_FOLD_CYCLES == 1
                             else np.mod(absolute_cycle, 2.0))
            cycle_index = np.floor((tb - tb[0]) / P + 1e-10).astype(int)

            valid = (np.isfinite(phase_display) & np.isfinite(phase_0_1)
                     & np.isfinite(Ib) & np.isfinite(Ibe)
                     & np.isfinite(Vb) & np.isfinite(Vbe))
            phase_0_1 = phase_0_1[valid]
            phase_display = phase_display[valid]
            cycle_index = cycle_index[valid]
            tb = tb[valid]
            Ib, Ibe, Vb, Vbe = Ib[valid], Ibe[valid], Vb[valid], Vbe[valid]

            with np.errstate(divide="ignore", invalid="ignore"):
                pol_percent = np.abs(100.0 * Vb / Ib)
                pol_err_percent = 100.0 * np.sqrt(
                    (Vbe / Ib) ** 2 + (Vb * Ibe / Ib ** 2) ** 2
                )
                i_snr = Ib / Ibe
            valid_ratio = (Ib > 0.0) & (Ibe > 0.0) & np.isfinite(pol_percent)
            polarization_included = (
                valid_ratio & (i_snr >= MIN_I_SNR)
                & np.isfinite(pol_err_percent)
                & (pol_err_percent <= MAX_POLARIZATION_ERROR_PERCENT)
            )
            exclusion_reason = np.full(phase_0_1.size, "", dtype=object)
            exclusion_reason[~valid_ratio] = "nonfinite_or_nonpositive_I"
            exclusion_reason[valid_ratio & ~(i_snr >= MIN_I_SNR)] = "I_snr_below_threshold"
            exclusion_reason[
                valid_ratio & (i_snr >= MIN_I_SNR)
                & ~(np.isfinite(pol_err_percent)
                    & (pol_err_percent <= MAX_POLARIZATION_ERROR_PERCENT))
            ] = "V_over_I_error_above_threshold"

            order = np.argsort(phase_display)
            raw_I = np.asarray(ep["I"], dtype=float)
            raw_V = np.asarray(ep["V"], dtype=float)
            fd = {
                "sbid": ep["sbid"], "phi": phase_display[order],
                "phi_0_1": phase_0_1[order],
                "cycle_index": cycle_index[order], "bjd": tb[order],
                "plot_label": ep["plot_label"], "color_label": ep["color_label"],
                "plot_slot": ep["plot_slot"],
                # 保留原始时间积分供相位图按固定 PHASE_NBINS 分箱；CSV 仍使用
                # 上面经过 BIN_SEC 时间分箱后的科学数据。
                "raw_absolute_cycle": (ep["bjd"] - PHASE_ZERO_TIME) / P,
                "raw_I": raw_I, "raw_V": raw_V,
                "I": Ib[order], "Ie": Ibe[order], "V": Vb[order],
                "Ve": Vbe[order], "pol": pol_percent[order],
                "pol_err": pol_err_percent[order],
                "pol_ok": polarization_included[order],
                "reason": exclusion_reason[order],
                "freq_min_mhz": ep["freq_min_mhz"],
                "freq_max_mhz": ep["freq_max_mhz"],
            }
            folded_radio.append(fd)
            phase_sigma = propagated_phase_uncertainty(
                ep["T0"] - PHASE_ZERO_TIME, P, period_error_days
            )
            print(f"  folded SB{ep['sbid']}: {len(fd['phi'])} bins, "
                  f"phi=[{np.nanmin(fd['phi']):.3f}, {np.nanmax(fd['phi']):.3f}], "
                  f"cycles={sorted(set(fd['cycle_index'].tolist()))}, "
                  f"phase_sigma={phase_sigma:.3f} cycles")
            for j in range(len(fd["phi"])):
                radio_rows.append({
                    "series": "radio", "sector": "", "sbid": f"SB{fd['sbid']}",
                    "bjd_tdb": fd["bjd"][j], "phase_0_1": fd["phi_0_1"][j],
                    "phase_display": fd["phi"][j],
                    "phase_fold_cycles": PHASE_FOLD_CYCLES,
                    "phase_display_mode": ("mod1_repeated" if PHASE_FOLD_CYCLES == 1
                                             else "mod2"),
                    "cycle_index": fd["cycle_index"][j],
                    "plot_label": fd["plot_label"], "color_label": fd["color_label"],
                    "plot_slot": fd["plot_slot"],
                    "freq_min_mhz": fd["freq_min_mhz"],
                    "freq_max_mhz": fd["freq_max_mhz"],
                    "flux_norm": np.nan, "flux_err": np.nan,
                    "I_mJy": fd["I"][j], "I_err_mJy": fd["Ie"][j],
                    "V_mJy": fd["V"][j], "V_err_mJy": fd["Ve"][j],
                    "v_over_i_percent": fd["pol"][j],
                    "v_over_i_err_percent": fd["pol_err"][j],
                    "polarization_included": bool(fd["pol_ok"][j]),
                    "polarization_exclusion_reason": fd["reason"][j],
                    "period_days": P, "phase_zero_bjd_tdb": PHASE_ZERO_TIME,
                    "ephemeris_label": ephemeris_label,
                })

        tess_folded = None
        optical_minimum_phase_0_1 = np.nan
        tess_phase_zero = PHASE_ZERO_TIME
        if INCLUDE_TESS and tess_jd is not None:
            tess_phase_mod = 1.0 if PHASE_FOLD_CYCLES == 1 else 2.0
            tess_bins = TESS_PHASE_BINS // 2 if PHASE_FOLD_CYCLES == 1 else TESS_PHASE_BINS
            # TESS 相位折叠与分箱仅在本脚本内使用，直接展开以确保 mod1/mod2 的
            # 分箱密度逻辑清晰可审计：mod1 使用半数基础 bin，之后复制到第二周期。
            phi_t_raw = np.mod(
                (tess_jd - PHASE_ZERO_TIME) / P, tess_phase_mod
            )
            tess_finite = np.isfinite(phi_t_raw) & np.isfinite(tess_flux)
            phi_t_raw = phi_t_raw[tess_finite]
            tess_flux_valid = np.asarray(tess_flux, dtype=float)[tess_finite]
            tess_edges = np.linspace(0.0, tess_phase_mod, tess_bins + 1)
            tess_centers = 0.5 * (tess_edges[:-1] + tess_edges[1:])
            tess_binned = np.full(tess_bins, np.nan)
            tess_errors = np.full(tess_bins, np.nan)
            for bin_number in range(tess_bins):
                in_bin = (
                    (phi_t_raw >= tess_edges[bin_number])
                    & (phi_t_raw < tess_edges[bin_number + 1])
                )
                count = int(np.sum(in_bin))
                if count > 3:
                    tess_binned[bin_number] = np.nanmedian(tess_flux_valid[in_bin])
                    tess_errors[bin_number] = (
                        np.nanstd(tess_flux_valid[in_bin]) / np.sqrt(count)
                    )
            tess_valid_bins = ~np.isnan(tess_binned)
            phi_t = tess_centers[tess_valid_bins]
            flux_b = tess_binned[tess_valid_bins]
            err_b = tess_errors[tess_valid_bins]
            tess_folded = {"phi": phi_t, "flux": flux_b, "err": err_b}
            finite_tess_bins = np.isfinite(flux_b)
            if np.any(finite_tess_bins):
                if PHASE_FOLD_CYCLES == 2:
                    first_cycle = finite_tess_bins & (phi_t < 1.0)
                    minimum_mask = first_cycle if np.any(first_cycle) else finite_tess_bins
                else:
                    minimum_mask = finite_tess_bins
                optical_minimum_phase_0_1 = float(
                    np.mod(phi_t[minimum_mask][np.nanargmin(flux_b[minimum_mask])], 1.0)
                )
            for j in range(len(phi_t)):
                tess_phase_0_1 = (phi_t[j] if PHASE_FOLD_CYCLES == 1
                                  else np.mod(phi_t[j], 1.0))
                radio_rows.append({
                    "series": "tess", "sector": f"S{int(tess_sector):02d}"
                    if str(tess_sector).isdigit() else str(tess_sector),
                    "sbid": "", "bjd_tdb": np.nan, "phase_0_1": tess_phase_0_1,
                    "phase_display": phi_t[j],
                    "phase_fold_cycles": PHASE_FOLD_CYCLES,
                    "phase_display_mode": ("mod1_repeated" if PHASE_FOLD_CYCLES == 1
                                             else "mod2"),
                    "cycle_index": np.nan, "plot_label": "TESS", "color_label": "",
                    "plot_slot": np.nan,
                    "freq_min_mhz": np.nan,
                    "freq_max_mhz": np.nan, "flux_norm": flux_b[j],
                    "flux_err": err_b[j], "I_mJy": np.nan, "I_err_mJy": np.nan,
                    "V_mJy": np.nan, "V_err_mJy": np.nan,
                    "v_over_i_percent": np.nan, "v_over_i_err_percent": np.nan,
                    "polarization_included": False,
                    "polarization_exclusion_reason": "not_applicable",
                    "period_days": P, "phase_zero_bjd_tdb": PHASE_ZERO_TIME,
                    "ephemeris_label": ephemeris_label,
                })

        for row in radio_rows:
            row["optical_minimum_phase_0_1"] = optical_minimum_phase_0_1

        # 用与中心星历配对的 bootstrap T0/P 样本传播光学极小值相位。
        # 只对实际射电历元附近的 cycle 绘制，避免把多年间未观测的周期填满。
        # 保存每个实际射电周期的边缘 95% 区间；随后只将其外包络用于绘图。
        # 因此不会把多历元的边界线叠画成多条“置信带”。
        optical_minimum_cycle_intervals = []
        optical_minimum_band_envelopes = []
        ephemeris_samples_loaded = False
        if (SHOW_OPTICAL_MINIMUM_BAND
                and np.isfinite(optical_minimum_phase_0_1)
                and EPHEMERIS_SAMPLES_FILE
                and os.path.isfile(EPHEMERIS_SAMPLES_FILE)):
            try:
                sample_table = pd.read_csv(EPHEMERIS_SAMPLES_FILE, sep="\t")
                required_sample_columns = {"alias_rank", "t0_bjd_tdb", "period_days"}
                if not required_sample_columns.issubset(sample_table.columns):
                    raise ValueError("bootstrap 文件缺少 alias_rank/t0_bjd_tdb/period_days 列")
                sample_table = sample_table.loc[
                    pd.to_numeric(sample_table["alias_rank"], errors="coerce")
                    == EPHEMERIS_ALIAS_RANK
                ]
                if sample_table.empty:
                    raise ValueError(f"bootstrap 文件没有 alias_rank={EPHEMERIS_ALIAS_RANK} 样本")
                sample_t0 = sample_table["t0_bjd_tdb"].to_numpy(dtype=float)
                sample_period = sample_table["period_days"].to_numpy(dtype=float)
                sample_valid = (np.isfinite(sample_t0)
                                & np.isfinite(sample_period)
                                & (sample_period > 0.0))
                sample_t0 = sample_t0[sample_valid]
                sample_period = sample_period[sample_valid]
                if sample_t0.size < 100:
                    raise ValueError(f"有效 paired 样本只有 {sample_t0.size} 个")
                sample_period_low, sample_period_high = np.quantile(
                    sample_period, (0.025, 0.975)
                )
                sample_t0_low, sample_t0_high = np.quantile(
                    sample_t0, (0.025, 0.975)
                )
                if not sample_period_low <= P <= sample_period_high:
                    raise ValueError("结果表中心周期不在 paired Bootstrap 的 95% 范围内")
                if not sample_t0_low <= ephemeris_t0_bjd_tdb <= sample_t0_high:
                    raise ValueError("结果表中心 T0 不在 paired Bootstrap 的 95% 范围内")
                lower_q = 0.5 * (1.0 - BAND_CONFIDENCE_LEVEL)
                upper_q = 1.0 - lower_q
                display_period = 1.0 if PHASE_FOLD_CYCLES == 1 else 2.0
                cycle_numbers = set()
                for ep in epochs:
                    epoch_start_cycle = int(np.floor(
                        (np.nanmin(ep["bjd"]) - ephemeris_t0_bjd_tdb) / P
                        - optical_minimum_phase_0_1
                    )) - 1
                    epoch_end_cycle = int(np.ceil(
                        (np.nanmax(ep["bjd"]) - ephemeris_t0_bjd_tdb) / P
                        - optical_minimum_phase_0_1
                    )) + 1
                    cycle_numbers.update(range(epoch_start_cycle, epoch_end_cycle + 1))

                for cycle_number in sorted(cycle_numbers):
                    nominal_event_bjd_tdb = ephemeris_t0_bjd_tdb + (
                        cycle_number + optical_minimum_phase_0_1
                    ) * P
                    sampled_event_bjd_tdb = sample_t0 + (
                        cycle_number + optical_minimum_phase_0_1
                    ) * sample_period
                    phase_delta = (
                        sampled_event_bjd_tdb - nominal_event_bjd_tdb
                    ) / P
                    delta_low, delta_high = np.quantile(
                        phase_delta, (lower_q, upper_q)
                    )
                    display_center = np.mod(
                        (nominal_event_bjd_tdb - PHASE_ZERO_TIME) / P,
                        display_period,
                    )
                    interval_width = float(delta_high - delta_low)
                    if interval_width >= display_period:
                        intervals = [(0.0, display_period)]
                    else:
                        wrapped_low = np.mod(
                            display_center + delta_low, display_period
                        )
                        wrapped_high = wrapped_low + interval_width
                        if wrapped_high <= display_period:
                            intervals = [(wrapped_low, wrapped_high)]
                        else:
                            intervals = [
                                (wrapped_low, display_period),
                                (0.0, wrapped_high - display_period),
                            ]
                    for band_low, band_high in intervals:
                        if band_high > band_low:
                            display_shifts = (0.0, 1.0) if PHASE_FOLD_CYCLES == 1 else (0.0,)
                            for display_shift in display_shifts:
                                optical_minimum_cycle_intervals.append({
                                    "low": float(band_low + display_shift),
                                    "high": float(band_high + display_shift),
                                    "cycle_number": cycle_number,
                                    "confidence_level": BAND_CONFIDENCE_LEVEL,
                                })
                # 仅合并显示上相交的区间；mod 1 的两个重复周期仍各保留一条带。
                # 此带是逐周期边缘 95% 区间的外包络，不能被解读为联合 95% 区间。
                for interval in sorted(
                        optical_minimum_cycle_intervals,
                        key=lambda item: (item["low"], item["high"])):
                    if (not optical_minimum_band_envelopes
                            or interval["low"]
                            > optical_minimum_band_envelopes[-1]["high"] + 1e-12):
                        optical_minimum_band_envelopes.append({
                            "low": interval["low"],
                            "high": interval["high"],
                        })
                    else:
                        optical_minimum_band_envelopes[-1]["high"] = max(
                            optical_minimum_band_envelopes[-1]["high"],
                            interval["high"],
                        )
                ephemeris_samples_loaded = bool(optical_minimum_band_envelopes)
                if ephemeris_samples_loaded:
                    print(
                        "[INFO] 光学极小值相位带："
                        f"{len(optical_minimum_cycle_intervals)} 个逐周期 {BAND_CONFIDENCE_LEVEL * 100:.0f}% 区间"
                        f"合并为 {len(optical_minimum_band_envelopes)} 条显示外包络。"
                    )
                    for envelope_index, envelope in enumerate(
                            optical_minimum_band_envelopes, start=1):
                        envelope_width = envelope["high"] - envelope["low"]
                        print(
                            f"  95% envelope {envelope_index}: "
                            f"{envelope['low']:.6f}--{envelope['high']:.6f} cycle; "
                            f"width={envelope_width:.6f} cycle"
                        )
            except (OSError, ValueError, TypeError) as error:
                print(f"[WARN] 光学极小值置信带未绘制：{error}")
        elif SHOW_OPTICAL_MINIMUM_BAND and EPHEMERIS_SAMPLES_FILE:
            print(f"[WARN] paired 星历样本不存在，保留光学极小值中心线：{EPHEMERIS_SAMPLES_FILE}")

        # 三面板：TESS；I/V；|V|/I。两种模式均保持 [0, 2) 显示范围，
        # mod 1 只在绘图时复制数据，CSV 仍保留每个真实点一次。
        fig, axes = plt.subplots(
            3, 1, figsize=(FIG_W, PANEL_H * 3), sharex=True,
            gridspec_kw={"hspace": 0},
        )
        tess_label = (f"TESS S{int(tess_sector):02d}"
                      if str(tess_sector).isdigit() else f"TESS {tess_sector}")
        if tess_folded is not None:
            tess_raw_phase_mod = 1.0 if PHASE_FOLD_CYCLES == 1 else 2.0
            tess_raw_phi = np.mod(
                (tess_jd - PHASE_ZERO_TIME) / P, tess_raw_phase_mod
            )
            tess_raw_plot_phi = tess_raw_phi
            tess_raw_plot_flux = tess_flux
            if PHASE_FOLD_CYCLES == 1:
                tess_raw_plot_phi = np.concatenate((tess_raw_phi, tess_raw_phi + 1.0))
                tess_raw_plot_flux = np.concatenate((tess_flux, tess_flux))
            axes[0].scatter(
                tess_raw_plot_phi, tess_raw_plot_flux, s=4, alpha=0.13,
                color="0.45", rasterized=True, zorder=1,
            )
            tess_plot_phi = tess_folded["phi"]
            tess_plot_flux = tess_folded["flux"]
            if PHASE_FOLD_CYCLES == 1:
                tess_plot_phi = np.concatenate((tess_plot_phi, tess_plot_phi + 1.0))
                tess_plot_flux = np.concatenate((tess_plot_flux, tess_plot_flux))
            template_order = np.argsort(tess_plot_phi)
            axes[0].plot(
                tess_plot_phi[template_order], tess_plot_flux[template_order],
                "o-", markersize=4.2, linewidth=1.2, color="#B2182B",
                label="TESS binned median", zorder=3,
            )
            axes[0].set_ylabel("Normalized TESS flux", fontsize=12)
            axes[0].legend(loc="upper right", fontsize=9, framealpha=0.85)
        else:
            axes[0].text(0.5, 0.5, "TESS unavailable", transform=axes[0].transAxes,
                         ha="center", va="center")
            axes[0].set_ylabel("Normalized TESS flux", fontsize=12)
        if SHOW_OPTICAL_FEATURE_BAND:
            # 低透明度白色填充配合银灰边界，提示相位范围但不遮盖原始观测点。
            for lo, hi in OPTICAL_FEATURE_PHASE_INTERVALS:
                if lo <= hi:
                    axes[0].axvspan(
                        lo, hi, facecolor=(1.0, 1.0, 1.0, 1),
                        edgecolor="#A6A6A6", linewidth=0.7, zorder=2.5,
                    )
                else:
                    axes[0].axvspan(
                        lo, 1.0, facecolor=(1.0, 1.0, 1.0, 1),
                        edgecolor="#A6A6A6", linewidth=0.7, zorder=2.5,
                    )
                    axes[0].axvspan(
                        0.0, hi, facecolor=(1.0, 1.0, 1.0, 1),
                        edgecolor="#A6A6A6", linewidth=0.7, zorder=2.5,
                    )
        phase_display_label = "mod 1" if PHASE_FOLD_CYCLES == 1 else "mod 2"
        add_panel_label(axes[0], tess_label, fontsize=11)

        # 论文色板：克莱因蓝 + 深学术红 + 学术橙；颜色只表示绘图槽位。
        # 点型只区分 Stokes I/V，不额外编码历元内部的连续周次。
        # 与 Radio_LS_PhaseFolding.py 一致：PHASE_NBINS（这里沿用
        # TESS_PHASE_BINS）表示整个 [0, 2) 展示范围的总分箱数。
        # mod 1 先复制相位数据再用同一个总 bin 数分箱，因此两种模式的
        # 每个展示周期点密度一致；CSV 仍保留时间分箱后的原始射电点。
        for fd in folded_radio:
            base_color = ("#002FA7", "#B2182B", "#E66101")[fd["plot_slot"]]
            if PHASE_FOLD_CYCLES == 1:
                bins_per_cycle = TESS_PHASE_BINS // 2
                cycle_values = np.unique(fd["cycle_index"])
                for cycle_position, cycle_index_value in enumerate(cycle_values):
                    cycle_mask = fd["cycle_index"] == cycle_index_value
                    i_plot_phi, i_plot_values, i_plot_errors = phase_bin_for_display(
                        fd["phi_0_1"][cycle_mask], fd["I"][cycle_mask],
                        bins_per_cycle, errors=fd["Ie"][cycle_mask],
                        min_count=1, phase_max=1.0,
                    )
                    v_plot_phi, v_plot_values, v_plot_errors = phase_bin_for_display(
                        fd["phi_0_1"][cycle_mask], fd["V"][cycle_mask],
                        bins_per_cycle, errors=fd["Ve"][cycle_mask],
                        min_count=1, phase_max=1.0,
                    )
                    i_plot_phi = np.concatenate((i_plot_phi, i_plot_phi + 1.0))
                    i_plot_values = np.concatenate((i_plot_values, i_plot_values))
                    i_plot_errors = np.concatenate((i_plot_errors, i_plot_errors))
                    v_plot_phi = np.concatenate((v_plot_phi, v_plot_phi + 1.0))
                    v_plot_values = np.concatenate((v_plot_values, v_plot_values))
                    v_plot_errors = np.concatenate((v_plot_errors, v_plot_errors))
                    frequency_label = (f"({fd['freq_min_mhz']:.0f}–"
                                       f"{fd['freq_max_mhz']:.0f} MHz)")
                    label_suffix = f" {frequency_label}" if cycle_position == 0 else ""
                    axes[1].errorbar(
                        i_plot_phi, i_plot_values, yerr=i_plot_errors,
                        fmt="o", linestyle="none",
                        ms=3.4, capsize=1.3, elinewidth=0.65, alpha=0.70,
                        color=base_color, markerfacecolor=base_color,
                        markeredgecolor=base_color,
                        label=(f"I {fd['plot_label']}{label_suffix}"
                               if cycle_position == 0 else "_nolegend_"),
                    )
                    axes[1].errorbar(
                        v_plot_phi, v_plot_values, yerr=v_plot_errors,
                        fmt="^", linestyle="none",
                        ms=3.4, capsize=1.3, elinewidth=0.65, alpha=0.70,
                        color=base_color, markerfacecolor="none",
                        markeredgecolor=base_color,
                        label=(f"V {fd['plot_label']}{label_suffix}"
                               if cycle_position == 0 else "_nolegend_"),
                    )
            else:
                raw_phase_mod = 2.0
                radio_plot_phi = np.mod(fd["raw_absolute_cycle"], raw_phase_mod)
                i_plot_phi, i_plot_values, i_plot_errors = phase_bin_for_display(
                    radio_plot_phi, fd["raw_I"], TESS_PHASE_BINS
                )
                v_plot_phi, v_plot_values, v_plot_errors = phase_bin_for_display(
                    radio_plot_phi, fd["raw_V"], TESS_PHASE_BINS
                )
                frequency_label = (f"({fd['freq_min_mhz']:.0f}–"
                                   f"{fd['freq_max_mhz']:.0f} MHz)")
                axes[1].errorbar(
                    i_plot_phi, i_plot_values, yerr=i_plot_errors,
                    fmt="o", linestyle="none",
                    ms=3.4, capsize=1.3, elinewidth=0.65, alpha=0.70,
                    color=base_color, markerfacecolor=base_color,
                    markeredgecolor=base_color,
                    label=f"I {fd['plot_label']} {frequency_label}",
                )
                axes[1].errorbar(
                    v_plot_phi, v_plot_values, yerr=v_plot_errors,
                    fmt="^", linestyle="none",
                    ms=3.4, capsize=1.3, elinewidth=0.65, alpha=0.70,
                    color=base_color, markerfacecolor="none",
                    markeredgecolor=base_color,
                    label=f"V {fd['plot_label']} {frequency_label}",
                )
        axes[1].set_ylabel("Flux density (mJy)", fontsize=12)
        add_panel_label(axes[1], "ASKAP Stokes I/V", fontsize=11)
        axes[1].legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.85)

        for fd in folded_radio:
            # |V|/I 保留原有质量筛选；单点 bin 使用已传播的 pol_err，避免
            # mod 2 因质量筛选后的时间分箱稀疏而整幅图没有有效 bin。
            pol_mask = fd["pol_ok"]
            if not np.any(pol_mask):
                continue
            if PHASE_FOLD_CYCLES == 1:
                bins_per_cycle = TESS_PHASE_BINS // 2
                cycle_values = np.unique(fd["cycle_index"][pol_mask])
                for cycle_position, cycle_index_value in enumerate(cycle_values):
                    cycle_mask = pol_mask & (fd["cycle_index"] == cycle_index_value)
                    pol_plot_phi, pol_plot_values, pol_plot_errors = phase_bin_for_display(
                        fd["phi_0_1"][cycle_mask], fd["pol"][cycle_mask],
                        bins_per_cycle, errors=fd["pol_err"][cycle_mask],
                        min_count=1, phase_max=1.0,
                    )
                    pol_plot_phi = np.concatenate((pol_plot_phi, pol_plot_phi + 1.0))
                    pol_plot_values = np.concatenate((pol_plot_values, pol_plot_values))
                    pol_plot_errors = np.concatenate((pol_plot_errors, pol_plot_errors))
                    axes[2].errorbar(
                        pol_plot_phi, pol_plot_values, yerr=pol_plot_errors,
                        fmt="o", ms=3.2, capsize=1.3, elinewidth=0.65, alpha=0.70,
                        color=("#002FA7", "#B2182B", "#E66101")[fd["plot_slot"]],
                        label=(fd["plot_label"] if cycle_position == 0 else "_nolegend_"),
                    )
            else:
                pol_raw_phase = fd["phi"][pol_mask]
                pol_raw_values = fd["pol"][pol_mask]
                pol_raw_errors = fd["pol_err"][pol_mask]
                pol_plot_phi, pol_plot_values, pol_plot_errors = phase_bin_for_display(
                    pol_raw_phase, pol_raw_values, TESS_PHASE_BINS,
                    errors=pol_raw_errors, min_count=1,
                )
                axes[2].errorbar(
                    pol_plot_phi, pol_plot_values, yerr=pol_plot_errors,
                    fmt="o", ms=3.2, capsize=1.3, elinewidth=0.65, alpha=0.70,
                    color=("#002FA7", "#B2182B", "#E66101")[fd["plot_slot"]],
                    label=fd["plot_label"],
                )
        axes[2].axhline(0.0, ls="--", lw=0.9, color="gray", alpha=0.8)
        axes[2].set_ylabel("|V|/I (%)", fontsize=12)
        axes[2].set_xlabel(
            "Folded phase (mod 1)" if PHASE_FOLD_CYCLES == 1
            else "Cycle phase (mod 2)", fontsize=13
        )
        add_panel_label(axes[2], "100 |V|/I", fontsize=11)
        axes[2].legend(loc="upper right", fontsize=9, framealpha=0.85)
        axes[0].tick_params(axis="x", which="both", labelbottom=False)
        axes[1].tick_params(axis="x", which="both", labelbottom=False)
        for ax in axes:
            ax.set_xlim(0.0, 2.0)
            ax.margins(x=0)
            ax.grid(color="0.86", linewidth=0.6, alpha=0.8)
            ax.tick_params(axis="both", labelsize=10)

        if ephemeris_samples_loaded:
            band_label_added = False
            for band in optical_minimum_band_envelopes:
                band_label = (
                    f"Envelope of TESS minimum {BAND_CONFIDENCE_LEVEL * 100:.0f}% intervals"
                    if not band_label_added else "_nolegend_"
                )
                # 低透明度鹅黄色外包络保留数据可见性，同时明确显示各实际周期区间的总范围。
                axes[0].axvspan(
                    band["low"], band["high"],
                    facecolor="#F3D36A",
                    alpha=0.24,
                    edgecolor="#A6A6A6", linewidth=0.7,
                    label=band_label, zorder=2.5,
                )
                for ax in axes[1:]:
                    ax.axvspan(
                        band["low"], band["high"],
                        facecolor="#F3D36A",
                        alpha=0.24,
                        edgecolor="#A6A6A6", linewidth=0.7,
                        zorder=2.5,
                    )
                band_label_added = True
            axes[0].legend(loc="upper right", fontsize=9, framealpha=0.85)

        if SHOW_OPTICAL_MINIMUM and np.isfinite(optical_minimum_phase_0_1):
            optical_marker_phases = [optical_minimum_phase_0_1,
                                     optical_minimum_phase_0_1 + 1.0]
            for phase_marker in optical_marker_phases:
                for ax in axes:
                    ax.axvline(
                        phase_marker, color="#A6A6A6",
                        linestyle="--", linewidth=1.0, alpha=0.8, zorder=4,
                    )

        hostname_match = re.search(r"(.+?)_SB\d+", os.path.basename(epochs[0]["path"]))
        hostname = hostname_match.group(1) if hostname_match else "Unknown"
        # 输出文件名中的源名和 SBID 均来自成功读取的实际文件名。
        sbid_tag = "_".join(f"SB{ep['sbid']}" for ep in epochs)
        fig.suptitle(
            f"{hostname} | {sbid_tag} | P={P:.6f} d | {phase_display_label}",
            fontsize=15, fontweight="bold", y=0.995,
        )
        fig.subplots_adjust(top=0.965, bottom=0.04, hspace=0)

        output_dir = os.path.join(OUTPUT_BASE, hostname)
        os.makedirs(output_dir, exist_ok=True)
        fold_tag = "mod1" if PHASE_FOLD_CYCLES == 1 else "mod2"
        output_stem = os.path.join(output_dir, f"{hostname}_{sbid_tag}_PhaseAligned_{fold_tag}")
        out_png = f"{output_stem}.png"
        fig.savefig(out_png, dpi=DPI, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        csv_out = f"{output_stem}.csv"
        pd.DataFrame(radio_rows).to_csv(csv_out, index=False, encoding="utf-8")
        print(f"\nSaved phase-aligned outputs:\n  {out_png}\n  {csv_out}")
        return {"radio_rows": radio_rows, "tess_label": tess_label,
                "optical_minimum_phase_0_1": optical_minimum_phase_0_1,
                "output_png": out_png, "output_csv": csv_out}

    PHASE_ZERO_TIME = epochs[0]['T0']
    print(f"\nPhase zero (first ASKAP epoch start): BJD_TDB = {PHASE_ZERO_TIME:.9f}")
    if PHASE_REFERENCE_MODE == "absolute_ephemeris":
        print("[WARN] 使用绝对星历相位：周期误差会随历元间隔线性累积；请检查下面的相位误差提示。")
    else:
        print("[WARN] per_epoch 模式：各历元/TESS 使用独立相位零点，只能比较折叠形态"
              "（峰位/宽度/振幅），不能用于比较不同历元的绝对相位、"
              "TESS-ASKAP 同相或射电峰长期相位锁定。断言同相需切换到 "
              "absolute_ephemeris 模式并配合可靠星历。")
    print("[NOTE] 本图误差条为 std/sqrt(n)（假设独立），相邻射电积分相关，可能偏小；"
          "仅作形态参考，不作为周期显著性证据（显著性以 LS + bootstrap FAP + LOEO 为准）。")

    folded_radio = []
    for i, ep in enumerate(epochs):
        # epoch_stack 与 phase_aligned 使用同一套时间分箱规则；这里直接展开，
        # 保持 BIN_SEC 与 std/sqrt(n) 误差定义完全一致。
        t_jd = np.asarray(ep['bjd'], float)
        y1 = np.asarray(ep['I'], float)
        y2 = np.asarray(ep['V'], float)
        finite = np.isfinite(t_jd) & np.isfinite(y1) & np.isfinite(y2)
        t_jd, y1, y2 = t_jd[finite], y1[finite], y2[finite]
        if t_jd.size == 0:
            tb, Ib, Ibe, Vb, Vbe = (np.array([]),) * 5
        else:
            dt = BIN_SEC / 86400.0
            edges = np.arange(np.nanmin(t_jd), np.nanmax(t_jd) + dt, dt)
            bin_index = np.digitize(t_jd, edges) - 1
            tb_list, y1_list, e1_list, y2_list, e2_list = [], [], [], [], []
            for bin_number in range(len(edges) - 1):
                in_bin = bin_index == bin_number
                count = int(np.sum(in_bin))
                if count <= 0:
                    continue
                tb_list.append(np.nanmean(t_jd[in_bin]))
                y1_list.append(np.nanmean(y1[in_bin]))
                y2_list.append(np.nanmean(y2[in_bin]))
                if count > 1:
                    e1_list.append(np.nanstd(y1[in_bin]) / np.sqrt(count))
                    e2_list.append(np.nanstd(y2[in_bin]) / np.sqrt(count))
                else:
                    e1_list.append(np.nan)
                    e2_list.append(np.nan)
            tb = np.asarray(tb_list)
            Ib = np.asarray(y1_list)
            Ibe = np.asarray(e1_list)
            Vb = np.asarray(y2_list)
            Vbe = np.asarray(e2_list)
        if PHASE_REFERENCE_MODE == "per_epoch":
            phase_zero = ep['T0']
        else:
            phase_zero = PHASE_ZERO_TIME
            phase_sigma = propagated_phase_uncertainty(
                ep['T0'] - PHASE_ZERO_TIME, P, period_error_days
            )
            if phase_sigma > ABSOLUTE_PHASE_WARN_CYCLES:
                print(f"  [WARN] SB{ep['sbid']} 相对首历元的周期外推误差约 "
                      f"{phase_sigma:.2f} 个周期，绝对相位不可可靠对齐。")
        # epoch_stack 保留真实的 mod 2 相位（不复制点），用于查看相邻实际周期。
        phi_abs = (tb - phase_zero) / P
        phi = np.mod(phi_abs, 2.0)
        valid_folded = (
            (phi >= XLO) & (phi < XHI) & np.isfinite(Ib) & np.isfinite(Vb)
        )
        order = np.argsort(phi[valid_folded])
        phi = phi[valid_folded][order]
        I_p = Ib[valid_folded][order]
        I_e = Ibe[valid_folded][order]
        V_p = Vb[valid_folded][order]
        V_e = Vbe[valid_folded][order]
        phase_title = (f"{ep['plot_label']} (epoch-relative phase)"
                       if PHASE_REFERENCE_MODE == "per_epoch" else ep["plot_label"])
        folded_radio.append({'phi': phi, 'I': I_p, 'Ie': I_e, 'V': V_p, 'Ve': V_e,
                             'title': phase_title, 'phase_zero': phase_zero, 'break': True})
        if phi.size:
            print(f"  folded SB{ep['sbid']}: {phi.size} pts, phi=[{np.nanmin(phi):.3f}, {np.nanmax(phi):.3f}]")

    # ── 折叠 TESS ──
    tess_phase_zero = (t0_tess if PHASE_REFERENCE_MODE == "per_epoch" else PHASE_ZERO_TIME)
    if INCLUDE_TESS and tess_folded is None and tess_jd is not None:
        try:
            if PHASE_REFERENCE_MODE == "absolute_ephemeris" and t0_tess is not None:
                tess_phase_sigma = propagated_phase_uncertainty(
                    t0_tess - PHASE_ZERO_TIME, P, period_error_days
                )
                if tess_phase_sigma > ABSOLUTE_PHASE_WARN_CYCLES:
                    print(f"  [WARN] TESS 首时刻相对首个 ASKAP 历元的周期外推误差约 "
                          f"{tess_phase_sigma:.2f} 个周期，TESS–ASKAP 绝对相位不可可靠对齐。")
            # epoch_stack 的 TESS 相位采用原始 mod 2 折叠和 60 个相位 bin，
            # 与旧版 fold_tess_to_radio 的默认参数保持一致。
            tess_phase_mod = 2.0
            tess_bins = 60
            phi_t_raw = np.mod(
                (tess_jd - tess_phase_zero) / P, tess_phase_mod
            )
            tess_finite = np.isfinite(phi_t_raw) & np.isfinite(tess_flux)
            phi_t_raw = phi_t_raw[tess_finite]
            tess_flux_valid = np.asarray(tess_flux, dtype=float)[tess_finite]
            tess_edges = np.linspace(0.0, tess_phase_mod, tess_bins + 1)
            tess_centers = 0.5 * (tess_edges[:-1] + tess_edges[1:])
            tess_binned = np.full(tess_bins, np.nan)
            tess_errors = np.full(tess_bins, np.nan)
            for bin_number in range(tess_bins):
                in_bin = (
                    (phi_t_raw >= tess_edges[bin_number])
                    & (phi_t_raw < tess_edges[bin_number + 1])
                )
                count = int(np.sum(in_bin))
                if count > 3:
                    tess_binned[bin_number] = np.nanmedian(tess_flux_valid[in_bin])
                    tess_errors[bin_number] = (
                        np.nanstd(tess_flux_valid[in_bin]) / np.sqrt(count)
                    )
            tess_valid_bins = ~np.isnan(tess_binned)
            phi_t = tess_centers[tess_valid_bins]
            flux_b = tess_binned[tess_valid_bins]
            err_b = tess_errors[tess_valid_bins]
            tess_title = ("TESS (epoch-relative phase)"
                          if PHASE_REFERENCE_MODE == "per_epoch" else "TESS (absolute ephemeris)")
            tess_folded = {'phi': phi_t, 'flux': flux_b, 'err': err_b, 'title': tess_title}
            print(f"  folded TESS: {len(phi_t)} pts")
        except Exception as e:
            print(f"  [WARN] TESS 折叠失败: {e}")
            tess_folded = None

    # ── 绘图：TESS 最上，射电依次 ──
    n_radio = len(folded_radio)
    n_tess = 1 if tess_folded is not None else 0
    n_total = n_tess + n_radio

    fig_h = PANEL_H * n_total
    fig, axes = plt.subplots(
        n_total, 1, figsize=(FIG_W, fig_h), sharex=True,
        gridspec_kw={"hspace": 0},
    )
    if n_total == 1:
        axes = [axes]

    idx = 0
    if tess_folded is not None:
        # TESS 面板：原始折叠点与分箱中位数/误差条。
        axes[idx].scatter(
            tess_folded['phi'], tess_folded['flux'], s=3, c="#4a8fd4",
            alpha=0.35, rasterized=True, linewidths=0,
        )
        axes[idx].errorbar(
            tess_folded['phi'], tess_folded['flux'], yerr=tess_folded['err'],
            fmt='o', ms=3, color="#e74c3c", capsize=2, lw=1.0,
            label="TESS folded",
        )
        axes[idx].axvline(1.0, ls="--", lw=1.0, color="gray", alpha=0.6)
        axes[idx].set_xlim(XLO, XHI)
        axes[idx].margins(x=0)
        axes[idx].set_ylabel("Relative Flux", fontsize=12)
        add_panel_label(axes[idx], tess_folded['title'], fontsize=11)
        axes[idx].tick_params(axis='both', labelsize=11)
        axes[idx].legend(loc="upper right", fontsize=10, framealpha=0.8,
                         edgecolor="#dddddd")
        axes[idx].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        idx += 1

    for i, fd in enumerate(folded_radio):
        # 射电面板：点和误差条保留原始样式；按相位断点分段连线，避免跨越观测空档。
        axes[idx].errorbar(
            fd['phi'], fd['I'], yerr=fd['Ie'], fmt='.', ms=ERR_MS,
            elinewidth=ERR_ELW, alpha=0.6, color="black", label="_nolegend_",
        )
        axes[idx].errorbar(
            fd['phi'], fd['V'], yerr=fd['Ve'], fmt='.', ms=ERR_MS,
            elinewidth=ERR_ELW, alpha=0.6, color="#e74c3c", label="_nolegend_",
        )
        line_x = np.asarray(fd['phi'], float)
        line_i = np.asarray(fd['I'], float)
        line_v = np.asarray(fd['V'], float)
        finite = np.isfinite(line_x) & np.isfinite(line_i) & np.isfinite(line_v)
        line_x, line_i, line_v = line_x[finite], line_i[finite], line_v[finite]
        if line_x.size:
            order = np.argsort(line_x)
            line_x, line_i, line_v = line_x[order], line_i[order], line_v[order]
            breaks = np.where(np.diff(line_x) > 0.2)[0]
            segment_start = 0
            for segment_number, break_index in enumerate(
                    np.append(breaks, line_x.size - 1)):
                segment_end = break_index + 1
                if segment_end - segment_start >= 2:
                    axes[idx].plot(
                        line_x[segment_start:segment_end],
                        line_i[segment_start:segment_end],
                        lw=LINE_LW, color="black",
                        label="Stokes I" if segment_number == 0 else "_nolegend_",
                    )
                    axes[idx].plot(
                        line_x[segment_start:segment_end],
                        line_v[segment_start:segment_end],
                        lw=LINE_LW, color="#e74c3c", zorder=10,
                        label="Stokes V" if segment_number == 0 else "_nolegend_",
                    )
                segment_start = segment_end
        axes[idx].axvline(1.0, ls="--", lw=1.0, color="gray", alpha=0.6)
        axes[idx].set_xlim(XLO, XHI)
        axes[idx].margins(x=0)
        add_panel_label(axes[idx], fd['title'], fontsize=11)
        axes[idx].tick_params(axis='both', labelsize=11)
        if i == 0:
            axes[idx].legend(loc="upper right", fontsize=10, framealpha=0.8,
                             edgecolor="#dddddd")
        if idx < n_total - 1:
            axes[idx].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        idx += 1

    axes[-1].set_xlabel("Cycle phase (mod 2)", fontsize=13)
    for ax in axes:
        # Skip ylabel for radio panels (they have I/V fluxes in mJy)
        pass

    # Set radio ylabels
    if tess_folded is not None:
        # TESS panel already has its own y-label set above.
        pass
    for ax_idx in range(n_tess, n_total):
        axes[ax_idx].set_ylabel("Flux (mJy)", fontsize=12)

    hostname_match = re.search(r'(.+?)_SB\d+', os.path.basename(epochs[0]["path"]))
    hostname = hostname_match.group(1) if hostname_match else "Unknown"
    sbid_tag = "_".join(f"SB{ep['sbid']}" for ep in epochs)
    fig.suptitle(
        f"{hostname} | {sbid_tag} | P={P:.6f} d | mod 2",
        fontsize=14, y=0.99,
    )

    fig.subplots_adjust(top=0.97, bottom=0.04, hspace=0)

    output_dir = os.path.join(OUTPUT_BASE, hostname)
    os.makedirs(output_dir, exist_ok=True)
    tag = "TESS_" if tess_folded else ""
    out_png = os.path.join(output_dir, f"{tag}{hostname}_MultiEpoch_P{P}.png")
    fig.savefig(out_png, dpi=DPI, facecolor="white", bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved phase-folded figure:\n  {out_png}")


if __name__ == "__main__":
    main()
