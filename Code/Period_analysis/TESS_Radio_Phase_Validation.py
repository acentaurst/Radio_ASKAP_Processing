#!/usr/bin/env python3
"""绘制 TESS 光学模板与 ASKAP 动态谱的共同星历相位对照图。"""

from __future__ import annotations

import csv
from pathlib import Path
import re
import sys
import warnings

import h5py  # 直接读取 DStools 生成的 HDF5 动态谱，不要求运行时导入 dstools。
import lightkurve as lk  # 读取和预处理 TESS delivered light curve。
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from science_utils import (  # noqa: E402
    add_panel_label,
    as_text,
    rebin2d,
    rebin,
    slice_open_end,
)


# ============================ 参数配置区 ============================
# 换目标或观测时原则上只修改这里；周期不会在本脚本中重新拟合。

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 上游周期结果沿用本项目 HST Result 目录中的已验证 S29+S69 星历。
PERIOD_DIR = Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/TESS_Period/s29+69")
PERIOD_RESULT_FILE = PERIOD_DIR / "TESS_Period_Result.md"
PERIOD_BOOTSTRAP_FILE = PERIOD_DIR / "TESS_Period_Bootstrap.txt"
PERIOD_TIMINGS_FILE = PERIOD_DIR / "TESS_Period_Timings.txt"
SELECTED_ALIAS_RANK = 1

# 本项目现有的 TESS S69 与 ASKAP 动态谱完整路径。
TESS_TEMPLATE_FILE = Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/TESS_Data/2MASS_J01033563-5515561_A/"
                          "hlsp_qlp_tess_ffi_s0069-0000000616014335_tess_v01_llc/"
                          "hlsp_qlp_tess_ffi_s0069-0000000616014335_tess_v01_llc.fits")
RADIO_FILE = Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/Ds/2MASS_J01033563-5515561_A/Flare/"
                  "2MASS_J01033563-5515561_A_SB59565_beam22.ds")

# 保留项目的 Result/根目录、按源分类目录和输出命名方式。
OUTPUT_BASE = Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/TESS_Radio_Phase_Validation")
OUTPUT_DIR = OUTPUT_BASE / RADIO_FILE.name.split("_SB", 1)[0]

# TESS flux 选择规则；None 表示按以下优先级自动选择。
FLUX_COLUMN: str | None = None
FLUX_PRIORITY = (
    "pdcsap_flux",
    "det_flux",
    "kspsap_flux",
    "sap_flux",
    "flux",
)

# 必须与产生周期结果时的 TESS 预处理保持一致。
DETREND_WINDOW_DAYS = 1.5
DETREND_POLYORDER = 2
OUTLIER_SIGMA_UPPER = 5.0
OUTLIER_SIGMA_LOWER = 7.0

# 用于复现 DStools DynamicSpectrum 的重采样设置。
DSTOOLS_TAVG = 12
DSTOOLS_FAVG = 5
DSTOOLS_INSERT_SCAN_GAPS = True
DSTOOLS_CORR_DUMPTIME_SECONDS = 10.1

# 当前 .ds 的原始 time 是自 MJD=0 起算的 UTC 秒，frequency 是 Hz，flux 是 Jy。
DS_TIME_MODE = "utc_seconds_since_mjd0"
DS_FREQUENCY_UNIT = "Hz"
DS_FLUX_UNIT = "Jy"
# 下面三个量是 ASKAP 阵列在地球上的固定台址坐标，所有 ASKAP 源共用，
# 不是目标源的天球坐标。只有更换观测台址时才需要修改它们。
ASKAP_LONGITUDE_DEG = 116.631425
ASKAP_LATITUDE_DEG = -26.697000
ASKAP_HEIGHT_M = 361.0

# 目标源的天球坐标（每个源可以不同）。保持为 None 时，优先从 .ds 的
# phasecentre 属性读取；只有元数据缺失或 phasecentre 不是目标实际坐标时才手动填写。
TARGET_RA_DEG: float | None = None
TARGET_DEC_DEG: float | None = None

# 相位模板、95% 相位置信带和动态谱色标的显示设置。
TESS_PHASE_BINS = 60
BAND_CONFIDENCE_LEVEL = 0.95
RADIO_FLUX_SCALE = 1.0
RADIO_FLUX_UNIT_LABEL = "mJy"
# None：分别根据 I/V 的有限像素自动设置；填写数值：固定显示为 ±该值 mJy。
DYNAMIC_I_COLORBAR_LIMIT_MJY =5 #: float | None = None
DYNAMIC_V_COLORBAR_LIMIT_MJY =5 #：float | None = None
# 自动色标使用有限像素绝对值的这个分位数，避免极少数异常像素拉伸色标。
COLORBAR_AUTO_QUANTILE = 0.99
FIGURE_DPI = 300

# ==================================================================


def main() -> None:
    """读取上游星历和一份 ASKAP .ds，生成共同相位图及数值表。"""

    iers.conf.auto_download = False
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 检查周期结果、bootstrap 样本和计时点接口。
    for required_file in (
        PERIOD_RESULT_FILE,
        PERIOD_BOOTSTRAP_FILE,
        PERIOD_TIMINGS_FILE,
    ):
        if not required_file.is_file():
            raise FileNotFoundError(f"缺少 TESS_Period 输出：{required_file.resolve()}")

    tess_path = Path(TESS_TEMPLATE_FILE).expanduser()
    radio_path = Path(RADIO_FILE).expanduser()
    if not tess_path.is_absolute():
        tess_path = PROJECT_ROOT / tess_path
    if not radio_path.is_absolute():
        radio_path = PROJECT_ROOT / radio_path
    tess_path = tess_path.resolve()
    radio_path = radio_path.resolve()
    if not tess_path.is_file():
        raise FileNotFoundError(f"TESS 模板文件不存在：{tess_path}")
    if not radio_path.is_file():
        raise FileNotFoundError(f"ASKAP .ds 文件不存在：{radio_path}")

    # 从实际射电文件名生成标题，避免把某个源或 SBID 写死在代码中。
    radio_name_match = re.match(
        r"(?P<source>.+?)_SB(?P<sbid>\d+)(?:_|$)",
        radio_path.stem,
        flags=re.IGNORECASE,
    )
    if radio_name_match is None:
        raise ValueError(
            "ASKAP 文件名应包含 '<source>_SB<数字>'，无法生成源名和 SBID 标题："
            f"{radio_path.name}"
        )
    source_label = radio_name_match.group("source").replace("_", " ")
    sbid_label = f"SB{radio_name_match.group('sbid')}"

    # 2. 解析当前 TESS_Period_Result.md 的混叠解表。
    result_lines = PERIOD_RESULT_FILE.read_text(encoding="utf-8").splitlines()
    table_header_index = next(
        (
            index
            for index, line in enumerate(result_lines)
            if line.strip().startswith("| Alias rank |")
        ),
        None,
    )
    if table_header_index is None:
        raise ValueError(f"结果文件中找不到混叠解表：{PERIOD_RESULT_FILE}")
    table_headers = [
        cell.strip()
        for cell in result_lines[table_header_index].strip().strip("|").split("|")
    ]
    expected_headers = {
        "Alias rank",
        "Preferred",
        "Period (d)",
        "Period (h)",
        "68% error - (s)",
        "68% error + (s)",
        "Delta BIC",
        "T0 (BJD_TDB)",
        "Cycle-count offsets relative to preferred",
    }
    if not expected_headers.issubset(table_headers):
        raise ValueError(f"混叠解表字段不完整：{table_headers}")

    alias_records: list[dict[str, object]] = []
    for line in result_lines[table_header_index + 2 :]:
        if not line.strip().startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(table_headers) or not cells[0].isdigit():
            continue
        row = dict(zip(table_headers, cells, strict=True))
        alias_records.append(
            {
                "alias_rank": int(row["Alias rank"]),
                "preferred": str(row["Preferred"]).lower() == "yes",
                "period_days": float(row["Period (d)"]),
                "period_hours": float(row["Period (h)"]),
                "period_error_minus_seconds": float(row["68% error - (s)"]),
                "period_error_plus_seconds": float(row["68% error + (s)"]),
                "delta_bic": float(row["Delta BIC"]),
                "t0_bjd_tdb": float(row["T0 (BJD_TDB)"]),
                "cycle_count_offsets": row[
                    "Cycle-count offsets relative to preferred"
                ],
            }
        )
    alias_records.sort(key=lambda row: int(row["alias_rank"]))
    alias_by_rank = {
        int(row["alias_rank"]): row for row in alias_records
    }
    if SELECTED_ALIAS_RANK not in alias_by_rank:
        raise ValueError(
            f"SELECTED_ALIAS_RANK={SELECTED_ALIAS_RANK} 不在结果表中；"
            f"可选值={sorted(alias_by_rank)}"
        )
    if any(float(row["delta_bic"]) >= 6 for row in alias_records):
        raise ValueError("输入结果表包含 Delta BIC >= 6 的行")
    selected_alias = alias_by_rank[SELECTED_ALIAS_RANK]
    central_t0_bjd_tdb = float(selected_alias["t0_bjd_tdb"])
    central_period_days = float(selected_alias["period_days"])

    # 3. 读取所选 alias 的成对 T0/P bootstrap 样本。
    with PERIOD_BOOTSTRAP_FILE.open(encoding="utf-8") as handle:
        bootstrap_rows = list(csv.DictReader(handle, delimiter="\t"))
    required_bootstrap_columns = {
        "alias_rank",
        "sample_index",
        "t0_bjd_tdb",
        "period_days",
    }
    if not bootstrap_rows or required_bootstrap_columns.difference(bootstrap_rows[0]):
        raise ValueError("TESS_Period_Bootstrap.txt 的字段不完整")
    selected_bootstrap_rows = [
        row
        for row in bootstrap_rows
        if int(row["alias_rank"]) == SELECTED_ALIAS_RANK
    ]
    selected_bootstrap_rows.sort(key=lambda row: int(row["sample_index"]))
    sample_indices = np.asarray(
        [int(row["sample_index"]) for row in selected_bootstrap_rows],
        dtype=int,
    )
    t0_samples = np.asarray(
        [float(row["t0_bjd_tdb"]) for row in selected_bootstrap_rows],
        dtype=float,
    )
    period_samples = np.asarray(
        [float(row["period_days"]) for row in selected_bootstrap_rows],
        dtype=float,
    )
    if len(selected_bootstrap_rows) < 100:
        raise ValueError("所选 alias 的 bootstrap 样本少于 100 个")
    if len(np.unique(sample_indices)) != len(sample_indices):
        raise ValueError("所选 alias 的 sample_index 重复")
    valid_samples = (
        np.isfinite(t0_samples)
        & np.isfinite(period_samples)
        & (period_samples > 0)
    )
    t0_samples = t0_samples[valid_samples]
    period_samples = period_samples[valid_samples]
    if len(period_samples) < 100:
        raise ValueError("所选 alias 的有效 bootstrap 样本少于 100 个")

    # 4. 预处理一份 TESS LC；这里只生成相位模板，不重新搜索周期。
    raw_tess = lk.read(tess_path)
    if not hasattr(raw_tess, "time") or not hasattr(raw_tess, "flux"):
        raise TypeError(f"{tess_path.name} 不是 Lightkurve 可识别的 LC")
    if np.asarray(raw_tess.flux.value).ndim != 1:
        raise TypeError("本脚本只读取一维 delivered LC，不读取 TPF 像素立方体")

    available_columns = {
        str(column).lower(): column for column in raw_tess.colnames
    }
    if FLUX_COLUMN is None:
        selected_flux_column = next(
            (available_columns[name] for name in FLUX_PRIORITY if name in available_columns),
            None,
        )
    else:
        selected_flux_column = available_columns.get(FLUX_COLUMN.lower())
    if selected_flux_column is None:
        raise ValueError(
            f"没有可用 TESS flux 列；优先级={FLUX_PRIORITY}；"
            f"实际列={tuple(available_columns)}"
        )

    tess_source = raw_tess.select_flux(selected_flux_column)
    if "quality" not in {
        str(column).lower() for column in tess_source.colnames
    }:
        raise ValueError("TESS light curve 缺少 QUALITY 列")
    tess_quality = np.asarray(tess_source.quality)
    tess_source = tess_source[np.isfinite(tess_quality) & (tess_quality == 0)]
    tess_source = tess_source.remove_nans().normalize()

    tess_time_for_cadence = np.sort(
        np.asarray(tess_source.time.tdb.jd, dtype=float)
    )
    tess_time_differences = np.diff(tess_time_for_cadence)
    tess_time_differences = tess_time_differences[tess_time_differences > 0]
    if len(tess_time_differences) == 0:
        raise ValueError("无法从 TESS LC 估计 cadence")
    tess_cadence_days = float(np.median(tess_time_differences))
    detrend_window_length = int(
        round(DETREND_WINDOW_DAYS / tess_cadence_days)
    )
    if detrend_window_length % 2 == 0:
        detrend_window_length += 1
    detrend_window_length = max(
        detrend_window_length,
        DETREND_POLYORDER + 3,
    )
    maximum_window_length = len(tess_source)
    if maximum_window_length % 2 == 0:
        maximum_window_length -= 1
    detrend_window_length = min(detrend_window_length, maximum_window_length)
    if detrend_window_length <= DETREND_POLYORDER:
        raise ValueError("TESS LC 点数不足以执行去趋势")

    tess_flat, _ = tess_source.flatten(
        window_length=detrend_window_length,
        polyorder=DETREND_POLYORDER,
        break_tolerance=5,
        return_trend=True,
    )
    tess_flat = tess_flat.remove_outliers(
        sigma_lower=OUTLIER_SIGMA_LOWER,
        sigma_upper=OUTLIER_SIGMA_UPPER,
        cenfunc="median",
        stdfunc="mad_std",
    )
    tess_time_bjd_tdb = np.asarray(tess_flat.time.tdb.jd, dtype=float)
    tess_normalized_flux = np.asarray(tess_flat.flux.value, dtype=float)
    tess_finite = np.isfinite(tess_time_bjd_tdb) & np.isfinite(tess_normalized_flux)
    tess_time_bjd_tdb = tess_time_bjd_tdb[tess_finite]
    tess_normalized_flux = tess_normalized_flux[tess_finite]
    if len(tess_time_bjd_tdb) == 0:
        raise ValueError("TESS LC 筛选后没有有限数据点")

    # 5. 直接读取 .ds，并按 DStools 2.0 的顺序恢复 I/V 动态谱。
    if DS_TIME_MODE not in (
        "utc_seconds_since_mjd0",
        "mjd_utc",
        "jd_tdb",
    ):
        raise ValueError(f"不支持 DS_TIME_MODE={DS_TIME_MODE!r}")
    if DSTOOLS_TAVG < 1 or DSTOOLS_FAVG < 1:
        raise ValueError("DSTOOLS_TAVG 和 DSTOOLS_FAVG 必须为正整数")

    with h5py.File(radio_path, "r") as handle:
        required_datasets = {"time", "frequency", "flux", "uvdist"}
        missing_datasets = required_datasets.difference(handle.keys())
        if missing_datasets:
            raise KeyError(f".ds 缺少数据集：{sorted(missing_datasets)}")
        raw_ds_time = np.asarray(handle["time"], dtype=float)
        raw_ds_frequency = np.asarray(handle["frequency"], dtype=float)
        raw_flux = np.asarray(handle["flux"])
        raw_uvdist = np.asarray(handle["uvdist"])
        ds_attributes = dict(handle.attrs)

    if raw_flux.dtype.fields and {"r", "i"}.issubset(raw_flux.dtype.fields):
        raw_flux = raw_flux["r"] + 1j * raw_flux["i"]
    if raw_flux.ndim != 4 or raw_flux.shape[-1] != 4:
        raise ValueError(
            f"当前读取器要求 flux=(baseline,time,frequency,4)，实际={raw_flux.shape}"
        )
    if raw_ds_time.ndim != 1 or raw_ds_frequency.ndim != 1:
        raise ValueError(".ds time 和 frequency 必须是一维数组")
    if raw_uvdist.ndim != 1 or raw_uvdist.shape[0] != raw_flux.shape[0]:
        raise ValueError(".ds uvdist 轴与 baseline 轴不一致")
    if raw_flux.shape[1] != len(raw_ds_time) or raw_flux.shape[2] != len(raw_ds_frequency):
        raise ValueError(
            "flux 的 time/frequency 轴与数据集长度不一致："
            f"flux={raw_flux.shape}，time={len(raw_ds_time)}，"
            f"frequency={len(raw_ds_frequency)}"
        )

    feeds = ds_attributes.get("feeds", "linear")
    if isinstance(feeds, bytes):
        feeds = feeds.decode("utf-8")
    feeds = as_text(feeds).lower()
    if feeds not in ("linear", "circular"):
        raise ValueError(f"当前脚本只支持 linear/circular feeds，实际为 {feeds!r}")

    # DStools 在 _load_data 中先平均 baseline，再按 XX,XY,YX,YY 分解。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        instrumental = np.nanmean(raw_flux, axis=0)
    xx = instrumental[:, :, 0]
    xy = instrumental[:, :, 1]
    yx = instrumental[:, :, 2]
    yy = instrumental[:, :, 3]

    # DStools 的 .ds 通量通常是 Jy，先记录单位换算，重采样后统一用于 Stokes I/V。
    try:
        flux_scale_to_mjy = (
            1.0 * u.Unit(DS_FLUX_UNIT)
        ).to_value(u.mJy)
        frequency_mhz = (
            raw_ds_frequency * u.Unit(DS_FREQUENCY_UNIT)
        ).to_value(u.MHz)
    except (TypeError, ValueError) as error:
        raise ValueError("DS_FLUX_UNIT 或 DS_FREQUENCY_UNIT 无法转换") from error
    frequency_mhz = np.asarray(frequency_mhz, dtype=float)

    # 频率轴按升序排列，并同步重排动态谱。
    frequency_order = np.argsort(frequency_mhz)
    frequency_mhz = frequency_mhz[frequency_order]
    xx = xx[:, frequency_order]
    xy = xy[:, frequency_order]
    yx = yx[:, frequency_order]
    yy = yy[:, frequency_order]

    # 与 DStools trim=True 一致：去除频率两端完全没有有效四极化数据的频道。
    all_polarisation_sum = np.nansum(
        xx + xy + yx + yy,
        axis=0,
    )
    all_polarisation_sum[all_polarisation_sum == 0.0 + 0.0j] = np.nan
    valid_frequency = np.isfinite(all_polarisation_sum)
    if not np.any(valid_frequency):
        raise ValueError(".ds 没有有效频率频道")
    first_channel = int(np.argmax(valid_frequency))
    last_channel = (
        len(valid_frequency)
        if valid_frequency[-1]
        else len(valid_frequency) - int(np.argmax(valid_frequency[::-1])) - 1
    )
    xx = slice_open_end(xx, 0, 0, first_channel, last_channel)
    xy = slice_open_end(xy, 0, 0, first_channel, last_channel)
    yx = slice_open_end(yx, 0, 0, first_channel, last_channel)
    yy = slice_open_end(yy, 0, 0, first_channel, last_channel)
    frequency_mhz = frequency_mhz[first_channel:last_channel]

    # 与 DStools calscans=True 一致：空档内补 NaN，避免重采样跨越扫描间隔。
    if not np.all(np.isfinite(raw_ds_time)):
        raise ValueError(".ds time 含非有限值")
    native_time_differences = np.diff(raw_ds_time)
    if not np.any(native_time_differences > 0.0):
        raise ValueError(".ds time 没有正的 cadence")
    if DS_TIME_MODE == "utc_seconds_since_mjd0":
        time_differences_seconds = native_time_differences
    else:
        time_differences_seconds = native_time_differences * 86400.0
    if np.any(time_differences_seconds <= 0.0):
        raise ValueError(".ds time 必须严格递增")
    cadence_native = float(np.median(native_time_differences[native_time_differences > 0.0]))
    scan_starts = np.r_[
        0,
        np.where(
            time_differences_seconds > DSTOOLS_CORR_DUMPTIME_SECONDS
        )[0] + 1,
    ]
    scan_ends = np.r_[scan_starts[1:] - 1, len(raw_ds_time) - 1]
    time_chunks = []
    correlation_chunks = [[], [], [], []]
    inserted_gap_samples = 0
    for scan_index, (start, end) in enumerate(
        zip(scan_starts, scan_ends, strict=True)
    ):
        time_chunks.append(raw_ds_time[start:end + 1])
        for pol_index, array in enumerate((xx, xy, yx, yy)):
            correlation_chunks[pol_index].append(array[start:end + 1])
        if DSTOOLS_INSERT_SCAN_GAPS and scan_index < len(scan_starts) - 1:
            next_start = scan_starts[scan_index + 1]
            gap_cycles = (raw_ds_time[next_start] - raw_ds_time[end]) / cadence_native
            gap_steps = max(0, int(round(gap_cycles) - 1))
            if gap_steps > 0:
                inserted_gap_samples += gap_steps
                time_chunks.append(
                    raw_ds_time[end]
                    + cadence_native * np.arange(1, gap_steps + 1)
                )
                for chunks in correlation_chunks:
                    chunks.append(
                        np.full(
                            (gap_steps, len(frequency_mhz)),
                            np.nan + 1j * np.nan,
                        )
                    )
    ds_time_native = np.concatenate(time_chunks)
    xx, xy, yx, yy = (
        np.vstack(chunks) for chunks in correlation_chunks
    )

    # 将原始 UTC 时间转换为 BJD_TDB。这里同时需要：
    # 1) ASKAP 台址坐标（EarthLocation，固定）；
    # 2) 目标源天球坐标（SkyCoord，每个源不同）。
    # 目标坐标优先来自 .ds 的 phasecentre，也可由配置区手动覆盖。
    askap_location = EarthLocation.from_geodetic(
        ASKAP_LONGITUDE_DEG * u.deg,
        ASKAP_LATITUDE_DEG * u.deg,
        ASKAP_HEIGHT_M * u.m,
    )
    telescope = as_text(ds_attributes.get("telescope", ""))
    if telescope and telescope.upper() != "ASKAP":
        raise ValueError(f"当前脚本的重心位置只针对 ASKAP，实际 telescope={telescope!r}")
    phasecentre = ds_attributes.get("phasecentre")
    if phasecentre is not None:
        phasecentre = as_text(phasecentre)
    target_coordinate: SkyCoord | None = None
    if TARGET_RA_DEG is not None and TARGET_DEC_DEG is not None:
        target_coordinate = SkyCoord(
            float(TARGET_RA_DEG) * u.deg,
            float(TARGET_DEC_DEG) * u.deg,
            frame="icrs",
        )
    elif phasecentre is not None:
        phase_tokens = str(phasecentre).split()
        if len(phase_tokens) != 2:
            raise ValueError(f"无法解析 .ds phasecentre={phasecentre!r}")
        target_coordinate = SkyCoord(
            phase_tokens[0],
            phase_tokens[1],
            unit=(u.hourangle, u.deg),
            frame="icrs",
        )

    if DS_TIME_MODE == "utc_seconds_since_mjd0":
        mjd_utc = ds_time_native / 86_400.0
        if not 30_000.0 < float(np.nanmedian(mjd_utc)) < 100_000.0:
            raise ValueError(".ds time 不像自 MJD=0 起算的 UTC 秒")
        if target_coordinate is None:
            raise ValueError("UTC .ds 缺少重心改正所需目标坐标")
        utc = Time(mjd_utc, format="mjd", scale="utc", location=askap_location)
        light_time = utc.light_travel_time(target_coordinate, kind="barycentric")
        full_radio_bjd_tdb = np.asarray((utc.tdb + light_time).jd, dtype=float)
        time_explanation = "原始 time 为 MJD0 起算的 UTC 秒；已加一次 ASKAP 重心光行时。"
    elif DS_TIME_MODE == "mjd_utc":
        if not 30_000.0 < float(np.nanmedian(ds_time_native)) < 100_000.0:
            raise ValueError(".ds time 不像 MJD_UTC")
        if target_coordinate is None:
            raise ValueError("MJD_UTC .ds 缺少重心改正所需目标坐标")
        utc = Time(ds_time_native, format="mjd", scale="utc", location=askap_location)
        light_time = utc.light_travel_time(target_coordinate, kind="barycentric")
        full_radio_bjd_tdb = np.asarray((utc.tdb + light_time).jd, dtype=float)
        time_explanation = "原始 time 为 MJD_UTC；已加一次 ASKAP 重心光行时。"
    else:
        if not 2_400_000.0 < float(np.nanmedian(ds_time_native)) < 3_000_000.0:
            raise ValueError(".ds time 不像 JD_TDB")
        full_radio_bjd_tdb = ds_time_native.copy()
        time_explanation = "原始 time 已是 JD_TDB；没有再次加入 time_start 或光行时。"

    if not np.all(np.isfinite(full_radio_bjd_tdb)):
        raise ValueError("转换后的 ASKAP BJD_TDB 含非有限值")
    if not np.all(np.diff(full_radio_bjd_tdb) >= 0):
        raise ValueError(".ds 时间不是单调递增，不能安全执行 DStools 顺序重采样")

    # DStools rebin2D 使用压缩矩阵而不是简单的等长切片平均；先处理四个相关量。
    time_sample_count = len(full_radio_bjd_tdb)
    frequency_sample_count = len(frequency_mhz)
    time_bin_count = time_sample_count // DSTOOLS_TAVG
    frequency_bin_count = frequency_sample_count // DSTOOLS_FAVG
    if time_bin_count < 2 or frequency_bin_count < 2:
        raise ValueError("时间或频率点数不足以执行 DStools 重采样")
    time_remainder = time_sample_count % DSTOOLS_TAVG
    frequency_remainder = frequency_sample_count % DSTOOLS_FAVG
    effective_time_compression = time_sample_count / time_bin_count
    effective_frequency_compression = frequency_sample_count / frequency_bin_count

    time_compressor = rebin(
        time_sample_count, time_bin_count, axis=0
    )
    frequency_compressor = rebin(
        frequency_sample_count, frequency_bin_count, axis=1
    )
    if not np.all(np.sum(time_compressor, axis=0) > 0.0):
        raise RuntimeError("DStools 时间压缩矩阵没有使用全部输入样本")
    if not np.all(np.sum(frequency_compressor, axis=1) > 0.0):
        raise RuntimeError("DStools 频率压缩矩阵没有使用全部输入频道")

    xx = rebin2d(xx, (time_bin_count, frequency_bin_count))
    xy = rebin2d(xy, (time_bin_count, frequency_bin_count))
    yx = rebin2d(yx, (time_bin_count, frequency_bin_count))
    yy = rebin2d(yy, (time_bin_count, frequency_bin_count))
    if feeds == "linear":
        stokes_i_map = np.real((xx + yy) / 2.0)
        stokes_v_map = np.real(1j * (yx - xy) / 2.0)
    else:
        stokes_i_map = np.real((xx + yy) / 2.0)
        stokes_v_map = np.real((xx - yy) / 2.0)
    stokes_i_map *= flux_scale_to_mjy * RADIO_FLUX_SCALE
    stokes_v_map *= flux_scale_to_mjy * RADIO_FLUX_SCALE
    relative_radio_hours = (full_radio_bjd_tdb - full_radio_bjd_tdb[0]) * 24.0
    radio_time_hours = time_compressor @ relative_radio_hours
    radio_bjd_tdb = full_radio_bjd_tdb[0] + radio_time_hours / 24.0
    frequency_mhz = frequency_mhz @ frequency_compressor

    # 6. 沿频率轴生成 Stokes I/V 光变和频道标准误。
    finite_i_count = np.sum(np.isfinite(stokes_i_map), axis=1)
    finite_v_count = np.sum(np.isfinite(stokes_v_map), axis=1)
    stokes_i_lightcurve = np.divide(
        np.nansum(stokes_i_map, axis=1),
        finite_i_count,
        out=np.full(len(finite_i_count), np.nan),
        where=finite_i_count > 0,
    )
    stokes_v_lightcurve = np.divide(
        np.nansum(stokes_v_map, axis=1),
        finite_v_count,
        out=np.full(len(finite_v_count), np.nan),
        where=finite_v_count > 0,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        stokes_i_error = np.nanstd(stokes_i_map, axis=1, ddof=1) / np.sqrt(finite_i_count)
        stokes_v_error = np.nanstd(stokes_v_map, axis=1, ddof=1) / np.sqrt(finite_v_count)
    stokes_i_error[finite_i_count < 2] = np.nan
    stokes_v_error[finite_v_count < 2] = np.nan

    radio_start_bjd_tdb = float(radio_bjd_tdb[0])
    continuous_local_phase = (
        radio_bjd_tdb - radio_start_bjd_tdb
    ) / central_period_days
    maximum_local_phase = float(np.max(continuous_local_phase))

    # 7. 把 TESS 光变折叠成一个周期，并找到模板极小值。
    if TESS_PHASE_BINS < 10:
        raise ValueError("TESS_PHASE_BINS 至少应为 10")
    if not 0.0 < BAND_CONFIDENCE_LEVEL < 1.0:
        raise ValueError("BAND_CONFIDENCE_LEVEL 必须位于 0 和 1 之间")
    tess_ephemeris_phase = np.mod(
        (tess_time_bjd_tdb - central_t0_bjd_tdb) / central_period_days,
        1.0,
    )
    phase_edges = np.linspace(0.0, 1.0, TESS_PHASE_BINS + 1)
    phase_centers = 0.5 * (phase_edges[:-1] + phase_edges[1:])
    binned_tess_median = np.full(TESS_PHASE_BINS, np.nan)
    binned_tess_count = np.zeros(TESS_PHASE_BINS, dtype=int)
    for bin_index, (left_edge, right_edge) in enumerate(
        zip(phase_edges[:-1], phase_edges[1:], strict=True)
    ):
        phase_mask = (
            (tess_ephemeris_phase >= left_edge)
            & (tess_ephemeris_phase < right_edge)
        )
        binned_tess_count[bin_index] = int(np.count_nonzero(phase_mask))
        if binned_tess_count[bin_index] > 0:
            binned_tess_median[bin_index] = float(
                np.nanmedian(tess_normalized_flux[phase_mask])
            )
    if not np.any(np.isfinite(binned_tess_median)):
        raise ValueError("TESS 相位模板没有有限点")
    optical_minimum_phase = float(
        phase_centers[np.nanargmin(binned_tess_median)]
    )

    # 8. 传播所选 alias 的光学极小包络，并合并相互重叠的区间。
    q_low = 0.5 * (1.0 - BAND_CONFIDENCE_LEVEL)
    q_high = 1.0 - q_low
    minimum_cycle_start = int(
        np.floor(
            (radio_bjd_tdb[0] - central_t0_bjd_tdb) / central_period_days
            - optical_minimum_phase
        )
    ) - 1
    minimum_cycle_end = int(
        np.ceil(
            (radio_bjd_tdb[-1] - central_t0_bjd_tdb) / central_period_days
            - optical_minimum_phase
        )
    ) + 1
    raw_minimum_bands: list[dict[str, object]] = []
    for cycle_number in range(minimum_cycle_start, minimum_cycle_end + 1):
        nominal_event_bjd_tdb = central_t0_bjd_tdb + (
            cycle_number + optical_minimum_phase
        ) * central_period_days
        nominal_local_phase = (
            nominal_event_bjd_tdb - radio_start_bjd_tdb
        ) / central_period_days
        sampled_event_bjd_tdb = t0_samples + (
            cycle_number + optical_minimum_phase
        ) * period_samples
        sampled_local_phase = (
            sampled_event_bjd_tdb - radio_start_bjd_tdb
        ) / central_period_days
        band_low, band_high = np.quantile(
            sampled_local_phase,
            (q_low, q_high),
        )
        if band_high < 0.0 or band_low > maximum_local_phase:
            continue
        raw_minimum_bands.append(
            {
                "cycle_number": cycle_number,
                "nominal_event_bjd_tdb": float(nominal_event_bjd_tdb),
                "nominal_local_phase": float(nominal_local_phase),
                "phase_low": float(band_low),
                "phase_high": float(band_high),
                "phase_width_cycles": float(band_high - band_low),
            }
        )
    raw_minimum_bands.sort(key=lambda row: float(row["phase_low"]))
    optical_minimum_bands: list[dict[str, object]] = []
    for band in raw_minimum_bands:
        if optical_minimum_bands and float(band["phase_low"]) <= float(
            optical_minimum_bands[-1]["phase_high"]
        ):
            current = optical_minimum_bands[-1]
            current["phase_high"] = max(
                float(current["phase_high"]), float(band["phase_high"])
            )
            current["phase_width_cycles"] = float(
                current["phase_high"] - current["phase_low"]
            )
            current["cycle_number"] = (
                f"{current['cycle_number']},{band['cycle_number']}"
            )
        else:
            optical_minimum_bands.append(dict(band))

    # 9. 绘制 TESS、宽带 I/V 和 I/V 动态谱四个面板。
    finite_i_values = np.abs(stokes_i_map[np.isfinite(stokes_i_map)])
    finite_v_values = np.abs(stokes_v_map[np.isfinite(stokes_v_map)])
    if len(finite_i_values) == 0 or len(finite_v_values) == 0:
        raise ValueError("I/V 动态谱没有有限像素")
    if not 0.0 < COLORBAR_AUTO_QUANTILE <= 1.0:
        raise ValueError("COLORBAR_AUTO_QUANTILE 必须位于 (0, 1] 之间")
    if DYNAMIC_I_COLORBAR_LIMIT_MJY is None:
        i_color_limit = float(
            np.quantile(finite_i_values, COLORBAR_AUTO_QUANTILE)
        )
    else:
        i_color_limit = float(DYNAMIC_I_COLORBAR_LIMIT_MJY)
    if DYNAMIC_V_COLORBAR_LIMIT_MJY is None:
        v_color_limit = float(
            np.quantile(finite_v_values, COLORBAR_AUTO_QUANTILE)
        )
    else:
        v_color_limit = float(DYNAMIC_V_COLORBAR_LIMIT_MJY)
    if (
        not np.isfinite(i_color_limit)
        or i_color_limit <= 0
        or not np.isfinite(v_color_limit)
        or v_color_limit <= 0
    ):
        raise ValueError("I/V 动态谱色标上限必须为正有限值")

    phase_at_radio_start = float(
        np.mod(
            (radio_start_bjd_tdb - central_t0_bjd_tdb) / central_period_days,
            1.0,
        )
    )
    tess_local_phase_one_cycle = np.mod(
        tess_ephemeris_phase - phase_at_radio_start,
        1.0,
    )
    template_local_phase_one_cycle = np.mod(
        phase_centers - phase_at_radio_start,
        1.0,
    )
    repeat_count = int(np.ceil(maximum_local_phase)) + 1

    # 主数据轴与 colorbar 使用独立列；colorbar 不再压缩动态谱或其他面板宽度。
    figure = plt.figure(figsize=(13.0, 13.5))
    grid = figure.add_gridspec(
        4,
        2,
        width_ratios=(1.0, 0.035),
        height_ratios=(1.20, 1.00, 1.15, 1.15),
        hspace=0.0,
        wspace=0.025,
    )
    axes = [figure.add_subplot(grid[0, 0])]
    axes.extend(
        figure.add_subplot(grid[index, 0], sharex=axes[0])
        for index in range(1, 4)
    )
    cax_i = figure.add_subplot(grid[2, 1])
    cax_v = figure.add_subplot(grid[3, 1])
    # 置信带表示光学极小值外推到射电时序后的相位范围，
    # 因此只覆盖 TESS 光变和 ASKAP 宽带 I/V 光变；动态谱保留原始颜色信息。
    for axis_index, axis in enumerate(axes[:2]):
        for band_index, band in enumerate(optical_minimum_bands):
            visible_low = max(0.0, float(band["phase_low"]))
            visible_high = min(maximum_local_phase, float(band["phase_high"]))
            if visible_high >= visible_low:
                axis.axvspan(
                    visible_low,
                    visible_high,
                    facecolor="#F2C94C",
                    alpha=0.28,
                    edgecolor="#A6A6A6",
                    linewidth=0.7,
                    label=(
                        f"TESS minimum {BAND_CONFIDENCE_LEVEL * 100:.0f}% interval"
                        if axis_index == 0 and band_index == 0
                        else "_nolegend_"
                    ),
                    zorder=2.5,
                )
    for axis in axes[:2]:
        axis.grid(color="0.86", linewidth=0.6, alpha=0.8)
    for axis in axes[2:]:
        axis.grid(False)

    for repeat_index in range(repeat_count):
        repeated_raw_phase = tess_local_phase_one_cycle + repeat_index
        raw_visible = (
            (repeated_raw_phase <= maximum_local_phase)
            & np.isfinite(tess_normalized_flux)
        )
        axes[0].scatter(
            repeated_raw_phase[raw_visible],
            tess_normalized_flux[raw_visible],
            s=4,
            alpha=0.13,
            color="#4682B4",
            rasterized=True,
        )
        repeated_template_phase = template_local_phase_one_cycle + repeat_index
        template_visible = (
            (repeated_template_phase <= maximum_local_phase)
            & np.isfinite(binned_tess_median)
        )
        template_sort = np.argsort(repeated_template_phase[template_visible])
        axes[0].plot(
            repeated_template_phase[template_visible][template_sort],
            binned_tess_median[template_visible][template_sort],
            "o-",
            markersize=3,
            linewidth=1.1,
            color="tab:red",
            label="TESS binned median" if repeat_index == 0 else None,
        )
    axes[0].set(ylabel="Normalized TESS flux")
    add_panel_label(axes[0], "TESS Phasefolding", fontsize=11)
    axes[0].legend(fontsize=8)

    axes[1].errorbar(
        continuous_local_phase,
        stokes_i_lightcurve,
        yerr=stokes_i_error,
        fmt="o-",
        markersize=3,
        linewidth=0.9,
        capsize=2,
        color="black",
        label="Stokes I",
    )
    axes[1].errorbar(
        continuous_local_phase,
        stokes_v_lightcurve,
        yerr=stokes_v_error,
        fmt="o-",
        markersize=3,
        linewidth=0.9,
        capsize=2,
        color="tab:red",
        label="Stokes V",
    )
    axes[1].axhline(0.0, color="0.45", linestyle="--", linewidth=0.9)
    axes[1].set(ylabel=f"Flux density ({RADIO_FLUX_UNIT_LABEL})")
    add_panel_label(axes[1], "Radio Lightcurve", fontsize=11)
    axes[1].legend(fontsize=8)

    phase_edges_for_map = np.r_[
        continuous_local_phase[0] - 0.5 * np.diff(continuous_local_phase[:2]),
        0.5 * (continuous_local_phase[:-1] + continuous_local_phase[1:]),
        continuous_local_phase[-1] + 0.5 * np.diff(continuous_local_phase[-2:]),
    ]
    frequency_edges_for_map = np.r_[
        frequency_mhz[0] - 0.5 * np.diff(frequency_mhz[:2]),
        0.5 * (frequency_mhz[:-1] + frequency_mhz[1:]),
        frequency_mhz[-1] + 0.5 * np.diff(frequency_mhz[-2:]),
    ]
    stokes_i_image = axes[2].pcolormesh(
        phase_edges_for_map,
        frequency_edges_for_map,
        stokes_i_map.T,
        shading="auto",
        cmap="coolwarm",
        vmin=-i_color_limit,
        vmax=i_color_limit,
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=True,
    )
    axes[2].set(ylabel="Frequency (MHz)")
    add_panel_label(axes[2], "Stokes I", fontsize=11)
    figure.colorbar(stokes_i_image, cax=cax_i)
    cax_i.set_ylabel(f"Stokes I ({RADIO_FLUX_UNIT_LABEL})")

    stokes_v_image = axes[3].pcolormesh(
        phase_edges_for_map,
        frequency_edges_for_map,
        stokes_v_map.T,
        shading="auto",
        cmap="coolwarm",
        vmin=-v_color_limit,
        vmax=v_color_limit,
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=True,
    )
    axes[3].set(
        xlim=(0.0, maximum_local_phase),
        xlabel="Continuous local phase from radio start (cycles)",
        ylabel="Frequency (MHz)",
    )
    add_panel_label(axes[3], "Stokes V", fontsize=11)
    figure.colorbar(stokes_v_image, cax=cax_v)
    cax_v.set_ylabel(f"Stokes V ({RADIO_FLUX_UNIT_LABEL})")
    figure.suptitle(
        f"{source_label} | {sbid_label}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    output_stem = f"{radio_path.stem}_P{central_period_days:.4f}"
    figure.subplots_adjust(left=0.07, right=0.965, top=0.95, bottom=0.06,
                           hspace=0.0, wspace=0.025)
    figure.savefig(
        OUTPUT_DIR / f"{output_stem}_Combined.png",
        dpi=FIGURE_DPI,
    )
    plt.close(figure)

    # 10. 保存每个真实时间箱；与本项目既有输出命名保持一致。
    radio_bins_path = OUTPUT_DIR / f"{output_stem}_PhaseInfo.csv"
    with radio_bins_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "bjd_tdb",
                "continuous_local_phase",
                "stokes_i_mjy",
                "stokes_i_channel_sem_mjy",
                "stokes_v_mjy",
                "stokes_v_channel_sem_mjy",
                "finite_i_channels",
                "finite_v_channels",
                "requested_tavg",
                "requested_favg",
                "effective_time_compression",
                "effective_frequency_compression",
                "inserted_gap_samples",
            )
        )
        for index in range(len(radio_bjd_tdb)):
            writer.writerow(
                (
                    f"{radio_bjd_tdb[index]:.17g}",
                    f"{continuous_local_phase[index]:.17g}",
                    f"{stokes_i_lightcurve[index]:.17g}",
                    f"{stokes_i_error[index]:.17g}",
                    f"{stokes_v_lightcurve[index]:.17g}",
                    f"{stokes_v_error[index]:.17g}",
                    int(finite_i_count[index]),
                    int(finite_v_count[index]),
                    DSTOOLS_TAVG,
                    DSTOOLS_FAVG,
                    f"{effective_time_compression:.17g}",
                    f"{effective_frequency_compression:.17g}",
                    inserted_gap_samples,
                )
            )

    # 保留原项目的精简输出：仅 PNG 与对应的数值 CSV。

    maximum_band_width = max(
        (float(band["phase_width_cycles"]) for band in optical_minimum_bands),
        default=float("nan"),
    )
    print(
        f"相位验证完成：alias rank={SELECTED_ALIAS_RANK}；"
        f"P={central_period_days:.12f} d；"
        f"光学极小={optical_minimum_phase:.6f}；"
        f"最大95%相位区间宽度={maximum_band_width:.6f} cycle；"
        f"输出={OUTPUT_DIR.resolve()}"
    )
    if len(alias_records) > 1:
        print(
            f"结果表中还有 {len(alias_records) - 1} 个 Delta BIC < 6 的竞争 alias；"
            "本图只显示配置的 SELECTED_ALIAS_RANK。"
        )


if __name__ == "__main__":
    main()
