#!/usr/bin/env python3
"""用一份或多份 TESS delivered light curve 测量常周期星历。

流程：显式路径输入 → 光变预处理 → Lomb--Scargle 初始周期 → 分段计时
→ alias 候选比较 → Statsmodels WLS 星历 → SciPy paired bootstrap 误差。
"""

from __future__ import annotations

import csv
from itertools import product
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

# 本脚本只读取 LC，不使用 TPF/PRF；仅在导入 Lightkurve 时屏蔽其可选依赖提示。
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Warning: the tpfmodel submodule is not available.*",
    )
    import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import find_peaks
from scipy.stats import DegenerateDataWarning, bootstrap

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from science_utils import add_panel_label  # noqa: E402


# ============================ 参数配置区 ============================
# 根据本脚本所在的 Code 文件夹自动定位项目根目录；Mac 和 Docker 均可使用。
PROJECT_DIR = Path(__file__).resolve().parents[2]

# 使用本项目现有的完整 HST 输入路径；默认生成当前共同星历所用的 S29+S69 结果。
TESS_INPUT_PATHS = [
    # Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/TESS_Data/2MASS_J01033563-5515561_A/"
    #      "hlsp_qlp_tess_ffi_s0029-0000000616014335_tess_v01_llc/"
    #      "hlsp_qlp_tess_ffi_s0029-0000000616014335_tess_v01_llc.fits"),
    Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/TESS_Data/2MASS_J01033563-5515561_A/"
         "hlsp_qlp_tess_ffi_s0069-0000000616014335_tess_v01_llc/"
         "hlsp_qlp_tess_ffi_s0069-0000000616014335_tess_v01_llc.fits"),
    # Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/TESS_Data/2MASS_J01033563-5515561_A/"
    #      "tess2026111101500-s0103-0000000206502540-0305-s/"
    #      "tess2026111101500-s0103-0000000206502540-0305-s_lc.fits"),
    # Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/TESS_Data/2MASS_J01033563-5515561_A/"
    #      "tess2026137223500-s0104-0000000206502540-0306-s/"
    #      "tess2026137223500-s0104-0000000206502540-0306-s_lc.fits"),
]

LC_FILE_PATTERNS = ("*_lc.fits", "*_llc.fits")

# 下面这些直接填写 FITS 文件名；留空表示读取每个输入文件夹中的全部 LC。
# 例如：SELECTED_LC_NAMES = ("my_sector69_lc.fits",)
SELECTED_LC_NAMES = ()

# 也可以直接把 TESS_INPUT_PATHS 中的一项写成文件路径，例如：
# PROJECT_DIR / "DATA/TESS_Data/example_lc.fits
# 2 "

# 输出目录同样从项目根目录生成；脚本会生成 Markdown、TXT 和 PNG。
OUTPUT_DIR = Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/TESS_Period")

# flux 列按优先级自动选择；需要固定某列时，把 FLUX_COLUMN 改成列名。
FLUX_COLUMN = None
FLUX_PRIORITY = ("pdcsap_flux", "det_flux", "kspsap_flux", "sap_flux", "flux")

# Lomb--Scargle 搜索和模板参数。
PERIOD_MIN_DAYS = 0.165
PERIOD_MAX_DAYS = 0.168
TEMPLATE_HARMONICS = 1
SAMPLES_PER_PEAK = 50

# 每份光变的慢趋势用 Lightkurve.flatten 的 Savitzky--Golay 方法去除。
DETREND_WINDOW_DAYS = 1.5
DETREND_POLYORDER = 2
OUTLIER_SIGMA_UPPER = 5.0
OUTLIER_SIGMA_LOWER = 7.0

# 正式采用每个计时片段约含 10 个完整周期。
# 敏感性测试表明该尺度能保留全部输入，并避免过细分段造成的相关计时点和边界拟合；
# 片段内只拟合一个模板相位，不是独立测量该段的周期。
TIMING_CYCLES_PER_CHUNK = 10.0

# 只用于复现旧的四日分段；正式分析保持 None。
TIMING_CHUNK_DAYS_OVERRIDE = None

# 数据间断、时间覆盖率和采样点覆盖率。
GAP_FACTOR = 5.0
MIN_CHUNK_TIME_COVERAGE = 0.75
MIN_CHUNK_POINT_COVERAGE = 0.70
MIN_CHUNK_POINTS_ABSOLUTE = 50

# 整份光变的最少有效点数，与单个计时片段的点数阈值分开。
MIN_LIGHT_CURVE_POINTS = 300
MIN_TIMINGS_TOTAL = 5
RECOMMENDED_TIMINGS_PER_INPUT = 8

# alias 搜索：LS 局部峰 × 各输入之间可能漏数/多数的整周数偏移。
PEAK_CANDIDATES = 10
PEAK_MIN_SEPARATION_SAMPLES = 10
# 对每个非参考输入尝试少算或多算 0--5 个完整周期。
ALIAS_OFFSETS = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
MAX_ALIAS_COMBINATIONS = 100_000
DELTA_BIC_REPORT_LIMIT = 6.0
CYCLE_SCALE = 10_000.0

# bootstrap 只在固定 alias 条件下估计周期不确定度。
BOOTSTRAP_ITERATIONS = 5_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.68
MIN_VALID_BOOTSTRAP_FRACTION = 0.95
# 仅用于直方图显示：隐藏两端合计 0.2% 的尾部样本，避免少数退化重采样压扁主体分布。
# 这不会删去样本，也不会改变下方 68% 周期误差的计算。
BOOTSTRAP_PLOT_CENTRAL_FRACTION = 0.998
RANDOM_SEED = 20_260_824
FIGURE_DPI = 300


def main() -> None:
    """按配置区顺序执行一次周期测量。"""

    # 1. 验证输入目录/文件路径，避免依赖当前工作目录或临时上传目录。
    if not TESS_INPUT_PATHS:
        raise ValueError("TESS_INPUT_PATHS 至少需要填写一个数据文件夹")
    if not OUTPUT_DIR.is_absolute():
        raise ValueError("OUTPUT_DIR 必须是完整绝对路径")
    if TIMING_CYCLES_PER_CHUNK <= 1.0:
        raise ValueError("TIMING_CYCLES_PER_CHUNK 必须大于 1")
    if TIMING_CHUNK_DAYS_OVERRIDE is not None and TIMING_CHUNK_DAYS_OVERRIDE <= 0.0:
        raise ValueError("TIMING_CHUNK_DAYS_OVERRIDE 必须为 None 或正数")
    if not 0.0 < MIN_CHUNK_TIME_COVERAGE <= 1.0:
        raise ValueError("MIN_CHUNK_TIME_COVERAGE 必须介于 0 和 1 之间")
    if not 0.0 < MIN_CHUNK_POINT_COVERAGE <= 1.0:
        raise ValueError("MIN_CHUNK_POINT_COVERAGE 必须介于 0 和 1 之间")
    if GAP_FACTOR <= 1.0:
        raise ValueError("GAP_FACTOR 必须大于 1")
    if not 0.0 < BOOTSTRAP_PLOT_CENTRAL_FRACTION < 1.0:
        raise ValueError("BOOTSTRAP_PLOT_CENTRAL_FRACTION 必须介于 0 和 1 之间")

    input_paths = [Path(path).expanduser() for path in TESS_INPUT_PATHS]
    if any(not path.is_absolute() for path in input_paths):
        raise ValueError("TESS_INPUT_PATHS 中每一项都必须是完整绝对路径")
    tess_paths = []
    for input_path in input_paths:
        if input_path.is_file():
            tess_paths.append(input_path)
            continue
        if not input_path.is_dir():
            raise FileNotFoundError(f"输入路径不存在：{input_path}")
        matches = [
            file_path
            for pattern in LC_FILE_PATTERNS
            for file_path in input_path.rglob(pattern)
            if file_path.is_file()
        ]
        if SELECTED_LC_NAMES:
            matches = [file_path for file_path in matches if file_path.name in SELECTED_LC_NAMES]
        if not matches:
            raise FileNotFoundError(f"文件夹中没有匹配的 TESS LC：{input_path}")
        tess_paths.extend(matches)
    tess_paths = sorted(
        {path.resolve() for path in tess_paths},
        key=lambda path: str(path).lower(),
    )
    if not tess_paths:
        raise ValueError("没有找到可分析的 TESS LC")
    if len({str(path) for path in tess_paths}) != len(tess_paths):
        raise ValueError("输入路径中存在重复文件")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 读取 light curve，并选择每份产品实际存在的 flux 列。
    selected_inputs = []
    for path in tess_paths:
        raw = lk.read(path)
        columns = tuple(str(column).lower() for column in raw.colnames)
        if not hasattr(raw, "time") or not hasattr(raw, "flux"):
            raise TypeError(f"{path.name} 不是 Lightkurve 可识别的 light curve")

        selected_column = FLUX_COLUMN
        if selected_column is None:
            selected_column = next(
                (column for column in FLUX_PRIORITY if column in columns), None
            )
        if selected_column is None or str(selected_column).lower() not in columns:
            raise ValueError(
                f"{path.name} 没有可用 flux 列；实际列为 {columns}"
            )

        source = raw.select_flux(str(selected_column))
        source_columns = tuple(str(column).lower() for column in source.colnames)
        if "quality" not in source_columns:
            raise ValueError(f"{path.name} 缺少 QUALITY 列")
        if source.flux_err is None:
            raise ValueError(f"{path.name} 没有可用的 flux_err")
        selected_inputs.append(
            {
                "path": path,
                "raw": raw,
                "source": source,
                "column": str(selected_column),
                "metadata": dict(getattr(raw, "meta", {}) or {}),
            }
        )

    # 3. 每份光变独立进行 QUALITY 筛选、归一化、去趋势和异常点剔除。
    curves = []
    for item in selected_inputs:
        path = item["path"]
        raw = item["raw"]
        source = item["source"]
        quality = np.asarray(source.quality)
        source = source[np.isfinite(quality) & (quality == 0)].remove_nans()
        if len(source) < MIN_LIGHT_CURVE_POINTS:
            raise ValueError(
                f"{path.name} 质量筛选后有效点少于 {MIN_LIGHT_CURVE_POINTS}"
            )

        source = source.normalize()
        source_time = np.asarray(source.time.tdb.jd, dtype=float)
        differences = np.diff(np.sort(source_time[np.isfinite(source_time)]))
        differences = differences[differences > 0]
        if len(differences) == 0:
            raise ValueError(f"{path.name} 无法估计 cadence")

        cadence_days = float(np.median(differences))
        window_length = int(round(DETREND_WINDOW_DAYS / cadence_days))
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, DETREND_POLYORDER + 3)
        largest_window = len(source) if len(source) % 2 else len(source) - 1
        window_length = min(window_length, largest_window)
        if window_length <= DETREND_POLYORDER:
            raise ValueError(f"{path.name} 有效点数不足以执行去趋势")

        flattened, _ = source.flatten(
            window_length=window_length,
            polyorder=DETREND_POLYORDER,
            break_tolerance=5,
            return_trend=True,
        )
        flattened, _ = flattened.remove_outliers(
            sigma_lower=OUTLIER_SIGMA_LOWER,
            sigma_upper=OUTLIER_SIGMA_UPPER,
            cenfunc="median",
            stdfunc="mad_std",
            return_mask=True,
        )

        time = np.asarray(flattened.time.tdb.jd, dtype=float)
        flux = np.asarray(flattened.flux.value, dtype=float) - 1.0
        error = np.asarray(flattened.flux_err.value, dtype=float)
        finite = np.isfinite(time) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
        if np.count_nonzero(finite) < MIN_LIGHT_CURVE_POINTS:
            raise ValueError(
                f"{path.name} 有限且误差为正的点少于 {MIN_LIGHT_CURVE_POINTS}"
            )
        finite_order = np.argsort(time[finite])
        time = time[finite][finite_order]
        flux = flux[finite][finite_order]
        error = error[finite][finite_order]

        metadata = item["metadata"]
        sector = metadata.get("SECTOR")
        label = f"S{sector}" if sector is not None else path.stem
        curves.append(
            {
                "path": path,
                "raw_count": len(raw),
                "column": item["column"],
                "metadata": metadata,
                "label": label,
                "time": time,
                "flux": flux,
                "error": error,
                "cadence_days": cadence_days,
            }
        )

    curves.sort(key=lambda curve: float(np.median(curve["time"])))
    for input_id, curve in enumerate(curves):
        curve["input_id"] = input_id

    # 不同 delivered 产品若覆盖同一时段，会造成重复加权；保留提示并继续运行。
    for left_index, left in enumerate(curves):
        for right in curves[left_index + 1 :]:
            overlap = min(np.max(left["time"]), np.max(right["time"])) - max(
                np.min(left["time"]), np.min(right["time"])
            )
            if overlap > max(left["cadence_days"], right["cadence_days"]):
                warnings.warn(
                    f"{left['label']} 与 {right['label']} 重叠 {overlap:.5f} d；"
                    "请确认不是同一时段的重复产品。",
                    stacklevel=2,
                )

    time_bjd_tdb = np.concatenate([curve["time"] for curve in curves])
    flux = np.concatenate([curve["flux"] for curve in curves])
    flux_error = np.concatenate([curve["error"] for curve in curves])
    input_ids = np.concatenate(
        [np.full(len(curve["time"]), curve["input_id"], dtype=int) for curve in curves]
    )
    order = np.argsort(time_bjd_tdb)
    time_bjd_tdb = time_bjd_tdb[order]
    flux = flux[order]
    flux_error = flux_error[order]
    input_ids = input_ids[order]

    # 4. 用 Astropy Lomb--Scargle 搜索初始周期和局部候选峰。
    if not 0 < PERIOD_MIN_DAYS < PERIOD_MAX_DAYS:
        raise ValueError("必须满足 0 < PERIOD_MIN_DAYS < PERIOD_MAX_DAYS")
    periodogram = LombScargle(
        time_bjd_tdb,
        flux,
        flux_error,
        fit_mean=True,
        center_data=True,
        nterms=TEMPLATE_HARMONICS,
    )
    frequency, power = periodogram.autopower(
        minimum_frequency=1.0 / PERIOD_MAX_DAYS,
        maximum_frequency=1.0 / PERIOD_MIN_DAYS,
        samples_per_peak=SAMPLES_PER_PEAK,
        method="fastchi2",
    )
    peak_indices, _ = find_peaks(power, distance=PEAK_MIN_SEPARATION_SAMPLES)
    peak_indices = np.unique(np.append(peak_indices, int(np.nanargmax(power))))
    peak_indices = peak_indices[np.argsort(power[peak_indices])[::-1]][:PEAK_CANDIDATES]
    if len(peak_indices) == 0:
        raise RuntimeError("Lomb--Scargle 周期图没有可用候选峰")
    best_power_index = int(np.nanargmax(power))
    initial_frequency = float(frequency[best_power_index])
    initial_period_days = 1.0 / initial_frequency

    # 5. 用 LS 模板在观测基线中部确定一个初始光学极大 T0。
    wanted_reference_time = 0.5 * (np.min(time_bjd_tdb) + np.max(time_bjd_tdb))
    reference_input_id = min(
        range(len(curves)),
        key=lambda index: abs(np.median(curves[index]["time"]) - wanted_reference_time),
    )
    reference_center = float(np.median(curves[reference_input_id]["time"]))
    arbitrary_epoch = float(np.min(time_bjd_tdb))
    nearest_cycle = int(np.rint((reference_center - arbitrary_epoch) / initial_period_days))
    search_center = arbitrary_epoch + nearest_cycle * initial_period_days
    maximum = minimize_scalar(
        lambda trial_time: -float(periodogram.model([trial_time], initial_frequency)[0]),
        bounds=(search_center - initial_period_days / 2, search_center + initial_period_days / 2),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    if not maximum.success:
        raise RuntimeError("无法定位 LS 模板光学极大")
    initial_t0_bjd_tdb = float(maximum.x)
    template_level = float(
        np.median(
            periodogram.model(
                initial_t0_bjd_tdb + np.linspace(0, initial_period_days, 512, endpoint=False),
                initial_frequency,
            )
        )
    )

    # 6. 根据初始周期定义片段长度，在每个连续数据块内拟合模板相位。
    # 每段默认约含 10 个周期，不是每个周期单独生成一个计时点。
    if TIMING_CHUNK_DAYS_OVERRIDE is None:
        chunk_days = TIMING_CYCLES_PER_CHUNK * initial_period_days
    else:
        chunk_days = float(TIMING_CHUNK_DAYS_OVERRIDE)

    timing_records = []
    for curve in curves:
        curve_time = curve["time"]
        curve_flux = curve["flux"]
        curve_error = curve["error"]
        time_steps = np.diff(curve_time)
        gap_limit_days = GAP_FACTOR * curve["cadence_days"]
        continuous_block_id = np.r_[0, np.cumsum(time_steps > gap_limit_days)]
        curve["continuous_block_count"] = int(len(np.unique(continuous_block_id)))

        for block_number in np.unique(continuous_block_id):
            block_mask = continuous_block_id == block_number
            block_time = curve_time[block_mask]
            block_flux = curve_flux[block_mask]
            block_error = curve_error[block_mask]
            if len(block_time) < 2:
                continue

            chunk_starts = np.arange(block_time[0], block_time[-1], chunk_days)
            expected_points = chunk_days / curve["cadence_days"]
            required_points = max(
                MIN_CHUNK_POINTS_ABSOLUTE,
                int(np.ceil(MIN_CHUNK_POINT_COVERAGE * expected_points)),
            )
            for chunk_start in chunk_starts:
                mask = (
                    (block_time >= chunk_start)
                    & (block_time < chunk_start + chunk_days)
                )
                if np.count_nonzero(mask) < required_points:
                    continue
                chunk_time = block_time[mask]
                chunk_flux = block_flux[mask]
                chunk_error = block_error[mask]
                if np.ptp(chunk_time) < MIN_CHUNK_TIME_COVERAGE * chunk_days:
                    continue

                # curve_fit 的回调只在这里用于拟合相位。
                def shifted_template(
                    trial_time: np.ndarray,
                    amplitude: float,
                    offset: float,
                    phase_shift: float,
                ) -> np.ndarray:
                    model_time = trial_time - phase_shift * initial_period_days
                    template = periodogram.model(model_time, initial_frequency)
                    return offset + amplitude * (template - template_level)

                try:
                    parameters, covariance = curve_fit(
                        shifted_template,
                        chunk_time,
                        chunk_flux,
                        p0=(1.0, 0.0, 0.0),
                        sigma=chunk_error,
                        absolute_sigma=False,
                        bounds=((0.0, -np.inf, -0.20), (np.inf, np.inf, 0.20)),
                        maxfev=20_000,
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    continue

                phase_variance = float(covariance[2, 2])
                if not np.isfinite(phase_variance) or phase_variance <= 0:
                    continue
                phase_shift = float(parameters[2])
                phase_error = float(np.sqrt(phase_variance))
                cycle = int(
                    np.rint(
                        (np.mean(chunk_time) - initial_t0_bjd_tdb)
                        / initial_period_days
                    )
                )
                timing_records.append(
                    {
                        "input_id": int(curve["input_id"]),
                        "input_label": str(curve["label"]),
                        "continuous_block": int(block_number),
                        "chunk_start_bjd_tdb": float(chunk_start),
                        "chunk_end_bjd_tdb": float(chunk_start + chunk_days),
                        "point_count": int(len(chunk_time)),
                        "expected_point_count": float(expected_points),
                        "cycle": cycle,
                        "phase_shift": phase_shift,
                        "timing_bjd_tdb": (
                            initial_t0_bjd_tdb
                            + (cycle + phase_shift) * initial_period_days
                        ),
                        "timing_error_days": phase_error * initial_period_days,
                    }
                )

    timings_per_input = {
        int(curve["input_id"]): sum(
            record["input_id"] == curve["input_id"]
            for record in timing_records
        )
        for curve in curves
    }
    if len(timing_records) < MIN_TIMINGS_TOTAL:
        raise RuntimeError(
            f"只有 {len(timing_records)} 个计时点，"
            f"少于 MIN_TIMINGS_TOTAL={MIN_TIMINGS_TOTAL}"
        )
    for curve in curves:
        count = timings_per_input[int(curve["input_id"])]
        if count < RECOMMENDED_TIMINGS_PER_INPUT:
            warnings.warn(
                f"{curve['label']} 只得到 {count} 个计时点；"
                "该输入的周期误差可能对分段定义较敏感。",
                stacklevel=2,
            )
    timing_input_id = np.asarray([record["input_id"] for record in timing_records], dtype=int)
    timing_bjd_tdb = np.asarray([record["timing_bjd_tdb"] for record in timing_records], dtype=float)
    timing_error_days = np.asarray([record["timing_error_days"] for record in timing_records], dtype=float)

    # 7. 枚举长时间空档中的整数周数 alias，并用 Statsmodels WLS 拟合 M0。
    non_reference_ids = [
        curve["input_id"] for curve in curves if curve["input_id"] != reference_input_id
    ]
    estimated_combinations = len(peak_indices) * len(ALIAS_OFFSETS) ** len(non_reference_ids)
    if estimated_combinations > MAX_ALIAS_COMBINATIONS:
        raise RuntimeError(
            f"预计 alias 组合数 {estimated_combinations} 超过 MAX_ALIAS_COMBINATIONS="
            f"{MAX_ALIAS_COMBINATIONS}；请减少输入文件、候选峰或 ALIAS_OFFSETS。"
        )

    timing_origin = float(np.median(timing_bjd_tdb))
    candidate_solutions = []
    seen_cycle_arrays = set()
    alias_boundary = max(abs(int(offset)) for offset in ALIAS_OFFSETS)
    for seed_rank, peak_index in enumerate(peak_indices, start=1):
        seed_period = 1.0 / float(frequency[peak_index])
        base_cycles = np.rint((timing_bjd_tdb - initial_t0_bjd_tdb) / seed_period).astype(int)
        for offsets in product(ALIAS_OFFSETS, repeat=len(non_reference_ids)):
            cycles = base_cycles.copy()
            offset_by_input = {int(curve["input_id"]): 0 for curve in curves}
            for input_id, offset in zip(non_reference_ids, offsets, strict=True):
                cycles[timing_input_id == input_id] += int(offset)
                offset_by_input[input_id] = int(offset)
            cycles -= int(np.rint(np.median(cycles[timing_input_id == reference_input_id])))
            cycle_key = tuple(int(value) for value in cycles)
            if cycle_key in seen_cycle_arrays:
                continue
            seen_cycle_arrays.add(cycle_key)

            cycle_origin = int(np.rint(np.median(cycles)))
            scaled_cycle = (cycles - cycle_origin) / CYCLE_SCALE
            design = np.column_stack((np.ones(len(scaled_cycle)), scaled_cycle))
            try:
                fit = sm.WLS(
                    timing_bjd_tdb - timing_origin,
                    design,
                    weights=1.0 / timing_error_days**2,
                ).fit()
            except (ValueError, np.linalg.LinAlgError):
                continue

            period_days = float(fit.params[1] / CYCLE_SCALE)
            if not np.isfinite(period_days) or period_days <= 0:
                continue
            t0_bjd_tdb = float(timing_origin + fit.params[0] - period_days * cycle_origin)
            candidate_solutions.append(
                {
                    "seed_rank": seed_rank,
                    "cycles": cycles,
                    "period_days": period_days,
                    "t0_bjd_tdb": t0_bjd_tdb,
                    "bic": float(fit.bic),
                    "offset_by_input": offset_by_input,
                    "boundary_hit": any(abs(int(offset)) == alias_boundary for offset in offsets),
                }
            )

    if not candidate_solutions:
        raise RuntimeError("所有 alias 候选的 WLS 拟合均失败")
    candidate_solutions.sort(key=lambda candidate: candidate["bic"])
    best_bic = candidate_solutions[0]["bic"]
    for rank, candidate in enumerate(candidate_solutions, start=1):
        candidate["alias_rank"] = rank
        candidate["delta_bic"] = candidate["bic"] - best_bic
    reported_aliases = [
        candidate for candidate in candidate_solutions
        if candidate["delta_bic"] < DELTA_BIC_REPORT_LIMIT
    ]

    # 用首选解作为零点，记录每个输入相对于首选解的整周数差。
    # 先去掉参考输入的共同整数平移；这不会改变周期，只是固定表示方式。
    preferred_cycles = np.asarray(candidate_solutions[0]["cycles"], dtype=int)
    reference_mask = timing_input_id == reference_input_id
    for candidate in candidate_solutions:
        relative_cycles = np.asarray(candidate["cycles"], dtype=int) - preferred_cycles
        common_shift = int(np.rint(np.median(relative_cycles[reference_mask])))
        relative_cycles = relative_cycles - common_shift
        relative_offsets = {}
        for curve in curves:
            input_id = int(curve["input_id"])
            values = tuple(
                int(value)
                for value in np.unique(relative_cycles[timing_input_id == input_id])
            )
            relative_offsets[input_id] = values
        candidate["relative_offsets"] = relative_offsets

        offset_parts = []
        for curve in curves:
            input_id = int(curve["input_id"])
            values = relative_offsets[input_id]
            value_text = "/".join(f"{value:+d}" for value in values)
            reference_text = " (reference)" if input_id == reference_input_id else ""
            offset_parts.append(f"{curve['label']}{reference_text}: {value_text}")
        candidate["relative_offset_text"] = "; ".join(offset_parts)

    first_rejected = next(
        (
            candidate
            for candidate in candidate_solutions
            if candidate["delta_bic"] >= DELTA_BIC_REPORT_LIMIT
        ),
        None,
    )

    # 8. 对每个未被 ΔBIC=6 排除的 alias 做成对 bootstrap，误差为第16/84百分位。
    def bootstrap_statistic(
        sample_cycles: np.ndarray,
        sample_times: np.ndarray,
        sample_errors: np.ndarray,
    ) -> np.ndarray:
        if len(np.unique(sample_cycles)) < 2:
            return np.asarray((np.nan, np.nan), dtype=float)
        local_origin = int(np.rint(np.median(sample_cycles)))
        local_cycle = (sample_cycles - local_origin) / CYCLE_SCALE
        local_design = np.column_stack((np.ones(len(local_cycle)), local_cycle))
        try:
            local_fit = sm.WLS(
                sample_times - timing_origin,
                local_design,
                weights=1.0 / sample_errors**2,
            ).fit()
        except (ValueError, np.linalg.LinAlgError):
            return np.asarray((np.nan, np.nan), dtype=float)
        local_period = float(local_fit.params[1] / CYCLE_SCALE)
        local_t0 = float(timing_origin + local_fit.params[0] - local_period * local_origin)
        if not np.isfinite(local_t0) or not np.isfinite(local_period) or local_period <= 0:
            return np.asarray((np.nan, np.nan), dtype=float)
        return np.asarray((local_t0, local_period), dtype=float)

    for candidate in reported_aliases:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DegenerateDataWarning)
            bootstrap_result = bootstrap(
                (
                    np.asarray(candidate["cycles"], dtype=int),
                    timing_bjd_tdb,
                    timing_error_days,
                ),
                bootstrap_statistic,
                paired=True,
                vectorized=False,
                n_resamples=BOOTSTRAP_ITERATIONS,
                method="percentile",
                confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
                rng=np.random.default_rng(RANDOM_SEED),
            )
        distribution = np.asarray(bootstrap_result.bootstrap_distribution, dtype=float)
        if distribution.ndim != 2 or distribution.shape[0] != 2:
            raise RuntimeError(f"bootstrap 返回数组形状异常：{distribution.shape}")
        valid = np.isfinite(distribution[0]) & np.isfinite(distribution[1]) & (distribution[1] > 0)
        if np.count_nonzero(valid) < MIN_VALID_BOOTSTRAP_FRACTION * BOOTSTRAP_ITERATIONS:
            raise RuntimeError(
                f"alias {candidate['alias_rank']} 只有 {np.count_nonzero(valid)} 个有效 bootstrap 样本"
            )
        t0_samples = distribution[0, valid]
        period_samples = distribution[1, valid]
        package_interval = bootstrap_result.confidence_interval
        period_low = float(package_interval.low[1])
        period_high = float(package_interval.high[1])
        if not np.isfinite(period_low) or not np.isfinite(period_high):
            raise RuntimeError(
                f"alias {candidate['alias_rank']} 的 SciPy bootstrap 置信区间不是有限值"
            )
        candidate["t0_samples"] = t0_samples
        candidate["period_samples"] = period_samples
        candidate["valid_bootstrap"] = int(len(period_samples))
        candidate["error_minus_seconds"] = (
            candidate["period_days"] - period_low
        ) * 86_400.0
        candidate["error_plus_seconds"] = (
            period_high - candidate["period_days"]
        ) * 86_400.0

    # 主结果用 Markdown 表格，方便直接在编辑器或 GitHub 中阅读。
    timing_input_summary = "; ".join(
        f"{curve['label']}={timings_per_input[curve['input_id']]}"
        for curve in curves
    )
    block_summary = "; ".join(
        f"{curve['label']}={curve['continuous_block_count']}"
        for curve in curves
    )
    if TIMING_CHUNK_DAYS_OVERRIDE is None:
        chunk_mode = f"{TIMING_CYCLES_PER_CHUNK:g} initial periods"
    else:
        chunk_mode = f"fixed override {chunk_days:.9f} d"
    result_path = OUTPUT_DIR / "TESS_Period_Result.md"
    result_lines = [
        "# TESS period result",
        "",
        f"- Input light curves: {len(curves)}",
        f"- Timing points: {len(timing_records)}",
        f"- Timing chunk: {chunk_days:.9f} d ({chunk_mode})",
        f"- Timing points per input: {timing_input_summary}",
        f"- Continuous blocks per input: {block_summary}",
        "- Bootstrap interval: central 68% (16th--84th percentiles)",
        "- Alias reporting rule: Delta BIC < 6",
        f"- Alias offsets searched per non-reference input: {min(ALIAS_OFFSETS):+d} to {max(ALIAS_OFFSETS):+d}",
        f"- Unique alias candidates evaluated: {len(candidate_solutions)}",
        f"- Candidates retained with Delta BIC < {DELTA_BIC_REPORT_LIMIT:g}: {len(reported_aliases)}",
        "",
        "| Alias rank | Preferred | Period (d) | Period (h) | 68% error - (s) | 68% error + (s) | Delta BIC | T0 (BJD_TDB) | Cycle-count offsets relative to preferred |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for candidate in reported_aliases:
        result_lines.append(
            f"| {candidate['alias_rank']} | "
            f"{'yes' if candidate['alias_rank'] == 1 else 'no'} | "
            f"{candidate['period_days']:.12f} | "
            f"{candidate['period_days'] * 24.0:.10f} | "
            f"{candidate['error_minus_seconds']:.6f} | "
            f"{candidate['error_plus_seconds']:.6f} | "
            f"{candidate['delta_bic']:.6f} | "
            f"{candidate['t0_bjd_tdb']:.12f} | "
            f"{candidate['relative_offset_text']} |"
        )
    result_lines.extend(
        [
            "",
            "## Alias offset interpretation",
            "",
            "`+1` means that this input is assigned one more complete cycle than in the preferred solution; `-1` means one fewer complete cycle.",
            "The reference input is fixed at `+0`. These are integer cycle-count differences, not phase uncertainties or period errors.",
        ]
    )
    if first_rejected is not None:
        result_lines.append(
            f"The first candidate excluded by the Delta BIC threshold has "
            f"P={first_rejected['period_days']:.12f} d and "
            f"Delta BIC={first_rejected['delta_bic']:.6f}; no bootstrap error was calculated for it."
        )
    result_lines.extend(["", "## Input paths", ""])
    result_lines.extend(f"- `{path}`" for path in tess_paths)
    result_lines.extend(
        [
            "",
            "> The period centre is obtained from the alias-conditioned WLS fit. "
            "The 68% errors are obtained from paired bootstrap resampling of the timing points.",
        ]
    )
    result_path.write_text("\n".join(result_lines) + "\n", encoding="utf-8")

    timings_path = OUTPUT_DIR / "TESS_Period_Timings.txt"
    with timings_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "input_label",
                "continuous_block",
                "chunk_start_bjd_tdb",
                "chunk_end_bjd_tdb",
                "point_count",
                "expected_point_count",
                "cycle",
                "phase_shift_cycle",
                "timing_bjd_tdb",
                "timing_error_seconds",
            )
        )
        for record in timing_records:
            writer.writerow(
                (
                    record["input_label"],
                    record["continuous_block"],
                    f'{record["chunk_start_bjd_tdb"]:.12f}',
                    f'{record["chunk_end_bjd_tdb"]:.12f}',
                    record["point_count"],
                    f'{record["expected_point_count"]:.2f}',
                    record["cycle"],
                    f'{record["phase_shift"]:.9f}',
                    f'{record["timing_bjd_tdb"]:.12f}',
                    f'{record["timing_error_days"] * 86400.0:.6f}',
                )
            )

    bootstrap_path = OUTPUT_DIR / "TESS_Period_Bootstrap.txt"
    with bootstrap_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("alias_rank", "sample_index", "t0_bjd_tdb", "period_days"))
        for candidate in reported_aliases:
            for sample_index, (sample_t0, sample_period) in enumerate(
                zip(candidate["t0_samples"], candidate["period_samples"], strict=True)
            ):
                writer.writerow(
                    (
                        candidate["alias_rank"], sample_index,
                        f"{sample_t0:.12f}", f"{sample_period:.12f}",
                    )
                )

    sectors = [
        str(curve["metadata"].get("SECTOR"))
        for curve in curves
        if curve["metadata"].get("SECTOR") is not None
    ]
    title = "_".join(f"S{sector}" for sector in sectors) if sectors else "TESS_inputs"
    preferred = reported_aliases[0]
    bootstrap_period_offset_seconds = (
        np.asarray(preferred["period_samples"], dtype=float) - preferred["period_days"]
    ) * 86_400.0
    plot_tail_fraction = (1.0 - BOOTSTRAP_PLOT_CENTRAL_FRACTION) / 2.0
    central_low, central_high = np.quantile(
        bootstrap_period_offset_seconds,
        (plot_tail_fraction, 1.0 - plot_tail_fraction),
    )
    plot_mask = (
        (bootstrap_period_offset_seconds >= central_low)
        & (bootstrap_period_offset_seconds <= central_high)
    )
    plotted_bootstrap_offsets = bootstrap_period_offset_seconds[plot_mask]
    hidden_tail_samples = int(len(bootstrap_period_offset_seconds) - len(plotted_bootstrap_offsets))
    central_fraction_percent = 100.0 * BOOTSTRAP_PLOT_CENTRAL_FRACTION

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    add_panel_label(axes[0], "Lomb–Scargle", fontsize=11)
    add_panel_label(axes[1], f"Bootstrap | central {central_fraction_percent:.1f}%", fontsize=11)
    axes[0].plot(1.0 / frequency, power, color="tab:blue", linewidth=1.2)
    axes[0].axvline(
        initial_period_days,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"P = {initial_period_days:.9f} d",
    )
    axes[0].axvline(
        preferred["period_days"],
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"WLS P = {preferred['period_days']:.9f} d",
    )
    axes[0].set(
        xlim=(PERIOD_MIN_DAYS, PERIOD_MAX_DAYS),
        xlabel="Period (days)",
        ylabel="Lomb--Scargle power",
        title=None,
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].hist(
        plotted_bootstrap_offsets,
        bins=40,
        color="tab:orange",
        edgecolor="white",
        linewidth=0.4,
    )
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].axvline(central_low, color="0.35", linestyle=":", linewidth=1.0)
    axes[1].axvline(central_high, color="0.35", linestyle=":", linewidth=1.0)
    axes[1].set(
        xlim=(central_low, central_high),
        xlabel="Bootstrap period minus WLS period (s)",
        ylabel="Count",
        title=None,
    )
    axes[1].text(
        0.03,
        0.96,
        f"Shown: central {central_fraction_percent:.1f}%\n"
        f"Tail samples hidden: {hidden_tail_samples}/{len(bootstrap_period_offset_seconds)}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
    )
    axes[1].grid(alpha=0.20)
    figure.savefig(OUTPUT_DIR / "TESS_Period.png", dpi=FIGURE_DPI)
    plt.close(figure)

    # 清理本脚本旧版本产生的冗余文件，避免用户误读旧结果。
    for old_name in (
        "TESS_Period_Result.txt", "period_result.txt", "ephemeris_samples.txt", "timing_measurements.txt",
        "alias_candidates.txt", "lightcurve.txt", "periodogram.txt",
        "period_summary.txt", "period_diagnostic.png", "period_validation.png",
    ):
        (OUTPUT_DIR / old_name).unlink(missing_ok=True)

    # 终端保留可直接汇报的数值；详细样本在 TXT 中。
    print(
        f"输入文件数: {len(curves)}；计时点数: {len(timing_records)}；"
        f"alias候选: {len(candidate_solutions)}；Delta BIC<6: {len(reported_aliases)}；"
        f"输出: {OUTPUT_DIR}"
    )
    print(
        f"计时片段: {chunk_days:.6f} d；"
        f"每份输入: "
        + ", ".join(
            f"{curve['label']}={timings_per_input[curve['input_id']]}"
            for curve in curves
        )
    )
    for candidate in reported_aliases:
        label = "首选解" if candidate["alias_rank"] == 1 else f"alias {candidate['alias_rank']}"
        print(
            f"{label}: P={candidate['period_days']:.12f} d；"
            f"68%误差=-{candidate['error_minus_seconds']:.5f}/"
            f"+{candidate['error_plus_seconds']:.5f} s；"
            f"Delta BIC={candidate['delta_bic']:.3f}；"
            f"整周偏移={candidate['relative_offset_text']}"
        )
    if first_rejected is not None:
        print(
            f"首个 Delta BIC>=6 候选: P={first_rejected['period_days']:.12f} d；"
            f"Delta BIC={first_rejected['delta_bic']:.3f}；未计算 bootstrap 误差"
        )


if __name__ == "__main__":
    main()
