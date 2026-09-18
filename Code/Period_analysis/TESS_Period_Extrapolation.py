#!/usr/bin/env python3
"""把 TESS_Period.py 的星历传播到一份或多份 ASKAP 动态谱观测。"""

from __future__ import annotations

import csv
import re
from pathlib import Path
import sys

import h5py  # 这里只读取 .ds 的观测时间；动态谱绘图由相位验证脚本完成。
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers
from scipy.stats import circmean

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from science_utils import add_panel_label  # noqa: E402


# ============================ 参数配置区 ============================
# 一次运行只修改这里；路径既可以写项目相对路径，也可以写绝对路径。

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 上游周期结果与下游外推均使用本项目既有的 HST Result 目录。
EPHEMERIS_RESULT_DIRS = [
    Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/TESS_Period/s69"),
    Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/TESS_Period/s29+69"),
    Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/TESS_Period/s29+69+103+104"),
]

# 本项目的三段 flare 动态谱；可按需要增删。
SBID_DS_PATHS = [
    Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/Ds/2MASS_J01033563-5515561_A/Flare/"
         "2MASS_J01033563-5515561_A_SB59565_beam22.ds"),
    Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/Ds/2MASS_J01033563-5515561_A/Flare/"
         "2MASS_J01033563-5515561_A_SB66827_beam10.ds"),
    Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/Ds/2MASS_J01033563-5515561_A/Flare/"
         "2MASS_J01033563-5515561_A_SB68040_beam10.ds"),
]
OUTPUT_DIR = Path("/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/TESS_Period_Extrapolation")

# 仅在 .ds 时间是 UTC 时用于重心改正；None 时从每个 .ds 的 phasecentre 读取。
TARGET_RA_DEG: float | None = None
TARGET_DEC_DEG: float | None = None

# ASKAP 固定台址坐标，不是目标源的 RA/Dec；不同源仍共用这三个值。
ASKAP_LONGITUDE_DEG = 116.631425
ASKAP_LATITUDE_DEG = -26.697000
ASKAP_HEIGHT_M = 361.0

# 当前项目的 .ds 原始 time 是自 MJD=0 起算的 UTC 秒。
DS_TIME_MODE = "utc_seconds_since_mjd0"
ALLOWED_DS_TIME_MODES = (
    "utc_seconds_since_mjd0",
    "mjd_utc",
    "jd_tdb",
)

CONFIDENCE_LEVEL = 0.95
PHASE_GOAL_CYCLES = 0.25
PHASE_MAX_CYCLES = 0.50
FIGURE_DPI = 300

# ==================================================================


def main() -> None:
    """读取多套上游星历，计算各 ASKAP 文件的相位中心和传播不确定度。"""

    iers.conf.auto_download = False
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 展开并检查一个或多个 TESS_Period 结果目录。
    period_result_dirs: list[Path] = []
    seen_period_dirs: set[Path] = set()
    for configured_path in EPHEMERIS_RESULT_DIRS:
        period_dir = Path(configured_path).expanduser()
        if not period_dir.is_absolute():
            period_dir = PROJECT_ROOT / period_dir
        period_dir = period_dir.resolve()
        if not period_dir.is_dir():
            raise FileNotFoundError(f"周期结果目录不存在：{period_dir}")
        if period_dir not in seen_period_dirs:
            period_result_dirs.append(period_dir)
            seen_period_dirs.add(period_dir)
    if not period_result_dirs:
        raise FileNotFoundError("EPHEMERIS_RESULT_DIRS 没有可用的周期结果目录")

    # 2. 逐目录读取 Markdown 周期表、计时范围和成对 Bootstrap 样本。
    period_runs: list[dict[str, object]] = []
    bootstrap_by_solution: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    used_run_labels: set[str] = set()
    for run_index, period_dir in enumerate(period_result_dirs, start=1):
        result_file = period_dir / "TESS_Period_Result.md"
        bootstrap_file = period_dir / "TESS_Period_Bootstrap.txt"
        timings_file = period_dir / "TESS_Period_Timings.txt"
        for required_file in (result_file, bootstrap_file, timings_file):
            if not required_file.is_file():
                raise FileNotFoundError(f"缺少 TESS_Period 输出：{required_file}")

        with timings_file.open(encoding="utf-8") as handle:
            timing_rows = list(csv.DictReader(handle, delimiter="\t"))
        required_timing_columns = {
            "input_label",
            "chunk_start_bjd_tdb",
            "chunk_end_bjd_tdb",
        }
        if not timing_rows or required_timing_columns.difference(timing_rows[0]):
            raise ValueError(f"{timings_file} 的字段不完整")
        input_labels = list(
            dict.fromkeys(
                str(row["input_label"]).strip()
                for row in timing_rows
                if str(row["input_label"]).strip()
            )
        )
        base_run_label = "+".join(input_labels) or period_dir.name
        run_label = base_run_label
        if run_label in used_run_labels:
            run_label = f"{base_run_label} [{period_dir.name}]"
        used_run_labels.add(run_label)
        timing_start = min(
            float(row["chunk_start_bjd_tdb"])
            for row in timing_rows
            if np.isfinite(float(row["chunk_start_bjd_tdb"]))
        )
        timing_end = max(
            float(row["chunk_end_bjd_tdb"])
            for row in timing_rows
            if np.isfinite(float(row["chunk_end_bjd_tdb"]))
        )

        result_lines = result_file.read_text(encoding="utf-8").splitlines()
        table_header_index = next(
            (
                index
                for index, line in enumerate(result_lines)
                if line.strip().startswith("| Alias rank |")
            ),
            None,
        )
        if table_header_index is None:
            raise ValueError(f"结果文件中找不到混叠解表：{result_file}")
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
            raise ValueError(
                f"{result_file} 的混叠解表字段不完整；实际字段={table_headers}"
            )

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
                    "period_run_id": run_index,
                    "period_run_label": run_label,
                    "period_result_dir": str(period_dir),
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
        if not alias_records:
            raise ValueError(f"{result_file} 的混叠解表没有数据行")
        if len({row["alias_rank"] for row in alias_records}) != len(alias_records):
            raise ValueError(f"{result_file} 中 alias_rank 重复")
        if sum(bool(row["preferred"]) for row in alias_records) != 1:
            raise ValueError(f"{result_file} 必须恰好有一个 Preferred=yes")
        if any(
            not np.isfinite(float(row["period_days"]))
            or float(row["period_days"]) <= 0
            or not np.isfinite(float(row["delta_bic"]))
            or float(row["delta_bic"]) < 0
            or float(row["delta_bic"]) >= 6
            for row in alias_records
        ):
            raise ValueError(f"{result_file} 包含无效周期或不满足 Delta BIC < 6 的行")
        alias_records.sort(key=lambda row: int(row["alias_rank"]))

        with bootstrap_file.open(encoding="utf-8") as handle:
            bootstrap_rows = list(csv.DictReader(handle, delimiter="\t"))
        required_bootstrap_columns = {
            "alias_rank",
            "sample_index",
            "t0_bjd_tdb",
            "period_days",
        }
        if not bootstrap_rows or required_bootstrap_columns.difference(bootstrap_rows[0]):
            raise ValueError(f"{bootstrap_file} 的字段不完整")
        for alias in alias_records:
            rank = int(alias["alias_rank"])
            rows_for_alias = [
                row for row in bootstrap_rows if int(row["alias_rank"]) == rank
            ]
            rows_for_alias.sort(key=lambda row: int(row["sample_index"]))
            sample_indices = np.asarray(
                [int(row["sample_index"]) for row in rows_for_alias],
                dtype=int,
            )
            t0_samples = np.asarray(
                [float(row["t0_bjd_tdb"]) for row in rows_for_alias],
                dtype=float,
            )
            period_samples = np.asarray(
                [float(row["period_days"]) for row in rows_for_alias],
                dtype=float,
            )
            if len(rows_for_alias) < 100:
                raise ValueError(f"{run_label} alias {rank} 的 Bootstrap 样本少于 100 个")
            if len(np.unique(sample_indices)) != len(sample_indices):
                raise ValueError(f"{run_label} alias {rank} 的 sample_index 重复")
            valid = (
                np.isfinite(t0_samples)
                & np.isfinite(period_samples)
                & (period_samples > 0)
            )
            t0_samples = t0_samples[valid]
            period_samples = period_samples[valid]
            if len(period_samples) < 100:
                raise ValueError(f"{run_label} alias {rank} 的有效 Bootstrap 样本少于 100 个")
            bootstrap_by_solution[(run_index, rank)] = (t0_samples, period_samples)

        period_runs.append(
            {
                "period_run_id": run_index,
                "period_run_label": run_label,
                "period_result_dir": str(period_dir),
                "input_labels": input_labels,
                "timing_count": len(timing_rows),
                "timing_start": timing_start,
                "timing_end": timing_end,
                "timing_span_days": timing_end - timing_start,
                "aliases": alias_records,
            }
        )

    # 3. 选择最长计时跨度且只有一个竞争解的结果作为周期数值对照。
    unique_runs = [run for run in period_runs if len(run["aliases"]) == 1]
    reference_run = (
        max(unique_runs, key=lambda run: float(run["timing_span_days"]))
        if unique_runs
        else None
    )
    reference_alias = reference_run["aliases"][0] if reference_run else None
    for run in period_runs:
        aliases = run["aliases"]
        for alias in aliases:
            alias["delta_period_to_reference_seconds"] = (
                (float(alias["period_days"]) - float(reference_alias["period_days"]))
                * 86_400.0
                if reference_alias is not None
                else None
            )
            alias["nearest_to_reference"] = False
        if reference_alias is not None and run is not reference_run:
            nearest_alias = min(
                aliases,
                key=lambda alias: abs(float(alias["delta_period_to_reference_seconds"])),
            )
            nearest_alias["nearest_to_reference"] = True
        elif reference_alias is not None:
            aliases[0]["nearest_to_reference"] = True

    # 4. 读取配置区指定的 ASKAP .ds 文件；射电时间只转换一次。
    radio_paths: list[Path] = []
    seen_radio_paths: set[Path] = set()
    for configured_path in SBID_DS_PATHS:
        radio_path = Path(configured_path).expanduser()
        if not radio_path.is_absolute():
            radio_path = PROJECT_ROOT / radio_path
        radio_path = radio_path.resolve()
        if not radio_path.is_file():
            raise FileNotFoundError(f"SBID .ds 文件不存在：{radio_path}")
        if radio_path.suffix.lower() != ".ds":
            raise ValueError(f"SBID 输入不是 .ds 文件：{radio_path}")
        if radio_path not in seen_radio_paths:
            radio_paths.append(radio_path)
            seen_radio_paths.add(radio_path)
    if not radio_paths:
        raise FileNotFoundError("SBID_DS_PATHS 没有可用的 .ds 文件")
    if DS_TIME_MODE not in ALLOWED_DS_TIME_MODES:
        raise ValueError(f"不支持 DS_TIME_MODE={DS_TIME_MODE!r}")
    if not 0.0 < CONFIDENCE_LEVEL < 1.0:
        raise ValueError("CONFIDENCE_LEVEL 必须位于 0 和 1 之间")

    askap_location = EarthLocation.from_geodetic(
        ASKAP_LONGITUDE_DEG * u.deg,
        ASKAP_LATITUDE_DEG * u.deg,
        ASKAP_HEIGHT_M * u.m,
    )
    q_low = 0.5 * (1.0 - CONFIDENCE_LEVEL)
    q_high = 1.0 - q_low
    observation_rows: list[dict[str, object]] = []
    for radio_path in radio_paths:
        # SBID 从文件名自动提取，避免用户在配置区重复填写并造成错配。
        sbid_match = re.search(r"(?i)(?:^|_)SB(\d+)(?:_|$)", radio_path.stem)
        if sbid_match is None:
            raise ValueError(f"无法从文件名识别 SBID：{radio_path.name}")
        sbid = f"SB{sbid_match.group(1)}"
        with h5py.File(radio_path, "r") as handle:
            if "time" not in handle:
                raise KeyError(f"{radio_path.name} 中没有 time 数据集")
            raw_time = np.asarray(handle["time"], dtype=float)
            phasecentre = handle.attrs.get("phasecentre")
        raw_time = raw_time[np.isfinite(raw_time)]
        if raw_time.ndim != 1 or len(raw_time) < 2:
            raise ValueError(f"{radio_path.name} 的 time 必须是一维且至少有两个点")

        target_coordinate: SkyCoord | None = None
        if TARGET_RA_DEG is not None and TARGET_DEC_DEG is not None:
            target_coordinate = SkyCoord(
                float(TARGET_RA_DEG) * u.deg,
                float(TARGET_DEC_DEG) * u.deg,
                frame="icrs",
            )
        elif phasecentre is not None:
            if isinstance(phasecentre, bytes):
                phasecentre = phasecentre.decode("utf-8")
            target_coordinate = SkyCoord(str(phasecentre), frame="icrs")

        if DS_TIME_MODE == "utc_seconds_since_mjd0":
            radio_mjd_utc = raw_time / 86_400.0
            if not 30_000.0 < float(np.median(radio_mjd_utc)) < 100_000.0:
                raise ValueError(f"{radio_path.name} 的 time 不像 MJD0 起算的 UTC 秒")
            if target_coordinate is None:
                raise ValueError(f"{radio_path.name} 缺少重心改正所需目标坐标")
            radio_time_utc = Time(
                radio_mjd_utc,
                format="mjd",
                scale="utc",
                location=askap_location,
            )
            light_time = radio_time_utc.light_travel_time(
                target_coordinate,
                kind="barycentric",
            )
            radio_bjd_tdb = np.asarray((radio_time_utc.tdb + light_time).jd, dtype=float)
        elif DS_TIME_MODE == "mjd_utc":
            if not 30_000.0 < float(np.median(raw_time)) < 100_000.0:
                raise ValueError(f"{radio_path.name} 的 time 不像 MJD_UTC")
            if target_coordinate is None:
                raise ValueError(f"{radio_path.name} 缺少重心改正所需目标坐标")
            radio_time_utc = Time(
                raw_time,
                format="mjd",
                scale="utc",
                location=askap_location,
            )
            light_time = radio_time_utc.light_travel_time(
                target_coordinate,
                kind="barycentric",
            )
            radio_bjd_tdb = np.asarray((radio_time_utc.tdb + light_time).jd, dtype=float)
        else:
            if not 2_400_000.0 < float(np.median(raw_time)) < 3_000_000.0:
                raise ValueError(f"{radio_path.name} 的 time 不像 JD_TDB")
            radio_bjd_tdb = raw_time.copy()

        radio_bjd_tdb = np.sort(radio_bjd_tdb)
        observation_rows.append(
            {
                "radio_file": str(radio_path),
                "radio_name": radio_path.stem,
                "sbid": sbid,
                # 本脚本最终把相位定义在射电观测起始时刻。
                "start_bjd_tdb": float(radio_bjd_tdb[0]),
                # 终点只用于判断 TESS 基线与整段射电观测的时间关系。
                "end_bjd_tdb": float(radio_bjd_tdb[-1]),
            }
        )

    # 同一 SBID 只能对应一个输出 stem，重复配置时直接报错，避免静默覆盖。
    sbid_to_paths: dict[str, list[Path]] = {}
    for observation in observation_rows:
        sbid_to_paths.setdefault(str(observation["sbid"]), []).append(
            Path(str(observation["radio_file"]))
        )
    duplicated_sbids = {
        sbid: paths for sbid, paths in sbid_to_paths.items() if len(paths) > 1
    }
    if duplicated_sbids:
        details = "; ".join(
            f"{sbid}: {', '.join(path.name for path in paths)}"
            for sbid, paths in duplicated_sbids.items()
        )
        raise ValueError(f"一个 SBID 配置了多个 .ds 文件，无法生成唯一输出：{details}")

    # 5. 对每套星历的每个 alias，只传播 ASKAP 射电观测起点的相位。
    prediction_rows: list[dict[str, object]] = []
    for run in period_runs:
        run_index = int(run["period_run_id"])
        run_label = str(run["period_run_label"])
        optical_start = float(run["timing_start"])
        optical_end = float(run["timing_end"])
        for observation in observation_rows:
            radio_start = float(observation["start_bjd_tdb"])
            radio_end = float(observation["end_bjd_tdb"])
            if optical_end < radio_start:
                time_relation = "前向外推"
            elif optical_start > radio_end:
                time_relation = "后向外推"
            else:
                time_relation = "回顾性基线内比较"
            for alias in run["aliases"]:
                rank = int(alias["alias_rank"])
                t0_samples, period_samples = bootstrap_by_solution[(run_index, rank)]
                central_t0 = float(alias["t0_bjd_tdb"])
                central_period = float(alias["period_days"])
                epoch = float(observation["start_bjd_tdb"])
                phase_samples = np.mod(
                    (epoch - t0_samples) / period_samples,
                    1.0,
                )
                phase_center = float(circmean(phase_samples, high=1.0, low=0.0))
                centered_phase = (phase_samples - phase_center + 0.5) % 1.0 - 0.5
                phase_low_relative, phase_high_relative = np.quantile(
                    centered_phase,
                    (q_low, q_high),
                )
                phase_width = float(phase_high_relative - phase_low_relative)
                phase_low_absolute = float(
                    (phase_center + phase_low_relative) % 1.0
                )
                phase_high_absolute = float(
                    (phase_center + phase_high_relative) % 1.0
                )
                prediction_rows.append(
                    {
                        "period_run_id": run_index,
                        "period_run_label": run_label,
                        "period_result_dir": run["period_result_dir"],
                        "input_labels": "+".join(run["input_labels"]),
                        "timing_start_bjd_tdb": optical_start,
                        "timing_end_bjd_tdb": optical_end,
                        "alias_rank": rank,
                        "preferred": bool(alias["preferred"]),
                        "cycle_count_offsets": alias["cycle_count_offsets"],
                        "period_days": central_period,
                        "period_error_minus_seconds": float(
                            alias["period_error_minus_seconds"]
                        ),
                        "period_error_plus_seconds": float(
                            alias["period_error_plus_seconds"]
                        ),
                        "delta_bic": float(alias["delta_bic"]),
                        "delta_period_to_reference_seconds": alias[
                            "delta_period_to_reference_seconds"
                        ],
                        "nearest_to_reference": bool(alias["nearest_to_reference"]),
                        "radio_file": observation["radio_file"],
                        "radio_name": observation["radio_name"],
                        "sbid": observation["sbid"],
                        "radio_start_bjd_tdb": epoch,
                        "central_ephemeris_phase": float(
                            np.mod((epoch - central_t0) / central_period, 1.0)
                        ),
                        "circular_sample_center": phase_center,
                        "confidence_level": CONFIDENCE_LEVEL,
                        "phase_low_relative": float(phase_low_relative),
                        "phase_high_relative": float(phase_high_relative),
                        "phase_low_absolute": phase_low_absolute,
                        "phase_high_absolute": phase_high_absolute,
                        "wraps_phase_zero": phase_low_absolute > phase_high_absolute,
                        "phase_width_cycles": phase_width,
                        "quarter_cycle_pass": phase_width <= PHASE_GOAL_CYCLES,
                        "half_cycle_pass": phase_width <= PHASE_MAX_CYCLES,
                        "phase_localized": phase_width <= PHASE_MAX_CYCLES,
                        "time_relation": time_relation,
                    }
                )

    # 6. 准备逐 SBID 输出所需的解索引；每个 SBID 只使用自己的观测行。
    solution_records = [
        (run, alias)
        for run in period_runs
        for alias in run["aliases"]
    ]
    start_by_key = {
        (
            int(row["period_run_id"]),
            int(row["alias_rank"]),
            str(row["sbid"]),
        ): row
        for row in prediction_rows
    }
    solution_count = len(solution_records)
    solution_labels = [
        f"{run['period_run_label']} · A{alias['alias_rank']}"
        f"{'*' if alias['preferred'] else ''}"
        for run, alias in solution_records
    ]
    y_positions = np.arange(solution_count, dtype=float)

    # 7. 每个 SBID 单独保存完整表格、Markdown 和一张相位条形图。
    successful_sbids: list[str] = []
    run_colors = ("#002FA7", "#B2182B", "#E66101", "#00876C", "#6A3D9A", "#A6761D")
    for observation in observation_rows:
        sbid = str(observation["sbid"])
        sbid_prediction_rows = [
            row for row in prediction_rows if str(row["sbid"]) == sbid
        ]
        output_stem = f"TESS_Period_Extrapolation_{sbid}"
        prediction_path = OUTPUT_DIR / f"{output_stem}_Predictions.txt"
        summary_path = OUTPUT_DIR / f"{output_stem}_Result.md"
        figure_path = OUTPUT_DIR / f"{output_stem}.png"

        prediction_fields = tuple(sbid_prediction_rows[0])
        with prediction_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=prediction_fields,
                delimiter="\t",
            )
            writer.writeheader()
            for row in sbid_prediction_rows:
                writer.writerow(
                    {
                        key: f"{value:.17g}" if isinstance(value, float) else value
                        for key, value in row.items()
                    }
                )

        # 单面板横向条形图：彩色底条表示相位轨道，白色斜线遮罩表示误差区间。
        figure, axis = plt.subplots(
            figsize=(10.5, max(4.2, 0.58 * solution_count + 1.8)),
            constrained_layout=True,
        )
        for solution_index, (run, alias) in enumerate(solution_records):
            row = start_by_key[
                (
                    int(run["period_run_id"]),
                    int(alias["alias_rank"]),
                    sbid,
                )
            ]
            y = float(y_positions[solution_index])
            color = run_colors[
                (int(run["period_run_id"]) - 1) % len(run_colors)
            ]
            center = float(row["circular_sample_center"])
            low = float(row["phase_low_absolute"])
            high = float(row["phase_high_absolute"])

            intervals = (
                [(0.0, high), (low, 1.0)]
                if bool(row["wraps_phase_zero"])
                else [(low, high)]
            )
            for left, right in intervals:
                # 浮动条只覆盖实际的 95% 相位区间，不再铺满整个 [0, 1) 轴。
                axis.barh(
                    y,
                    right - left,
                    left=left,
                    height=0.58,
                    color=color,
                    alpha=0.34,
                    edgecolor=color,
                    linewidth=0.8,
                    zorder=2,
                )
                # 白色半透明方块略高于区间条，白色斜线因此更容易辨认。
                axis.barh(
                    y,
                    right - left,
                    left=left,
                    height=0.58 * 1.14,
                    facecolor=(1.0, 1.0, 1.0, 0.14),
                    edgecolor="white",
                    linewidth=0.8,
                    hatch="////",
                    zorder=3,
                )
                # 白色遮罩在浅色背景上边界较弱，叠加极细原色边界保持清晰。
                axis.barh(
                    y,
                    right - left,
                    left=left,
                    height=0.58 * 1.14,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=0.55,
                    zorder=3.1,
                )
            axis.vlines(
                center,
                y - 0.58 / 2,
                y + 0.58 / 2,
                color="#202020",
                linewidth=1.8,
                zorder=4,
            )

        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(-0.8, solution_count - 0.2)
        axis.set_yticks(y_positions, solution_labels)
        axis.invert_yaxis()
        axis.set_ylabel("")
        axis.set_xlabel("ASKAP Start Phase (cycle)")
        add_panel_label(axis, f"{sbid} | Ephemeris phase", fontsize=11)
        axis.grid(axis="x", color="0.86", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.legend(
            handles=[
                Patch(
                    facecolor="#777777",
                    edgecolor="white",
                    alpha=0.85,
                    hatch="////",
                    label=f"{CONFIDENCE_LEVEL:.0%} phase interval",
                ),
                Line2D(
                    [0], [0],
                    color="#202020",
                    linewidth=1.8,
                    label="phase center",
                ),
            ],
            loc="lower right",
            fontsize=8.5,
            framealpha=0.88,
        )
        figure.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

    # 8. 每个 SBID 单独汇总周期解、中心相位和时间关系。
    for observation in observation_rows:
        sbid = str(observation["sbid"])
        sbid_prediction_rows = [
            row for row in prediction_rows if str(row["sbid"]) == sbid
        ]
        summary_path = OUTPUT_DIR / f"TESS_Period_Extrapolation_{sbid}_Result.md"
        figure_path = OUTPUT_DIR / f"TESS_Period_Extrapolation_{sbid}.png"
        prediction_path = OUTPUT_DIR / f"TESS_Period_Extrapolation_{sbid}_Predictions.txt"
        summary_lines = [
            f"# {sbid} TESS phase extrapolation",
            "",
            f"- ASKAP input: `{observation['radio_name']}`",
            f"- TESS period runs: `{len(period_runs)}`",
            f"- Competitive solutions (`Delta BIC < 6`): `{solution_count}`",
            f"- Phase interval: central `{CONFIDENCE_LEVEL:.1%}` interval",
            f"- DS time mode: `{DS_TIME_MODE}`",
            (
                f"- Cross-run period reference: `{reference_run['period_run_label']}`"
                if reference_run is not None
                else "- Cross-run period reference: `not available; no unique-alias run`"
            ),
            "",
            "## Period solutions",
            "",
            "| TESS input | Alias | Preferred | Cycle-count offsets | Period (d) | 68% error - (s) | 68% error + (s) | Delta BIC | ΔP vs reference (s) |",
            "|:---|---:|:---:|:---|---:|---:|---:|---:|---:|",
        ]
        for run, alias in solution_records:
            delta_reference = alias["delta_period_to_reference_seconds"]
            summary_lines.append(
                "| {label} | {rank} | {preferred} | {offsets} | "
                "{period:.12f} | {minus:.5f} | {plus:.5f} | {bic:.6f} | {delta} |".format(
                    label=run["period_run_label"],
                    rank=alias["alias_rank"],
                    preferred="yes" if alias["preferred"] else "no",
                    offsets=alias["cycle_count_offsets"],
                    period=float(alias["period_days"]),
                    minus=float(alias["period_error_minus_seconds"]),
                    plus=float(alias["period_error_plus_seconds"]),
                    bic=float(alias["delta_bic"]),
                    delta=(
                        f"{float(delta_reference):+.5f}"
                        if delta_reference is not None
                        else "n/a"
                    ),
                )
            )
        summary_lines.extend(
            [
                "",
                "不同 TESS 数据组合中的 alias rank 只在本组合内部有效；不同 alias 的周期和相位区间保持为离散分支，不合并平均。",
                "",
                "## ASKAP start-time prediction",
                "",
                "| TESS input | Alias | Time relation | Start phase | 95% phase interval | Crosses phase 0 | Full width (cycle) | <=0.25 | <=0.50 |",
                "|:---|---:|:---|---:|:---|:---:|---:|:---:|:---:|",
            ]
        )
        for row in sbid_prediction_rows:
            low = float(row["phase_low_absolute"])
            high = float(row["phase_high_absolute"])
            if bool(row["wraps_phase_zero"]):
                interval_text = f"[{low:.6f},1.000000] ∪ [0.000000,{high:.6f}]"
            else:
                interval_text = f"[{low:.6f},{high:.6f}]"
            summary_lines.append(
                "| {label} | {rank} | {relation} | {center:.6f} | {interval} | "
                "{wraps} | {width:.6f} | {quarter} | {half} |".format(
                    label=row["period_run_label"],
                    rank=row["alias_rank"],
                    relation=row["time_relation"],
                    center=float(row["circular_sample_center"]),
                    interval=interval_text,
                    wraps="yes" if row["wraps_phase_zero"] else "no",
                    width=float(row["phase_width_cycles"]),
                    quarter="yes" if row["quarter_cycle_pass"] else "no",
                    half="yes" if row["half_cycle_pass"] else "no",
                )
            )
        summary_lines.extend(
            [
                "",
                "## Interpretation boundary",
                "",
                "S69-only 没有跨扇区整数周数 alias，但短基线仍可能造成远期相位未局域；没有 alias 不等于周期误差很小。",
                "S29+S69 是射电观测前的前向外推，必须保留其全部竞争 alias；这些 alias 的中心相位差异是离散星历歧义，不属于 Bootstrap 误差。",
                "四扇区结果如果覆盖 ASKAP 观测之后的 TESS 数据，只能作为回顾性基线内比较，不能改写成观测前预测。",
                "若相位全宽超过 0.50 cycle，中心相位只作完整记录，不能解释为有效局域的预测相位。",
                "",
                "## Output files",
                "",
                f"- `{figure_path.name}`",
                f"- `{summary_path.name}`",
                f"- `{prediction_path.name}`",
            ]
        )
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        successful_sbids.append(sbid)
        print(
            f"{sbid}: {solution_count} 个周期解；"
            f"PNG/MD/TXT 已保存到 {OUTPUT_DIR.resolve()}"
        )

    print(
        f"外推完成：周期组合={len(period_runs)}；竞争解={solution_count}；"
        f"SBID={len(successful_sbids)}；输出={OUTPUT_DIR.resolve()}"
    )
    if any(len(run["aliases"]) > 1 for run in period_runs):
        print("注意：多 alias 结果保持为离散分支，不能合并解释为唯一星历。")


if __name__ == "__main__":
    main()
