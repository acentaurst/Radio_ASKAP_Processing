#!/usr/bin/env python3
"""从已经生成的 ASKAP `.ds` 文件绘制 Stokes I/Q/U/V 动态谱。

本脚本只读取 DStools 生成的 HDF5 `.ds`，不在本地导入或运行 dstools。
它保留原脚本的四个 Stokes 面板和像素长表 CSV，同时遵循本项目
`Code/Dstools_Pipeline` 的配置、源/SBID 分目录输出和独立 colorbar 布局。
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
import sys
import warnings

import h5py
import matplotlib
from astropy.time import Time

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from science_utils import (  # noqa: E402
    add_panel_label,
    as_text,
    rebin,
    rebin2d,
    slice_open_end,
)


# ============================ 参数配置区 ============================
# True：递归处理 DS_INPUT_DIR 下的全部 .ds；False：只处理 SINGLE_DS_FILE。
BATCH_PROCESS = True
DS_INPUT_DIR = Path(
    "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/Ds/"
    "2MASS_J01033563-5515561_A/Flare/"
)
SINGLE_DS_FILE = Path(
    "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/Ds/"
    "2MASS_J01033563-5515561_A/Flare/"
    "2MASS_J01033563-5515561_A_SB68040_beam10.ds"
)

# 输出按源名分目录，与 DS_Plot.py 的输出机制一致；每个 .ds 生成一张 PNG 和一个 CSV。
OUTPUT_BASE = Path(
    "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/DS_Plot_IQUV"
)
CHECK_LOCAL_EXISTS = True

# 时间平均参考 DS_Plot.py：短观测不额外压缩，长观测使用 tavg=6。
# 阈值按原始 .ds time 的小时跨度判断，不受后续补 NaN 扫描间隔影响。
TAVG_THRESHOLD_HOURS = 1.0
TAVG_SHORT = 1
TAVG_LONG = 6
# 频率平均保持 DS_Plot.py 的设置；尾部样本由压缩矩阵按部分权重覆盖。
DSTOOLS_FAVG = 5
INSERT_SCAN_GAPS = True
CORRELATOR_DUMPTIME_SECONDS = 10.1

# 当前项目 `.ds` 的单位契约。
DS_TIME_MODE = "utc_seconds_since_mjd0"
DS_FREQUENCY_UNIT = "Hz"
DS_FLUX_UNIT = "Jy"

# 四个 Stokes 面板统一使用固定对称色标；CSV 保留色标范围外的有限值。
COLOR_LIMIT_MJY = 5.0
FIGURE_DPI = 300

# =====================================================================


def main() -> None:
    """按配置批量读取 `.ds`，写出 I/Q/U/V 动态谱 PNG 和像素 CSV。"""

    if BATCH_PROCESS:
        if not DS_INPUT_DIR.is_dir():
            raise FileNotFoundError(f"DS 输入目录不存在：{DS_INPUT_DIR}")
        input_files = sorted(
            path
            for path in DS_INPUT_DIR.rglob("*.ds")
            if not path.name.startswith("._")
        )
    else:
        input_files = [SINGLE_DS_FILE]

    if not input_files:
        raise FileNotFoundError(f"没有找到可处理的 .ds 文件：{DS_INPUT_DIR}")
    if DS_TIME_MODE != "utc_seconds_since_mjd0":
        raise ValueError(
            f"当前读取器只验证 DS_TIME_MODE='utc_seconds_since_mjd0'，实际为 {DS_TIME_MODE!r}"
        )
    if DS_FREQUENCY_UNIT != "Hz":
        raise ValueError(
            f"当前读取器只验证 DS_FREQUENCY_UNIT='Hz'，实际为 {DS_FREQUENCY_UNIT!r}"
        )
    if DS_FLUX_UNIT != "Jy":
        raise ValueError(
            f"当前读取器只验证 DS_FLUX_UNIT='Jy'，实际为 {DS_FLUX_UNIT!r}"
        )
    if (
        TAVG_THRESHOLD_HOURS <= 0
        or TAVG_SHORT < 1
        or TAVG_LONG < 1
        or DSTOOLS_FAVG < 1
    ):
        raise ValueError("时间/频率平均参数必须为正数")
    if COLOR_LIMIT_MJY <= 0:
        raise ValueError("COLOR_LIMIT_MJY 必须为正数")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(" ASKAP full-Stokes dynamic-spectrum plotting")
    print(f" 输入文件数：{len(input_files)}")
    print(f" 输出根目录：{OUTPUT_BASE.resolve()}")
    print(
        " 显示压缩："
        f" tavg=auto({TAVG_SHORT}/{TAVG_LONG}, threshold={TAVG_THRESHOLD_HOURS:g} h),"
        f" favg={DSTOOLS_FAVG}; 固定色标=±{COLOR_LIMIT_MJY:g} mJy"
    )
    print("=" * 72)

    for ds_path in input_files:
        ds_path = Path(ds_path).expanduser().resolve()
        print(f"\n处理：{ds_path}")

        filename_match = re.fullmatch(
            r"(?P<source>.+)_(?P<sbid>SB\d+)_(?P<beam>beam[^.]+)\.ds",
            ds_path.name,
            flags=re.IGNORECASE,
        )
        if filename_match is None:
            message = f"无法从文件名解析 source/SBID/beam：{ds_path.name}"
            if BATCH_PROCESS:
                print(f"  [跳过] {message}")
                continue
            raise ValueError(message)
        source_name = filename_match.group("source")
        sbid = filename_match.group("sbid")
        beam_name = filename_match.group("beam")
        output_dir = OUTPUT_BASE / source_name
        output_stem = f"{source_name}_{sbid}_{beam_name}_IQUV_dynamic_spectrum"
        png_path = output_dir / f"{output_stem}.png"
        csv_path = output_dir / f"{output_stem}.csv"

        if CHECK_LOCAL_EXISTS and png_path.is_file() and csv_path.is_file():
            print(f"  [跳过] 输出已存在：{png_path.name}")
            continue
        if not ds_path.is_file():
            message = f"输入文件不存在：{ds_path}"
            if BATCH_PROCESS:
                print(f"  [跳过] {message}")
                continue
            raise FileNotFoundError(message)

        try:
            # 1. 读取并验证 DStools HDF5 数据契约。
            with h5py.File(ds_path, "r") as handle:
                required_datasets = {"time", "frequency", "flux", "uvdist"}
                missing_datasets = required_datasets.difference(handle.keys())
                if missing_datasets:
                    raise KeyError(f".ds 缺少数据集：{sorted(missing_datasets)}")
                raw_time = np.asarray(handle["time"], dtype=float)
                raw_frequency = np.asarray(handle["frequency"], dtype=float)
                raw_flux = np.asarray(handle["flux"])
                raw_uvdist = np.asarray(handle["uvdist"], dtype=float)
                ds_attributes = dict(handle.attrs)

            if raw_flux.dtype.fields and {"r", "i"}.issubset(raw_flux.dtype.fields):
                raw_flux = raw_flux["r"] + 1j * raw_flux["i"]
            raw_flux = np.asarray(raw_flux, dtype=np.complex128)
            if raw_time.ndim != 1 or raw_frequency.ndim != 1:
                raise ValueError(".ds 的 time 和 frequency 必须是一维数组")
            if raw_flux.ndim != 4 or raw_flux.shape[-1] != 4:
                raise ValueError(
                    "当前读取器要求 flux=(baseline,time,frequency,XX/XY/YX/YY)，"
                    f"实际形状为 {raw_flux.shape}"
                )
            if raw_uvdist.ndim != 1 or raw_uvdist.shape[0] != raw_flux.shape[0]:
                raise ValueError(".ds uvdist 轴与 baseline 轴不一致")
            if raw_flux.shape[1] != len(raw_time) or raw_flux.shape[2] != len(raw_frequency):
                raise ValueError(
                    "flux 的 time/frequency 轴与坐标长度不一致："
                    f" flux={raw_flux.shape}, time={len(raw_time)},"
                    f" frequency={len(raw_frequency)}"
                )
            if len(raw_time) < 2 or not np.all(np.isfinite(raw_time)):
                raise ValueError(".ds time 必须至少有两个有限时间点")
            raw_duration_hours = (raw_time[-1] - raw_time[0]) / 3600.0
            if raw_duration_hours < 0.0:
                raise ValueError(".ds time 的观测跨度不能为负")
            time_average = (
                TAVG_SHORT
                if raw_duration_hours < TAVG_THRESHOLD_HOURS
                else TAVG_LONG
            )
            print(
                f"  Duration: {raw_duration_hours:.2f} h"
                f" -> tavg={time_average}, favg={DSTOOLS_FAVG}"
            )

            feeds = as_text(ds_attributes.get("feeds", "")).lower()
            telescope = as_text(ds_attributes.get("telescope", ""))
            if telescope and telescope.upper() != "ASKAP":
                raise ValueError(f"当前脚本只验证 ASKAP，实际 telescope={telescope!r}")
            if feeds != "linear":
                raise ValueError(
                    "全 Stokes 转换目前仅验证 linear feeds；"
                    f"当前文件 feeds={feeds!r}"
                )

            # 2. 平均 baseline，并按 XX、XY、YX、YY 拆分相关量。
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                instrumental = np.nanmean(raw_flux, axis=0)
            xx = instrumental[:, :, 0]
            xy = instrumental[:, :, 1]
            yx = instrumental[:, :, 2]
            yy = instrumental[:, :, 3]

            # 3. 频率转 MHz、升序排列，并同步重排四个相关量。
            frequency_mhz = raw_frequency / 1.0e6
            frequency_order = np.argsort(frequency_mhz)
            frequency_mhz = frequency_mhz[frequency_order]
            xx = xx[:, frequency_order]
            xy = xy[:, frequency_order]
            yx = yx[:, frequency_order]
            yy = yy[:, frequency_order]

            # 4. 与 DStools trim=True 一致：去掉频带两端没有有效相关量的频道。
            all_polarisation_sum = np.nansum(xx + xy + yx + yy, axis=0)
            all_polarisation_sum[
                all_polarisation_sum == 0.0 + 0.0j
            ] = np.nan + 1j * np.nan
            valid_frequency = np.isfinite(all_polarisation_sum)
            if not np.any(valid_frequency):
                raise ValueError(".ds 没有有效频率频道")
            first_channel = int(np.argmax(valid_frequency))
            last_channel = (
                len(valid_frequency)
                if valid_frequency[-1]
                else len(valid_frequency)
                - int(np.argmax(valid_frequency[::-1]))
                - 1
            )
            xx = slice_open_end(xx, 0, 0, first_channel, last_channel)
            xy = slice_open_end(xy, 0, 0, first_channel, last_channel)
            yx = slice_open_end(yx, 0, 0, first_channel, last_channel)
            yy = slice_open_end(yy, 0, 0, first_channel, last_channel)
            frequency_mhz = frequency_mhz[first_channel:last_channel]

            # 5. 检查时间，并在扫描间隔中插入 NaN 行，保持 DStools calscans 语义。
            raw_time_differences = np.diff(raw_time)
            if len(raw_time_differences) == 0 or np.any(raw_time_differences <= 0.0):
                raise ValueError(".ds time 必须严格递增")
            cadence_native = float(np.median(raw_time_differences))
            if cadence_native <= 0.0:
                raise ValueError("无法从 .ds time 得到正的 cadence")

            scan_starts = np.r_[
                0,
                np.where(
                    raw_time_differences > CORRELATOR_DUMPTIME_SECONDS
                )[0] + 1,
            ]
            scan_ends = np.r_[scan_starts[1:] - 1, len(raw_time) - 1]
            time_chunks: list[np.ndarray] = []
            correlation_chunks: list[list[np.ndarray]] = [[], [], [], []]
            inserted_gap_samples = 0
            for scan_index, (start, end) in enumerate(
                zip(scan_starts, scan_ends, strict=True)
            ):
                time_chunks.append(raw_time[start:end + 1])
                for pol_index, array in enumerate((xx, xy, yx, yy)):
                    correlation_chunks[pol_index].append(array[start:end + 1])

                if INSERT_SCAN_GAPS and scan_index < len(scan_starts) - 1:
                    next_start = scan_starts[scan_index + 1]
                    gap_cycles = (
                        raw_time[next_start] - raw_time[end]
                    ) / cadence_native
                    gap_steps = max(0, int(round(gap_cycles)) - 1)
                    if gap_steps > 0:
                        inserted_gap_samples += gap_steps
                        time_chunks.append(
                            raw_time[end]
                            + cadence_native * np.arange(1, gap_steps + 1)
                        )
                        for chunks in correlation_chunks:
                            chunks.append(
                                np.full(
                                    (gap_steps, len(frequency_mhz)),
                                    np.nan + 1j * np.nan,
                                )
                            )

            time_with_gaps = np.concatenate(time_chunks)
            xx, xy, yx, yy = (
                np.vstack(chunks) for chunks in correlation_chunks
            )

            # 6. 使用项目共享的 DStools 兼容压缩矩阵，覆盖尾部样本。
            time_bin_count = len(time_with_gaps) // time_average
            frequency_bin_count = len(frequency_mhz) // DSTOOLS_FAVG
            if time_bin_count < 2 or frequency_bin_count < 2:
                raise ValueError("时间或频率点数不足以执行 DStools 重采样")
            time_compressor = rebin(
                len(time_with_gaps), time_bin_count, axis=0
            )
            frequency_compressor = rebin(
                len(frequency_mhz), frequency_bin_count, axis=1
            )
            if not np.all(np.sum(time_compressor, axis=0) > 0.0):
                raise RuntimeError("时间压缩矩阵没有覆盖全部输入样本")
            if not np.all(np.sum(frequency_compressor, axis=1) > 0.0):
                raise RuntimeError("频率压缩矩阵没有覆盖全部输入频道")

            xx = rebin2d(xx, (time_bin_count, frequency_bin_count))
            xy = rebin2d(xy, (time_bin_count, frequency_bin_count))
            yx = rebin2d(yx, (time_bin_count, frequency_bin_count))
            yy = rebin2d(yy, (time_bin_count, frequency_bin_count))

            # 7. 线性馈源关系恢复全 Stokes；保持现有 V 符号约定。
            stokes_maps = {
                "I": np.real((xx + yy) / 2.0) * 1000.0,
                "Q": np.real((xx - yy) / 2.0) * 1000.0,
                "U": np.real((xy + yx) / 2.0) * 1000.0,
                "V": np.real(1j * (yx - xy) / 2.0) * 1000.0,
            }

            # 8. 构造观测开始后的小时坐标；本图不做 TESS 相位折叠。
            time_mjd_utc = time_with_gaps / 86400.0
            time_hours = (time_mjd_utc - time_mjd_utc[0]) * 24.0
            binned_time_mjd_utc = time_compressor @ time_mjd_utc
            binned_time_hours = time_compressor @ time_hours
            binned_frequency_mhz = frequency_mhz @ frequency_compressor

            # 横轴显示与 DS_Plot.py 一致：CSV 始终保留小时，短观测图轴改用分钟。
            display_time = binned_time_hours.copy()
            display_duration_hours = float(
                binned_time_hours[-1] - binned_time_hours[0]
            )
            # `.ds` 没有 time_start 属性；按 DS_TIME_MODE 从原始绝对 UTC 秒恢复。
            # 这与 DStools 将 time 转成 MJD UTC 后写入 header.time_start 的结果一致。
            if DS_TIME_MODE == "utc_seconds_since_mjd0":
                observation_start = Time(
                    raw_time[0] / 86400.0,
                    format="mjd",
                    scale="utc",
                ).iso
            else:
                observation_start = as_text(
                    ds_attributes.get("time_start", "observation start")
                )
            if display_duration_hours < 1.0:
                display_time *= 60.0
                major_step = 1.0
                time_label = f"Time (minutes since {observation_start})"
            elif display_duration_hours < 5.0:
                major_step = 0.5
                time_label = f"Time (hours since {observation_start})"
            else:
                major_step = 1.0
                time_label = f"Time (hours since {observation_start})"
            x_major_locator = ticker.MultipleLocator(major_step)
            x_minor_locator = ticker.AutoMinorLocator(5)

            # 9. 导出与图中像素一一对应的长表 CSV；色标截断不改写数值。
            csv_columns = [
                "source",
                "sbid",
                "beam",
                "time_bin_index",
                "frequency_bin_index",
                "time_mjd_utc",
                "time_since_start_hour",
                "frequency_mhz",
                "stokes_i_mjy",
                "stokes_q_mjy",
                "stokes_u_mjy",
                "stokes_v_mjy",
                "finite_i",
                "finite_q",
                "finite_u",
                "finite_v",
                "requested_tavg",
                "requested_favg",
                "effective_time_compression",
                "effective_frequency_compression",
                "inserted_gap_samples",
            ]
            effective_time_compression = len(time_with_gaps) / time_bin_count
            effective_frequency_compression = len(frequency_mhz) / frequency_bin_count
            written_rows = 0
            output_dir.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(csv_columns)
                for time_index in range(time_bin_count):
                    for frequency_index in range(frequency_bin_count):
                        row_values = [
                            source_name,
                            sbid,
                            beam_name,
                            time_index,
                            frequency_index,
                            f"{binned_time_mjd_utc[time_index]:.17g}",
                            f"{binned_time_hours[time_index]:.17g}",
                            f"{binned_frequency_mhz[frequency_index]:.17g}",
                        ]
                        for stokes_label in ("I", "Q", "U", "V"):
                            value = stokes_maps[stokes_label][
                                time_index, frequency_index
                            ]
                            row_values.append(
                                f"{value:.17g}" if np.isfinite(value) else ""
                            )
                        for stokes_label in ("I", "Q", "U", "V"):
                            row_values.append(
                                int(
                                    np.isfinite(
                                        stokes_maps[stokes_label][
                                            time_index, frequency_index
                                        ]
                                    )
                                )
                            )
                        row_values.extend(
                            [
                                time_average,
                                DSTOOLS_FAVG,
                                f"{effective_time_compression:.17g}",
                                f"{effective_frequency_compression:.17g}",
                                inserted_gap_samples,
                            ]
                        )
                        writer.writerow(row_values)
                        written_rows += 1

            expected_rows = time_bin_count * frequency_bin_count
            if written_rows != expected_rows:
                raise RuntimeError(
                    f"CSV 行数错误：写入 {written_rows}，预期 {expected_rows}"
                )
            if not np.all(np.diff(binned_time_hours) >= 0.0):
                raise RuntimeError("输出时间坐标不是单调不减")
            if not np.all(np.diff(binned_frequency_mhz) > 0.0):
                raise RuntimeError("输出频率坐标不是严格递增")

            # 10. 画图：四行主数据轴与独立 colorbar 列，面板之间不留纵向空隙。
            time_edges = np.r_[
                display_time[0] - 0.5 * np.diff(display_time[:2]),
                0.5 * (display_time[:-1] + display_time[1:]),
                display_time[-1] + 0.5 * np.diff(display_time[-2:]),
            ]
            frequency_edges = np.r_[
                binned_frequency_mhz[0]
                - 0.5 * np.diff(binned_frequency_mhz[:2]),
                0.5 * (binned_frequency_mhz[:-1] + binned_frequency_mhz[1:]),
                binned_frequency_mhz[-1]
                + 0.5 * np.diff(binned_frequency_mhz[-2:]),
            ]
            colormap = plt.get_cmap("coolwarm").copy()
            colormap.set_bad("white")
            figure = plt.figure(figsize=(15, 12), facecolor="white")
            grid = figure.add_gridspec(
                4,
                2,
                width_ratios=(20, 0.5),
                hspace=0.0,
                wspace=0.08,
            )
            axes = [figure.add_subplot(grid[0, 0])]
            axes.extend(
                figure.add_subplot(grid[index, 0], sharex=axes[0], sharey=axes[0])
                for index in range(1, 4)
            )
            colorbar_axes = [figure.add_subplot(grid[index, 1]) for index in range(4)]
            figure.suptitle(
                f"{source_name} | {sbid} | {beam_name} | IQUV dynamic spectrum",
                fontsize=18,
                fontweight="bold",
                y=0.98,
            )
            for axis, colorbar_axis, stokes_label in zip(
                axes,
                colorbar_axes,
                ("I", "Q", "U", "V"),
                strict=True,
            ):
                image = axis.pcolormesh(
                    time_edges,
                    frequency_edges,
                    stokes_maps[stokes_label].T,
                    shading="auto",
                    cmap=colormap,
                    vmin=-COLOR_LIMIT_MJY,
                    vmax=COLOR_LIMIT_MJY,
                    edgecolors="none",
                    linewidth=0.0,
                    antialiased=False,
                    rasterized=True,
                )
                axis.set_ylabel("Frequency (MHz)", fontsize=11)
                add_panel_label(axis, f"Stokes {stokes_label}", fontsize=12)
                axis.grid(False)
                figure.colorbar(
                    image,
                    cax=colorbar_axis,
                    label=f"Stokes {stokes_label} (mJy)",
                )
                axis.xaxis.set_major_locator(x_major_locator)
                axis.xaxis.set_minor_locator(x_minor_locator)
                axis.tick_params(axis="both", labelsize=10, colors="#555555")
            axes[-1].set_xlabel(time_label, fontsize=12, color="#333333")
            for axis in axes[:-1]:
                axis.tick_params(axis="x", labelbottom=False)
            axes[-1].set_xlim(display_time[0], display_time[-1])
            figure.subplots_adjust(
                left=0.07,
                right=0.96,
                top=0.95,
                bottom=0.03,
                hspace=0.0,
                wspace=0.08,
            )
            figure.savefig(
                png_path,
                dpi=FIGURE_DPI,
                facecolor="white",
                bbox_inches="tight",
            )
            plt.close(figure)

            print(
                f"  [OK] {sbid}：raw={raw_flux.shape}, "
                f"output=({time_bin_count}, {frequency_bin_count}), "
                f"frequency={binned_frequency_mhz[0]:.3f}–"
                f"{binned_frequency_mhz[-1]:.3f} MHz, "
                f"duration={binned_time_hours[-1]:.3f} h, "
                f"inserted_gap_samples={inserted_gap_samples}"
            )
            print(f"       PNG: {png_path}")
            print(f"       CSV: {csv_path}")
        except Exception as error:
            print(f"  [ERROR] {ds_path.name}: {type(error).__name__}: {error}")
            if not BATCH_PROCESS:
                raise


if __name__ == "__main__":
    main()
