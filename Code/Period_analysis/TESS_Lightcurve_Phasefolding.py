"""Fold configured TESS light curves, estimate their period, and save diagnostic plots."""

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import lightkurve as lk
from astropy.io import fits
from matplotlib.lines import Line2D

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from science_utils import add_panel_label  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

# ========================
DEFAULT_DATA_DIR = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/TESS_Data/2MASS_J01033563-5515561_A"
OUTPUT_BASE = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/TESS_Lightcurve_Phasefolding"

# 周期分析
USE_MANUAL_PERIOD = False   # True: 用手动指定周期; False: Lomb-Scargle 自动搜索
MANUAL_PERIOD = 1.06        # 手动周期（天）
PERIOD_MIN = 0.05           # LS 周期搜索下限（天）
PERIOD_MAX = 20             # LS 周期搜索上限（天），自动限制 ≤ 基线/2
USE_FLATTEN_FOR_LS = True   # True: LS 用去趋势数据; False: 用原始归一化数据
FLATTEN_WINDOW = 721        # flatten 窗口长度上限（采样点数，~24h for 2-min cadence）


def main() -> None:
    parser = argparse.ArgumentParser(description="批量绘制 TESS 光变曲线和周期图")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="TESS 数据目录路径")
    parser.add_argument("--output-dir", default="", help="图片输出目录（默认: OUTPUT_BASE/源名）")
    parser.add_argument("--tic-id", help="只绘制指定 TIC ID 的文件")
    parser.add_argument("--no-skip-foreign", action="store_true", help="不跳过非目标 TIC ID 的文件")
    args = parser.parse_args()

    data_dir = args.data_dir
    source_name = os.path.basename(data_dir.rstrip("/"))
    target_name = source_name.replace("_", " ")
    output_dir = args.output_dir or os.path.join(OUTPUT_BASE, source_name)
    if not os.path.isdir(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return

    print(f"Target: {target_name}")
    print(f"Data:   {data_dir}")
    print(f"Output: {output_dir}\n")
    fits_files = sorted(
        str(path) for path in Path(data_dir).rglob("*.fits")
        if not path.name.startswith("._")
    )
    print(f"找到 {len(fits_files)} 个 FITS 文件")
    if not fits_files:
        return

    # TIC ID 统计与过滤；缓存解析结果，保持原有文件选择规则。
    from collections import Counter
    tic_counter = Counter()
    tic_by_file = {}
    for fits_path in fits_files:
        basename = os.path.basename(fits_path)
        tic_id = None
        match = re.search(r"(\d{16})", basename)
        if match:
            tic_id = str(int(match.group(1)))
        else:
            match = re.search(r"tic(\d+)", basename, re.IGNORECASE)
            if match:
                tic_id = str(int(match.group(1)))
        if tic_id is None:
            try:
                with fits.open(fits_path, memmap=True) as hdul:
                    for hdu in hdul:
                        match = re.search(r"TIC\s*(\d+)", str(hdu.header.get("OBJECT", "")), re.IGNORECASE)
                        if match:
                            tic_id = str(int(match.group(1)))
                            break
            except Exception:
                pass
        tic_by_file[fits_path] = tic_id
        tic_counter[tic_id] += 1
    tic_counter.pop(None, None)

    print("  TIC ID 分布:")
    most_common_tic = tic_counter.most_common(1)
    for tic, count in tic_counter.most_common():
        marker = " ←" if most_common_tic and count == most_common_tic[0][1] else ""
        print(f"    TIC {tic}: {count} 个{marker}")

    filter_tic = args.tic_id
    if filter_tic is None and not args.no_skip_foreign and tic_counter:
        filter_tic = tic_counter.most_common(1)[0][0]
    if filter_tic and not args.no_skip_foreign:
        filtered = []
        skipped = 0
        for fits_path in fits_files:
            file_tic = tic_by_file[fits_path]
            if file_tic is None or file_tic == filter_tic:
                filtered.append(fits_path)
            else:
                print(f"  跳过非目标: {os.path.basename(fits_path)[:70]}  [TIC {file_tic}]")
                skipped += 1
        if skipped:
            print(f"  已跳过 {skipped} 个非目标文件（目标 TIC {filter_tic}）\n")
        fits_files = filtered

    # FITS 产品类型只在本脚本的统计和处理循环中使用；在这里一次解析并缓存，
    # 不把只有两个调用点的识别函数放入公共工具模块。
    type_by_file = {}
    for fits_path in fits_files:
        basename = os.path.basename(fits_path)
        path_upper = str(fits_path).upper()
        if "HLSP" in path_upper:
            data_type = "HLSP"
        elif basename.endswith("_lc.fits") or "_lc." in basename:
            data_type = "LC"
        elif basename.endswith("_tp.fits") or "_tp." in basename:
            data_type = "TPF"
        else:
            data_type = "FITS"
        type_by_file[fits_path] = data_type
    type_counts = Counter(type_by_file.values())
    print()
    for dtype, count in sorted(type_counts.items()):
        print(f"  {dtype}: {count} 个")
    print()

    group_index = {}
    succeeded = 0
    for fits_path in fits_files:
        rel_path = os.path.relpath(fits_path, data_dir)
        data_type = type_by_file[fits_path]

        # 先从 FITS 头读取 Sector，再按原规则从路径回退。
        sector = None
        try:
            with fits.open(fits_path, memmap=True) as hdul:
                for hdu in hdul:
                    header_sector = hdu.header.get("SECTOR")
                    if header_sector is not None:
                        sector = int(header_sector)
                        break
        except Exception:
            pass
        if sector is None or sector == 0:
            for part in fits_path.split(os.sep):
                if part.startswith("Sector_"):
                    try:
                        path_sector = int(part.replace("Sector_", ""))
                        if path_sector != 0:
                            sector = path_sector
                            break
                    except ValueError:
                        pass

        file_index = group_index.get((sector, data_type), 0)
        group_index[(sector, data_type)] = file_index + 1
        print(f"[{file_index}] {data_type}: {rel_path}")

        try:
            # 仅局部抑制已知的 astropy RuntimeWarning，不影响其他质量提示。
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="astropy")
                data = lk.read(fits_path)
                lc = data.to_lightcurve(aperture_mask="pipeline") if hasattr(data, "to_lightcurve") else data
        except Exception as exc:
            print(f"    无法读取，跳过: {exc}")
            continue

        try:
            lc_clean = lc.remove_nans().remove_outliers(sigma=5)
            time_btjd = lc_clean.time.value
            flux_median = float(np.median(lc_clean.flux.value))
            lc_norm = lc_clean / flux_median
            flux = lc_norm.flux.value
            if len(time_btjd) < 10:
                print(f"    数据点太少 ({len(time_btjd)}), 跳过")
                continue

            period_max = min(PERIOD_MAX, (time_btjd[-1] - time_btjd[0]) / 2)
            sector_str = f"Sector {sector}" if sector else rel_path[:50]
            fig = plt.figure(figsize=(12, 10), facecolor="white")
            # 三种横轴单位不同，因此只消除行间距，不使用 sharex。
            gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1, 1.2], hspace=0)
            fig.suptitle(f"{target_name} | {sector_str}", fontsize=14, fontweight="bold", y=0.995)

            # ==================== 光变曲线 ====================
            ax_lc = fig.add_subplot(gs[0])
            ax_lc.scatter(time_btjd, flux, s=4, c="#4a8fd4", alpha=0.55, rasterized=True, linewidths=0)
            trend_label = ""
            try:
                n_data = len(lc_norm)
                window = min(FLATTEN_WINDOW, n_data // 5 * 2 + 1)
                window = max(window, 101)
                if window % 2 == 0:
                    window -= 1
                lc_flat, trend_lc = lc_norm.flatten(window_length=window, return_trend=True)
                ax_lc.plot(trend_lc.time.value, trend_lc.flux.value, color="#e74c3c", linewidth=1.4, alpha=0.9)
                trend_label = " (spline trend)"
            except Exception:
                print("     flatten 失败，回退到原始归一化数据")
                lc_flat = lc_norm
            lc_analysis = lc_flat if USE_FLATTEN_FOR_LS else lc_norm
            flux_phase = lc_analysis.flux.value
            ax_lc.set_ylabel("Normalized Flux", fontsize=13, color="#333333")
            ax_lc.set_xlabel("BTJD  (BJD - 2457000)  [days]", fontsize=13, color="#333333")
            ax_lc.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, color="#aaaaaa")
            ax_lc.set_axisbelow(True)
            for spine in ax_lc.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#cccccc")
            add_panel_label(ax_lc, "Lightcurve", fontsize=11)
            legend_elements = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#4a8fd4", markersize=6, alpha=0.7, label="Raw flux")]
            if trend_label:
                legend_elements.append(Line2D([0], [0], color="#e74c3c", linewidth=1.4, label="Spline trend"))
            ax_lc.legend(handles=legend_elements, fontsize=10, loc="upper right", framealpha=0.8, edgecolor="#dddddd")
            ax_lc.xaxis.set_major_locator(ticker.MaxNLocator(8))
            ax_lc.yaxis.set_major_locator(ticker.MaxNLocator(5))
            ax_lc.tick_params(labelsize=10, colors="#555555")

            # ==================== Lomb--Scargle 周期图 ====================
            ax_pg = fig.add_subplot(gs[1])
            peak_period = None
            peak_amplitude = None
            try:
                pg = lc_analysis.to_periodogram(minimum_period=PERIOD_MIN, maximum_period=period_max, oversample_factor=10, normalization="amplitude")
                periods = pg.period.value
                power = pg.power.value
                if USE_MANUAL_PERIOD:
                    peak_period = MANUAL_PERIOD
                    peak_amplitude = power[np.abs(periods - peak_period).argmin()]
                else:
                    peak_period = pg.period_at_max_power.value
                    peak_amplitude = pg.max_power.value
                ax_pg.plot(periods, power, color="#2980b9", linewidth=0.8, alpha=0.9)
                ax_pg.fill_between(periods, 0, power, color="#2980b9", alpha=0.08)
                ax_pg.set_xscale("log")
                ax_pg.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
                ax_pg.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))
                ax_pg.xaxis.set_minor_formatter(ticker.NullFormatter())
                if not USE_MANUAL_PERIOD:
                    for fap_level, color, label in [(0.01, "#e74c3c", "FAP 1%"), (0.05, "#e67e22", "FAP 5%"), (0.10, "#f1c40f", "FAP 10%")]:
                        try:
                            amp_level = pg.false_alarm_level(fap_level)
                            ax_pg.axhline(amp_level, color=color, linestyle="--", linewidth=0.8, alpha=0.7)
                            ax_pg.text(periods[-1] * 0.95, amp_level, f"  {label}", fontsize=8, color=color, va="bottom", alpha=0.85)
                        except Exception:
                            pass
                ax_pg.axvline(peak_period, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.8)
                ax_pg.axvline(peak_period, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.8)
                label_text = f"  Manual: {peak_period:.4f} d" if USE_MANUAL_PERIOD else f"  Peak: {peak_period:.4f} d"
                ax_pg.annotate(label_text, xy=(peak_period, peak_amplitude), xytext=(peak_period * 1.5, peak_amplitude * 0.88), fontsize=11, color="#e74c3c", fontweight="bold", arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=0.8, connectionstyle="arc3,rad=0.2"))
                ylim = ax_pg.get_ylim()
                for mult, alias_label in [(0.5, "1/2"), (2, "2x"), (3, "3x")]:
                    alias_p = peak_period * mult
                    if periods.min() < alias_p < periods.max():
                        ax_pg.axvline(alias_p, color="#999999", linestyle=":", linewidth=0.6, alpha=0.5)
                        ax_pg.text(alias_p, ylim[1] * 0.92, alias_label, fontsize=9, color="#999999", ha="center")
                ax_pg.set_xlim(periods.min(), periods.max())
                ax_pg.set_xlabel("Period  [days]  (log scale)", fontsize=13, color="#333333")
                ax_pg.set_ylabel("LS Power  [normalized flux]", fontsize=13, color="#333333")
                label = "Lomb–Scargle" + (f" | P={peak_period:.4f} d" if peak_period is not None else "")
                add_panel_label(ax_pg, label, fontsize=11)
            except Exception as exc:
                if USE_MANUAL_PERIOD:
                    peak_period = MANUAL_PERIOD
                    print(f"     LS failed, using manual period P={MANUAL_PERIOD:.4f} d")
                else:
                    peak_period = None
                    peak_amplitude = None
                ax_pg.text(0.5, 0.5, f"LS failed\n{exc}", transform=ax_pg.transAxes, ha="center", va="center", fontsize=10, color="#999999")
                ax_pg.set_xticks([])
                ax_pg.set_yticks([])
            ax_pg.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, color="#aaaaaa")
            ax_pg.set_axisbelow(True)
            for spine in ax_pg.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#cccccc")
            ax_pg.yaxis.set_major_locator(ticker.MaxNLocator(5))
            ax_pg.tick_params(labelsize=10, colors="#555555")

            # ==================== 相位折叠 ====================
            ax_pf = fig.add_subplot(gs[2])
            if peak_period is not None and peak_period > 0:
                phase = (time_btjd / peak_period) % 1.0
                ax_pf.scatter(phase, flux_phase, s=4, c="#4a8fd4", alpha=0.25, rasterized=True, linewidths=0)
                ax_pf.scatter(phase + 1.0, flux_phase, s=4, c="#4a8fd4", alpha=0.25, rasterized=True, linewidths=0)
                n_bins = max(20, min(80, int(len(flux_phase) / 15)))
                bins = np.linspace(0, 2, n_bins * 2 + 1)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                bin_medians = np.full(len(bin_centers), np.nan)
                bin_errs = np.full(len(bin_centers), np.nan)
                phase_ext = np.concatenate([phase, phase + 1.0])
                flux_ext = np.concatenate([flux_phase, flux_phase])
                for j in range(len(bin_centers)):
                    mask = (phase_ext >= bins[j]) & (phase_ext < bins[j + 1])
                    n = np.sum(mask)
                    if n > 5:
                        bin_medians[j] = np.nanmedian(flux_ext[mask])
                        bin_errs[j] = 1.253 * np.nanstd(flux_ext[mask]) / np.sqrt(n)
                valid = ~np.isnan(bin_medians)
                ax_pf.errorbar(bin_centers[valid], bin_medians[valid], yerr=bin_errs[valid], fmt="o", color="#e74c3c", markersize=3, linewidth=1.2, capsize=2, alpha=0.9, label="Binned median ± SEM")
                ax_pf.set_xlim(0, 2)
                ax_pf.set_xlabel("Cycle phase (mod 2)", fontsize=13, color="#333333")
                ax_pf.set_ylabel("Detrended Flux" if USE_FLATTEN_FOR_LS else "Normalized Flux", fontsize=13, color="#333333")
                add_panel_label(ax_pf, f"Phase fold | P={peak_period:.4f} d", fontsize=11)
                ax_pf.legend(fontsize=10, loc="upper right", framealpha=0.8, edgecolor="#dddddd")
            else:
                ax_pf.text(0.5, 0.5, "No period available, skip phase folding", transform=ax_pf.transAxes, ha="center", va="center", fontsize=10, color="#999999")
                ax_pf.set_xticks([])
                ax_pf.set_yticks([])
            ax_pf.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, color="#aaaaaa")
            ax_pf.set_axisbelow(True)
            for spine in ax_pf.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#cccccc")
            ax_pf.xaxis.set_major_locator(ticker.MaxNLocator(8))
            ax_pf.yaxis.set_major_locator(ticker.MaxNLocator(5))
            ax_pf.tick_params(labelsize=10, colors="#555555")

            os.makedirs(output_dir, exist_ok=True)
            type_suffix = {"LC": "_LC", "TPF": "_TPF", "HLSP": "_HLSP"}.get(data_type, "_FITS")
            base = f"Sector_{sector:02d}{type_suffix}" if sector is not None else f"{data_type}"
            output_path = os.path.join(output_dir, f"{base}_{file_index:02d}_lightcurve.png" if file_index > 0 else f"{base}_lightcurve.png")
            fig.subplots_adjust(top=0.96, bottom=0.06, hspace=0)
            fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
            plt.close(fig)
            print(f"    -> {os.path.basename(output_path)}")
            succeeded += 1
        except Exception as exc:
            print(f"  失败: {exc}")

    print(f"\n完成: {succeeded}/{len(fits_files)} 个文件成功")
    print(f"图片保存在: {output_dir}")


if __name__ == "__main__":
    main()
