import os
import glob
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import lightkurve as lk
from astropy.io import fits

warnings.filterwarnings("ignore", category=RuntimeWarning, module="astropy")

plt.rcParams["axes.unicode_minus"] = False

# ========================
DEFAULT_DATA_DIR = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/TESS_Data/2MASS_J01033563-5515561_A"
OUTPUT_BASE = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/TESS_Lightcurve_PhaseFolding"

# 周期分析
USE_MANUAL_PERIOD = False   # True: 用手动指定周期; False: Lomb-Scargle 自动搜索
MANUAL_PERIOD = 1.06        # 手动周期（天）
PERIOD_MIN = 0.05           # LS 周期搜索下限（天）
PERIOD_MAX = 20             # LS 周期搜索上限（天），自动限制 ≤ 基线/2
USE_FLATTEN_FOR_LS = True   # True: LS 用去趋势数据; False: 用原始归一化数据
FLATTEN_WINDOW = 721        # flatten 窗口长度上限（采样点数，~24h for 2-min cadence）

COLOR_DATA = "#4a8fd4"
COLOR_TREND = "#e74c3c"


def find_fits_files(data_dir):
    pattern = os.path.join(data_dir, "**", "*.fits")
    all_files = glob.glob(pattern, recursive=True)
    return sorted(
        [f for f in all_files if not os.path.basename(f).startswith("._")]
    )


def detect_data_type(fits_path):
    """从文件路径识别数据类型：LC / TPF / HLSP"""
    basename = os.path.basename(fits_path)
    path_upper = fits_path.upper()
    if "HLSP" in path_upper:
        return "HLSP"
    if basename.endswith("_lc.fits") or "_lc." in basename:
        return "LC"
    if basename.endswith("_tp.fits") or "_tp." in basename:
        return "TPF"
    return "FITS"


def read_sector_from_header(fits_path):
    try:
        with fits.open(fits_path, memmap=True) as hdul:
            for hdu in hdul:
                sector = hdu.header.get("SECTOR")
                if sector is not None:
                    return int(sector)
        return None
    except Exception:
        return None


def extract_tic_from_file(fits_path):
    """从文件名或 FITS 头提取 TIC ID"""
    import re
    basename = os.path.basename(fits_path)
    # 16 位补零格式
    m = re.search(r"(\d{16})", basename)
    if m:
        return str(int(m.group(1)))
    # TASOC 格式: tic00206502540
    m = re.search(r"tic(\d+)", basename, re.IGNORECASE)
    if m:
        return str(int(m.group(1)))
    # FITS 头 OBJECT 字段
    try:
        with fits.open(fits_path, memmap=True) as hdul:
            for hdu in hdul:
                obj = hdu.header.get("OBJECT", "")
                m = re.search(r"TIC\s*(\d+)", str(obj), re.IGNORECASE)
                if m:
                    return str(int(m.group(1)))
    except Exception:
        pass
    return None


def load_lightcurve(fits_path):
    """读取 FITS 文件为 LightCurve 对象。TPF 自动做孔径测光。"""
    data = lk.read(fits_path)
    if hasattr(data, "to_lightcurve"):
        lc = data.to_lightcurve(aperture_mask="pipeline")
    else:
        lc = data
    return lc


def extract_sector(fits_path):
    sector = read_sector_from_header(fits_path)
    if sector is not None and sector != 0:
        return sector
    for part in fits_path.split(os.sep):
        if part.startswith("Sector_"):
            num = part.replace("Sector_", "")
            try:
                s = int(num)
                if s != 0:
                    return s
            except ValueError:
                pass
    return None


def build_output_name(output_dir, sector, data_type, index):
    """生成输出文件名，同 Sector 同类型多文件时自动编号"""
    type_suffix = {"LC": "_LC", "TPF": "_TPF", "HLSP": "_HLSP"}.get(data_type, "_FITS")
    if sector is not None:
        base = f"Sector_{sector:02d}{type_suffix}"
    else:
        base = f"{data_type}"
    if index > 0:
        return os.path.join(output_dir, f"{base}_{index:02d}_lightcurve.png")
    return os.path.join(output_dir, f"{base}_lightcurve.png")


def plot_one(fits_path, target_name, output_dir, data_dir, file_index):
    rel_path = os.path.relpath(fits_path, data_dir)
    data_type = detect_data_type(fits_path)
    print(f"[{file_index}] {data_type}: {rel_path}")

    # 读取、清洗、归一化
    try:
        lc = load_lightcurve(fits_path)
    except Exception as e:
        print(f"    无法读取，跳过: {e}")
        return None

    # 从原始 LC（清洗前）取观测时间范围（lc.time 已是 astropy Time，无需手动转换）
    btjd_start = lc.time[0].value
    btjd_end = lc.time[-1].value
    date_label = (
        f"BTJD {btjd_start:.3f}–{btjd_end:.3f}    "
        f"{lc.time[0].utc.strftime('%Y-%m-%d %H:%M')} — "
        f"{lc.time[-1].utc.strftime('%Y-%m-%d %H:%M')} UTC"
    )

    lc_clean = lc.remove_nans().remove_outliers(sigma=5)
    time_btjd = lc_clean.time.value
    flux_median = float(np.median(lc_clean.flux.value))
    lc_norm = lc_clean / flux_median
    flux = lc_norm.flux.value

    if len(time_btjd) < 10:
        print(f"    数据点太少 ({len(time_btjd)}), 跳过")
        return None

    # 自动限制周期搜索上限（不超过基线一半）
    baseline = time_btjd[-1] - time_btjd[0]
    period_max = min(PERIOD_MAX, baseline / 2)

    sector = extract_sector(fits_path)
    sector_str = f"Sector {sector}" if sector else rel_path[:50]

    # 画布（3 行：光变曲线 / 周期图 / 相位折叠）
    fig = plt.figure(figsize=(12, 10), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1, 1.2], hspace=0.35)

    # =================================================================
    # 上: 光变曲线 + 趋势拟合
    # =================================================================
    ax_lc = fig.add_subplot(gs[0])

    ax_lc.scatter(time_btjd, flux, s=4, c=COLOR_DATA, alpha=0.55,
                  rasterized=True, linewidths=0)

    trend_label = ""
    try:
        # 自适应窗口长度，避免超过数据点数
        n_data = len(lc_norm)
        window = min(FLATTEN_WINDOW, n_data // 5 * 2 + 1)
        window = max(window, 101)
        if window % 2 == 0:
            window -= 1
        lc_flat, trend_lc = lc_norm.flatten(
            window_length=window, return_trend=True
        )
        ax_lc.plot(trend_lc.time.value, trend_lc.flux.value,
                   color=COLOR_TREND, linewidth=1.4, alpha=0.9)
        trend_label = " (spline trend)"
    except Exception:
        print(f"     flatten 失败，回退到原始归一化数据")
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

    ax_lc.set_title(
        f"{target_name}\n"
        f"{sector_str} [{data_type}]    {date_label}",
        fontsize=13, fontweight="bold", color="#2c3e50", loc="left",
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLOR_DATA, markersize=6, alpha=0.7,
               label="Raw flux"),
    ]
    if trend_label:
        legend_elements.append(
            Line2D([0], [0], color=COLOR_TREND, linewidth=1.4,
                   label="Spline trend")
        )
    ax_lc.legend(handles=legend_elements, fontsize=10, loc="upper right",
                 framealpha=0.8, edgecolor="#dddddd")

    ax_lc.xaxis.set_major_locator(ticker.MaxNLocator(8))
    ax_lc.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax_lc.tick_params(labelsize=10, colors="#555555")

    # =================================================================
    # 中: Lomb-Scargle Periodogram
    # =================================================================
    ax_pg = fig.add_subplot(gs[1])

    peak_period = None
    peak_amplitude = None

    try:
        pg = lc_analysis.to_periodogram(
            minimum_period=PERIOD_MIN, maximum_period=period_max,
            oversample_factor=10, normalization="amplitude",
        )
        periods = pg.period.value
        power = pg.power.value

        if USE_MANUAL_PERIOD:
            peak_period = MANUAL_PERIOD
            peak_idx = np.abs(periods - peak_period).argmin()
            peak_amplitude = power[peak_idx]
        else:
            peak_period = pg.period_at_max_power.value
            peak_amplitude = pg.max_power.value

        # 共用绘图
        ax_pg.plot(periods, power, color="#2980b9", linewidth=0.8, alpha=0.9)
        ax_pg.fill_between(periods, 0, power, color="#2980b9", alpha=0.08)
        ax_pg.set_xscale("log")
        ax_pg.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
        ax_pg.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))
        ax_pg.xaxis.set_minor_formatter(ticker.NullFormatter())

        # FAP（仅自动模式有意义）
        if not USE_MANUAL_PERIOD:
            for fap_level, color, label in [
                (0.01, "#e74c3c", "FAP 1%"), (0.05, "#e67e22", "FAP 5%"), (0.10, "#f1c40f", "FAP 10%")
            ]:
                try:
                    amp_level = pg.false_alarm_level(fap_level)
                    ax_pg.axhline(amp_level, color=color, linestyle="--", linewidth=0.8, alpha=0.7)
                    ax_pg.text(periods[-1] * 0.95, amp_level, f"  {label}", fontsize=8, color=color, va="bottom", alpha=0.85)
                except Exception:
                    pass

        # 标记峰值
        ax_pg.axvline(peak_period, color=COLOR_TREND, linestyle="--", linewidth=1.2, alpha=0.8)
        label_text = f"  Manual: {peak_period:.4f} d" if USE_MANUAL_PERIOD else f"  Peak: {peak_period:.4f} d"
        ax_pg.annotate(
            label_text,
            xy=(peak_period, peak_amplitude),
            xytext=(peak_period * 1.5, peak_amplitude * 0.88),
            fontsize=11, color=COLOR_TREND, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLOR_TREND, lw=0.8, connectionstyle="arc3,rad=0.2"),
        )

        # Alias 线
        ylim = ax_pg.get_ylim()
        for mult, alias_label in [(0.5, "1/2"), (2, "2x"), (3, "3x")]:
            alias_p = peak_period * mult
            if periods.min() < alias_p < periods.max():
                ax_pg.axvline(alias_p, color="#999999", linestyle=":", linewidth=0.6, alpha=0.5)
                ax_pg.text(alias_p, ylim[1] * 0.92, alias_label, fontsize=9, color="#999999", ha="center")

        ax_pg.set_xlim(periods.min(), periods.max())
        ax_pg.set_xlabel("Period  [days]  (log scale)", fontsize=13, color="#333333")
        ax_pg.set_ylabel("LS Power  [normalized flux]", fontsize=13, color="#333333")
        detrend_tag = " (detrended)" if USE_FLATTEN_FOR_LS else ""
        title = f"Lomb-Scargle{detrend_tag}    Manual P = {MANUAL_PERIOD:.4f} d" if USE_MANUAL_PERIOD else f"Lomb-Scargle{detrend_tag}    P = {peak_period:.4f} d    Amp = {peak_amplitude:.4f}"
        if not USE_MANUAL_PERIOD:
            try:
                fap = pg.false_alarm_probability(peak_amplitude)
                title += f"    FAP = {fap:.2e}"
            except Exception:
                pass
        ax_pg.set_title(title, fontsize=12, color="#555555", loc="left", fontweight="normal")

    except Exception as e:
        if USE_MANUAL_PERIOD:
            peak_period = MANUAL_PERIOD
            print(f"     LS failed, using manual period P={MANUAL_PERIOD:.4f} d")
        else:
            peak_period = None
            peak_amplitude = None
        ax_pg.text(0.5, 0.5, f"LS failed\n{e}", transform=ax_pg.transAxes, ha="center", va="center",
                   fontsize=10, color="#999999")
        ax_pg.set_xticks([])
        ax_pg.set_yticks([])

    ax_pg.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, color="#aaaaaa")
    ax_pg.set_axisbelow(True)
    for spine in ax_pg.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")
    ax_pg.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax_pg.tick_params(labelsize=10, colors="#555555")

    # =================================================================
    # 下: 相位折叠
    # =================================================================
    ax_pf = fig.add_subplot(gs[2])

    if peak_period is not None and peak_period > 0:
        phase = (time_btjd / peak_period) % 1.0
        ax_pf.scatter(phase, flux_phase, s=4, c=COLOR_DATA, alpha=0.25,
                       rasterized=True, linewidths=0)
        ax_pf.scatter(phase + 1.0, flux_phase, s=4, c=COLOR_DATA, alpha=0.25,
                       rasterized=True, linewidths=0)

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
        ax_pf.errorbar(bin_centers[valid], bin_medians[valid],
                       yerr=bin_errs[valid],
                       fmt='o', color=COLOR_TREND, markersize=3,
                       linewidth=1.2, capsize=2, alpha=0.9,
                       label="Binned median ± SEM")

        ax_pf.set_xlim(0, 2)
        ax_pf.set_xlabel("Phase", fontsize=13, color="#333333")
        phase_ylabel = "Detrended Flux" if USE_FLATTEN_FOR_LS else "Normalized Flux"
        ax_pf.set_ylabel(phase_ylabel, fontsize=13, color="#333333")
        ax_pf.set_title(
            f"Phase Folding    P = {peak_period:.6f} d",
            fontsize=12, color="#555555", loc="left", fontweight="normal",
        )
        ax_pf.legend(fontsize=10, loc="upper right",
                     framealpha=0.8, edgecolor="#dddddd")
    else:
        ax_pf.text(0.5, 0.5, "No period available, skip phase folding",
                   transform=ax_pf.transAxes, ha="center", va="center",
                   fontsize=10, color="#999999")
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

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = build_output_name(output_dir, sector, data_type, file_index)
    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"    -> {os.path.basename(output_path)}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="批量绘制 TESS 光变曲线和周期图"
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="TESS 数据目录路径",
    )
    parser.add_argument(
        "--output-dir", default="",
        help="图片输出目录（默认: OUTPUT_BASE/源名）",
    )
    parser.add_argument(
        "--tic-id",
        help="只绘制指定 TIC ID 的文件",
    )
    parser.add_argument(
        "--no-skip-foreign",
        action="store_true",
        help="不跳过非目标 TIC ID 的文件",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    source_name = os.path.basename(data_dir.rstrip("/"))
    target_name = source_name.replace("_", " ")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(OUTPUT_BASE, source_name)

    if not os.path.isdir(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return

    print(f"Target: {target_name}")
    print(f"Data:   {data_dir}")
    print(f"Output: {output_dir}\n")

    fits_files = find_fits_files(data_dir)
    print(f"找到 {len(fits_files)} 个 FITS 文件")

    if not fits_files:
        return

    # TIC ID 统计与过滤
    from collections import Counter
    tic_counter = Counter()
    for f in fits_files:
        tic_counter[extract_tic_from_file(f)] += 1
    tic_counter.pop(None, None)

    print(f"  TIC ID 分布:")
    for tic, count in tic_counter.most_common():
        marker = " ←" if count == tic_counter.most_common(1)[0][1] else ""
        print(f"    TIC {tic}: {count} 个{marker}")

    # 确定目标 TIC ID
    filter_tic = args.tic_id
    if filter_tic is None and not args.no_skip_foreign and tic_counter:
        filter_tic = tic_counter.most_common(1)[0][0]

    if filter_tic and not args.no_skip_foreign:
        filtered = []
        skipped = 0
        for f in fits_files:
            ftic = extract_tic_from_file(f)
            if ftic is None or ftic == filter_tic:
                filtered.append(f)
            else:
                print(f"  跳过非目标: {os.path.basename(f)[:70]}  [TIC {ftic}]")
                skipped += 1
        if skipped > 0:
            print(f"  已跳过 {skipped} 个非目标文件（目标 TIC {filter_tic}）\n")
        fits_files = filtered

    # 按类型统计
    type_counts = Counter(detect_data_type(f) for f in fits_files)
    print()
    for dtype, count in sorted(type_counts.items()):
        print(f"  {dtype}: {count} 个")
    print()

    # 按 Sector + 类型分组，组内编号防重名
    group_index = {}
    succeeded = 0

    for fits_path in fits_files:
        sector = extract_sector(fits_path)
        dtype = detect_data_type(fits_path)
        key = (sector, dtype)
        idx = group_index.get(key, 0)
        group_index[key] = idx + 1

        try:
            result = plot_one(fits_path, target_name, output_dir, data_dir, idx)
            if result:
                succeeded += 1
        except Exception as e:
            print(f"  失败: {e}")

    print(f"\n完成: {succeeded}/{len(fits_files)} 个文件成功")
    print(f"图片保存在: {output_dir}")


if __name__ == "__main__":
    main()
