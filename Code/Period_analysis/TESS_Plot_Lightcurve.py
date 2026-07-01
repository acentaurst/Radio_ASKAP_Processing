import os
import glob
import datetime
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import lightkurve as lk
from astropy.io import fits
from scipy.signal import find_peaks

warnings.filterwarnings("ignore", category=RuntimeWarning, module="astropy")

plt.rcParams["axes.unicode_minus"] = False

# ========================
DEFAULT_DATA_DIR = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/TESS_Data/GJ_896_A"
OUTPUT_BASE = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/TESS_Lightcurve_PhaseFolding"

# 周期分析方法与手动周期
METHOD = "manual"           # "LS": Lomb-Scargle 周期图 / "ACF": 自相关函数 / "manual": 手动指定
MANUAL_PERIOD = 1.06   # METHOD="manual" 时生效（天）
PERIOD_MIN = 0.01      # LS 周期搜索下限 / ACF 最短滞后（天）
PERIOD_MAX = 200       # LS 周期搜索上限 / ACF 最长滞后（天）

COLOR_DATA = "#4a8fd4"
COLOR_TREND = "#e74c3c"


def sanitize_name(name):
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def format_date_obs(date_str):
    try:
        s = date_str.strip()
        dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return f"{dt.year}-{dt.month:02d}-{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}"
    except Exception:
        return date_str


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


def read_header_date(fits_path):
    """从 FITS 头读取观测起始时间，兼容多种关键字和 HDU 位置。
    返回 (date_str, is_approx): date_str 为日期字符串，is_approx 表示是否为 BJD 近似反推。
    """
    date_keys = [
        "DATE-OBS", "DATE_OBS", "DATE-BEG", "DATE_BEG",
        "DATEBEG", "DATE-BEG",
    ]
    try:
        with fits.open(fits_path, memmap=True) as hdul:
            for hdu in hdul:
                for key in date_keys:
                    val = hdu.header.get(key)
                    if val is not None:
                        return str(val), False
            # DATE 字段都不存在时，从 TSTART + BJDREFI 反推（常见于 QLP/TGLC）
            for hdu in hdul:
                tstart = hdu.header.get("TSTART")
                bjdrefi = hdu.header.get("BJDREFI", 2457000)
                if tstart is not None:
                    try:
                        bjd = float(bjdrefi) + float(tstart)
                        jd2000 = 2451544.5
                        dt = datetime.datetime(2000, 1, 1) + datetime.timedelta(days=bjd - jd2000)
                        return f"{dt.year}-{dt.month:02d}-{dt.day:02d}", True
                    except Exception:
                        pass
            return "未知", False
    except Exception:
        return "未知", False


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

    lc_clean = lc.remove_nans().remove_outliers(sigma=5)
    time_btjd = lc_clean.time.value
    flux_median = float(np.median(lc_clean.flux.value))
    lc_norm = lc_clean / flux_median
    flux = lc_norm.flux.value

    if len(time_btjd) < 10:
        print(f"    数据点太少 ({len(time_btjd)}), 跳过")
        return None

    sector = extract_sector(fits_path)
    date_obs, is_approx = read_header_date(fits_path)
    date_label = format_date_obs(date_obs)
    if is_approx:
        date_label = f"~ {date_label}  (BJD approx)"
    sector_str = f"Sector {sector}" if sector else rel_path[:50]

    # 画布（3 行：光变曲线 / 周期图 / 相位折叠）
    fig = plt.figure(figsize=(13, 11), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1, 1.2], hspace=0.35)

    # =================================================================
    # 上: 光变曲线 + 趋势拟合
    # =================================================================
    ax_lc = fig.add_subplot(gs[0])

    ax_lc.scatter(time_btjd, flux, s=4, c=COLOR_DATA, alpha=0.55,
                  rasterized=True, linewidths=0)

    trend_label = ""
    try:
        lc_flat, trend_lc = lc_norm.flatten(
            window_length=101, return_trend=True
        )
        ax_lc.plot(trend_lc.time.value, trend_lc.flux.value,
                   color=COLOR_TREND, linewidth=1.4, alpha=0.9)
        trend_label = " (spline trend)"
    except Exception:
        lc_flat = lc_norm

    ax_lc.set_ylabel("Normalized Flux", fontsize=13, color="#333333")
    ax_lc.set_xlabel("BTJD  (BJD - 2457000)  [days]", fontsize=13, color="#333333")
    ax_lc.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, color="#aaaaaa")
    ax_lc.set_axisbelow(True)

    for spine in ax_lc.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")

    ax_lc.set_title(
        f"{target_name}    {sector_str}  [{data_type}]    "
        f"Obs. start: {date_label}",
        fontsize=14, fontweight="bold", color="#2c3e50", loc="left",
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
    # 中: 周期分析 (LS / ACF / manual)
    # =================================================================
    ax_pg = fig.add_subplot(gs[1])

    peak_period = None
    peak_amplitude = None

    if METHOD == "LS":
        try:
            pg = lc_flat.to_periodogram(
                minimum_period=PERIOD_MIN, maximum_period=PERIOD_MAX,
                oversample_factor=10, normalization="amplitude",
            )

            periods = pg.period.value
            power = pg.power.value
            peak_period = pg.period_at_max_power.value
            peak_amplitude = pg.max_power.value

            ax_pg.plot(periods, power, color="#2980b9", linewidth=0.8, alpha=0.9)
            ax_pg.fill_between(periods, 0, power, color="#2980b9", alpha=0.08)
            ax_pg.set_xscale("log")
            ax_pg.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
            ax_pg.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))
            ax_pg.xaxis.set_minor_formatter(ticker.NullFormatter())

            fap_styles = [
                (0.01, "#e74c3c", "FAP 1%"),
                (0.05, "#e67e22", "FAP 5%"),
                (0.10, "#f1c40f", "FAP 10%"),
            ]
            for fap_level, color, label in fap_styles:
                try:
                    amp_level = pg.false_alarm_level(fap_level)
                    ax_pg.axhline(amp_level, color=color, linestyle="--",
                                  linewidth=0.8, alpha=0.7)
                    ax_pg.text(periods[-1] * 0.95, amp_level,
                               f"  {label}", fontsize=8, color=color,
                               va="bottom", alpha=0.85)
                except Exception:
                    pass

            ax_pg.axvline(peak_period, color=COLOR_TREND, linestyle="--",
                          linewidth=1.2, alpha=0.8)
            ax_pg.annotate(
                f"  Peak: {peak_period:.4f} d",
                xy=(peak_period, peak_amplitude),
                xytext=(peak_period * 1.5, peak_amplitude * 0.88),
                fontsize=11, color=COLOR_TREND, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLOR_TREND,
                                lw=0.8, connectionstyle="arc3,rad=0.2"),
            )

            ylim = ax_pg.get_ylim()
            for mult, label in [(0.5, "1/2"), (2, "2x")]:
                alias_p = peak_period * mult
                if periods.min() < alias_p < periods.max():
                    ax_pg.axvline(alias_p, color="#999999", linestyle=":",
                                  linewidth=0.6, alpha=0.5)
                    ax_pg.text(alias_p, ylim[1] * 0.92, label,
                               fontsize=9, color="#999999", ha="center")

            x_min = max(periods.min(), peak_period * 0.3)
            x_max = min(periods.max(), peak_period * 8)
            ax_pg.set_xlim(x_min, x_max)

            ax_pg.set_xlabel("Period  [days]  (log scale)", fontsize=13, color="#333333")
            ax_pg.set_ylabel("Amplitude  [normalized flux]", fontsize=13, color="#333333")
            ax_pg.set_title(
                f"Lomb-Scargle    Peak: {peak_period:.4f} d    Amp: {peak_amplitude:.4f}",
                fontsize=12, color="#555555", loc="left", fontweight="normal",
            )

        except Exception as e:
            ax_pg.text(0.5, 0.5, f"LS failed\n{e}",
                       transform=ax_pg.transAxes, ha="center", va="center",
                       fontsize=10, color="#999999")
            ax_pg.set_xticks([])
            ax_pg.set_yticks([])

    elif METHOD == "ACF":
        try:
            dt = np.nanmedian(np.diff(time_btjd))
            flux_centered = flux - np.nanmedian(flux)
            flux_filled = np.where(np.isnan(flux_centered), 0.0, flux_centered)
            acf = np.correlate(flux_filled, flux_filled, mode='full')
            acf = acf[len(acf)//2:]
            acf /= acf[0]

            # 找 ACF 峰（只在 [PERIOD_MIN, PERIOD_MAX] 范围内找）
            lag_min = max(0, int(PERIOD_MIN / dt))
            lag_max = min(len(acf), int(PERIOD_MAX / dt) + 1)
            acf_crop = acf[lag_min:lag_max]
            peaks, props = find_peaks(acf_crop, prominence=0.03, distance=3)
            if len(peaks) > 0:
                best_idx = lag_min + peaks[np.argmax(props['prominences'])]
                peak_period = best_idx * dt
                peak_amplitude = acf[best_idx]

            lags = np.arange(len(acf)) * dt
            ax_pg.plot(lags, acf, color="#2980b9", linewidth=0.8, alpha=0.9)
            ax_pg.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
            if peak_period is not None:
                ax_pg.axvline(peak_period, color=COLOR_TREND, linestyle="--",
                              linewidth=1.2, alpha=0.8)
                ax_pg.annotate(
                    f"  P = {peak_period:.4f} d",
                    xy=(peak_period, peak_amplitude),
                    xytext=(peak_period * 1.5, peak_amplitude * 0.88),
                    fontsize=11, color=COLOR_TREND, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=COLOR_TREND,
                                    lw=0.8, connectionstyle="arc3,rad=0.2"),
                )
            ax_pg.set_xlim(0, min(lags[-1], PERIOD_MAX))
            ax_pg.set_xlabel("Lag  [days]", fontsize=13, color="#333333")
            ax_pg.set_ylabel("ACF", fontsize=13, color="#333333")
            ax_pg.set_title(
                f"Autocorrelation    "
                f"P = {peak_period:.4f} d    "
                f"Peak = {peak_amplitude:.3f}" if peak_period else "Autocorrelation",
                fontsize=12, color="#555555", loc="left", fontweight="normal",
            )

        except Exception as e:
            ax_pg.text(0.5, 0.5, f"ACF failed\n{e}",
                       transform=ax_pg.transAxes, ha="center", va="center",
                       fontsize=10, color="#999999")
            ax_pg.set_xticks([])
            ax_pg.set_yticks([])

    else:  # manual
        peak_period = MANUAL_PERIOD
        ax_pg.text(0.5, 0.5, f"Manual P = {MANUAL_PERIOD:.4f} d",
                   transform=ax_pg.transAxes, ha="center", va="center",
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
        phase = ((time_btjd - time_btjd[0]) / peak_period) % 1.0
        ax_pf.scatter(phase, flux, s=4, c=COLOR_DATA, alpha=0.45,
                      rasterized=True, linewidths=0)

        # 二倍相位展示（视觉连续）
        ax_pf.scatter(phase + 1.0, flux, s=4, c=COLOR_DATA, alpha=0.45,
                      rasterized=True, linewidths=0)

        # 分箱中值曲线
        n_bins = 60
        bins = np.linspace(0, 2, n_bins * 2 + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_medians = np.full(len(bin_centers), np.nan)
        phase_ext = np.concatenate([phase, phase + 1.0])
        flux_ext = np.concatenate([flux, flux])
        for j in range(len(bin_centers)):
            mask = (phase_ext >= bins[j]) & (phase_ext < bins[j + 1])
            if np.sum(mask) > 5:
                bin_medians[j] = np.median(flux_ext[mask])
        valid = ~np.isnan(bin_medians)
        ax_pf.plot(bin_centers[valid], bin_medians[valid],
                   color=COLOR_TREND, linewidth=1.6, alpha=0.9,
                   label="Binned median")

        ax_pf.set_xlim(0, 2)
        ax_pf.set_xlabel("Phase", fontsize=13, color="#333333")
        ax_pf.set_ylabel("Normalized Flux", fontsize=13, color="#333333")
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
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
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
    output_dir = os.path.join(output_dir, METHOD)  # 按方法分子目录

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
