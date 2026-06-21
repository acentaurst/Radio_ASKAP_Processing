import os
import sys
import glob
import re
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# 强制在无图形界面的服务器环境下运行
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

from dstools.dynamic_spectrum import DynamicSpectrum

# 自适应路径定位（优先从脚本位置向上找项目根，PyCharm中则用容器挂载目录）
_PROJECT_MOUNT = "/home/dev/projects/ASKAP_Stellar_with_Exoplanet"

def project_path(relative_path: str) -> str:
    current = os.path.abspath(os.path.dirname(__file__))
    while not (os.path.isdir(os.path.join(current, "Code")) and os.path.isdir(os.path.join(current, "Processed_Data"))):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    else:
        return os.path.join(current, relative_path)
    if os.path.isdir(_PROJECT_MOUNT):
        return os.path.join(_PROJECT_MOUNT, relative_path)
    return os.path.join(os.getcwd(), relative_path)


# ==========================================
# 1. 全局核心配置
# ==========================================
SOURCE_NAME = "2MASS_J01033563-5515561_A"
TESS_PERIOD = 0.166  # TESS测定的恒星光学周期 (天)

# 指定需要处理的 SBID 列表 (仅处理列表中的观测块)
TARGET_SBIDS = ['68040']

# DS_FILES_PATTERN = project_path(f"Pipeline_Results/{SOURCE_NAME}/DS_Results/*.ds")  # 原服务器路径
DS_FILES_PATTERN = f"/Volumes/HST/Research/ASKAP_Stellar_with_Planet_localbin/Data/Ds/{SOURCE_NAME}/*.ds"
# MASTER_OUTPUT_DIR = project_path(f"Processed_Data/Radio_Period_Verification/{SOURCE_NAME}")  # 原服务器路径
MASTER_OUTPUT_DIR = f"/Volumes/HST/Research/ASKAP_Stellar_with_Planet_localbin/Result/Period_Verification/{SOURCE_NAME}"
os.makedirs(MASTER_OUTPUT_DIR, exist_ok=True)

# Lomb-Scargle 周期搜索范围 (天)
PERIOD_MIN = 0.01
PERIOD_MAX = 5.0


# ==========================================
# 2. 核心数学与数据辅助函数
# ==========================================
def extract_sbid(filepath: str) -> str:
    """从文件路径中提取 SBID 标识符"""
    m = re.search(r"SB\d+", os.path.basename(filepath), re.IGNORECASE)
    return m.group(0) if m else os.path.basename(filepath)[:20]


def robust_normalize(flux):
    """
    使用 Median 和 MAD (Median Absolute Deviation) 对流量进行鲁棒归一化，
    以降低极端离群值（如耀发）对整体尺度的影响。
    """
    med = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - med))
    return (flux - med) / mad if mad != 0 else flux - med


def phase_bin_by_epoch(phase, flux, epoch_ids, nbins=40):
    """
    将相位数据分箱并计算平均值与标准误差。
    要求每个 bin 内至少有 3 个数据点才进行计算，否则返回 NaN。
    """
    edges = np.linspace(0, 1, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned, errs = np.full(nbins, np.nan), np.full(nbins, np.nan)
    for i in range(nbins):
        mask = (phase >= edges[i]) & (phase < edges[i + 1])
        if mask.sum() >= 3:
            binned[i] = np.nanmean(flux[mask])
            errs[i] = np.nanstd(flux[mask]) / np.sqrt(mask.sum())
    return centers, binned, errs


# ==========================================
# 3. 数据加载、过滤与拼接
# ==========================================
def load_and_stitch_long_tracks():
    """读取动态频谱文件，过滤指定 SBID，提取光变曲线并拼接"""
    files = sorted(glob.glob(DS_FILES_PATTERN))
    if not files:
        print(f"[ERROR] 未找到匹配的 .ds 文件: {DS_FILES_PATTERN}")
        sys.exit(1)

    print(f"[INFO] 目标源: {SOURCE_NAME}")
    print(f"[INFO] 目标 SBID 列表: {TARGET_SBIDS}")

    # 读取 catalogue CSV 以获取各 SBID 的绝对起始时间 (MJD)
    csv_path = project_path('Processed_Data/Catalogue/01.askap_catalogue.csv')
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

        # 若不在目标列表中则跳过
        if TARGET_SBIDS and sbid_num not in TARGET_SBIDS:
            continue

        sbid = f"SB{sbid_num}"

        # 提取 beam 编号用于文件命名
        beam_match = re.search(r'beam(\d+)', basename, re.IGNORECASE)
        beam_num = beam_match.group(1) if beam_match else "X"

        try:
            ds = DynamicSpectrum(file)
            t_hours = getattr(ds, "time", None)
            if t_hours is None: continue

            duration = t_hours[-1] - t_hours[0]
            # 过滤掉观测时长过短的数据块
            if duration < 2.0: continue

            if sbid not in sbid_mjd_map:
                print(f"  [WARN] 跳过 {sbid}: 未在 CSV 目录中找到对应的 MJD 时间")
                continue

            base_mjd = sbid_mjd_map[sbid]
            useful_count += 1

            if sbid not in sbid_map:
                sbid_map[sbid] = len(sbid_map)
                beam_map[sbid] = beam_num

            # 将相对时间(小时)转换为绝对时间(MJD)
            t_days = base_mjd + (np.array(t_hours, dtype=float) / 24.0)

            stokes_i_data, stokes_v_data = ds.data.get("I"), ds.data.get("V")
            if stokes_i_data is None or stokes_v_data is None: continue

            # 沿频率轴求平均，获取宽频光变曲线
            t_len = len(t_hours)
            flux_i = np.nanmean(stokes_i_data, axis=1 if stokes_i_data.shape[0] == t_len else 0)
            flux_v = np.nanmean(stokes_v_data, axis=1 if stokes_v_data.shape[0] == t_len else 0)

            # 简单的去趋势处理：减去中值
            flux_i = np.real(flux_i) - np.nanmedian(np.real(flux_i))
            flux_v = np.real(flux_v) - np.nanmedian(np.real(flux_v))

            all_mjd.extend(t_days)
            all_flux_i.extend(flux_i)
            all_flux_v.extend(flux_v)
            all_sbid_id.extend([sbid_map[sbid]] * t_len)

            print(f"  [OK] 加载 {sbid} | Beam: {beam_num} | 时长: {duration:.2f}h | MJD: {base_mjd:.4f}")

        except Exception as e:
            print(f"  [WARN] 读取 {basename} 失败: {e}")

    if useful_count == 0:
        print("[ERROR] 未找到匹配的目标 SBID 数据，请检查 TARGET_SBIDS 配置。")
        sys.exit(1)

    all_mjd, all_flux_i, all_flux_v = np.array(all_mjd), np.array(all_flux_i), np.array(all_flux_v)
    all_sbid_id = np.array(all_sbid_id, dtype=int)

    # 按时间排序
    sort_idx = np.argsort(all_mjd)
    all_mjd, all_flux_i, all_flux_v, all_sbid_id = all_mjd[sort_idx], all_flux_i[sort_idx], all_flux_v[sort_idx], \
        all_sbid_id[sort_idx]

    # 剔除无效值 (NaN)
    valid = ~np.isnan(all_flux_i) & ~np.isnan(all_flux_v)

    print(f"\n[INFO] 数据拼接完成。共包含 {useful_count} 个观测块，有效积分点: {np.sum(valid)}")
    return all_mjd[valid], all_flux_i[valid], all_flux_v[valid], all_sbid_id[valid], sbid_map, beam_map


# ==================================
# 4. 主程序流程：周期分析与绘图
# ==================================
def main():
    mjd, flux_i, flux_v, all_sbid_id, sbid_map, beam_map = load_and_stitch_long_tracks()

    # 动态构建输出文件名后缀
    unique_sbids = list(sbid_map.keys())
    if len(unique_sbids) == 1:
        single_sbid = unique_sbids[0]
        single_beam = beam_map[single_sbid]
        file_suffix = f"_{single_sbid}_beam{single_beam}"
    else:
        file_suffix = f"_Stitched_{len(unique_sbids)}obs"

    # --- Lomb-Scargle 周期图计算 ---
    # 1. 计算 Stokes V
    flux_v_norm = robust_normalize(flux_v)
    ls_v = LombScargle(mjd, flux_v_norm)
    frequency, power_v = ls_v.autopower(minimum_frequency=1 / PERIOD_MAX, maximum_frequency=1 / PERIOD_MIN,
                                        samples_per_peak=15)
    periods = 1.0 / frequency

    # 2. 计算 Stokes I
    flux_i_norm = robust_normalize(flux_i)
    ls_i = LombScargle(mjd, flux_i_norm)
    _, power_i = ls_i.autopower(minimum_frequency=1 / PERIOD_MAX, maximum_frequency=1 / PERIOD_MIN, samples_per_peak=15)

    # 3. 计算窗函数 (Window Function)
    window_power = LombScargle(mjd, np.ones_like(mjd), fit_mean=False, center_data=False).power(frequency)

    # 分别提取 Stokes I 和 Stokes V 的最佳周期
    best_p_v = periods[np.argmax(power_v)]
    best_p_i = periods[np.argmax(power_i)]

    print(f"\n[INFO] === 射电周期拟合结果 ===")
    print(f"   TESS 周期: {TESS_PERIOD:.5f} d")
    print(f"   TESS 半周期  : {TESS_PERIOD / 2.0:.5f} d")
    print(f"   Stokes V (圆偏振) 最佳周期: {best_p_v:.5f} d")
    print(f"   Stokes I (总流量) 最佳周期: {best_p_i:.5f} d")
    print(f"=====================================\n")

    # ---------------------------------------------
    # 绘图 1：Lomb-Scargle 周期图
    # ---------------------------------------------
    fig, ax1 = plt.subplots(figsize=(11, 5.5), dpi=150)

    # 副Y轴：绘制窗函数 (灰色阴影)
    ax2 = ax1.twinx()
    ax2.fill_between(periods, 0, window_power, color="gray", alpha=0.12)
    ax2.plot(periods, window_power, color="gray", linewidth=0.6, alpha=0.3)
    ax2.set_ylabel("Window Function Power", color="gray", fontsize=9)
    ax2.set_ylim(0, max(1.1, np.max(window_power) * 1.2))
    ax2.tick_params(axis='y', labelcolor="gray", labelsize=8)

    # 主Y轴：绘制数据功率谱 (置于顶层)
    ax1.set_zorder(10)
    ax1.patch.set_visible(False)

    ax1.plot(periods, power_v, color="darkorange", linewidth=1.5, label="Radio Stokes V (Circular)")
    ax1.plot(periods, power_i, color="steelblue", linewidth=1.0, alpha=0.6, label="Radio Stokes I (Total)")

    # 标记参考周期线
    ax1.axvline(x=TESS_PERIOD, color="red", linestyle="-.", linewidth=1.8, label=f"TESS P = {TESS_PERIOD:.4f} d")
    ax1.axvline(x=TESS_PERIOD / 2.0, color="blue", linestyle=":", linewidth=1.5,
                label=f"TESS Half-P = {TESS_PERIOD / 2.0:.4f} d")
    ax1.axvline(x=best_p_v, color="green", linestyle="-", linewidth=1.5, label=f"Stokes V Peak = {best_p_v:.4f} d")
    ax1.axvline(x=best_p_i, color="purple", linestyle="--", linewidth=1.5, label=f"Stokes I Peak = {best_p_i:.4f} d")

    ax1.set_xlim(0.04, 0.5)
    ax1.set_xlabel("Period (Days)", fontweight='bold')
    ax1.set_ylabel("Lomb-Scargle Power (Data)", fontweight='bold')
    ax1.set_title(f"{SOURCE_NAME} {file_suffix} - Optimized Periodogram", fontsize=11)
    ax1.legend(fontsize=9, loc="upper right")

    # 坐标轴刻度与网格设置
    ax1.xaxis.set_major_locator(MultipleLocator(0.05))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax1.grid(True, which='major', color='gray', linestyle='-', alpha=0.3)
    ax1.grid(False, which='minor')

    out_ls = os.path.join(MASTER_OUTPUT_DIR, f"{SOURCE_NAME}{file_suffix}_LS.png")
    fig.tight_layout()
    fig.savefig(out_ls, dpi=150)
    plt.close(fig)

    # ---------------------------------------------
    # 绘图 2：相位折叠图 (Phase Folding)
    # ---------------------------------------------
    # 将折叠目标增加为 4 个，分别验证 I 和 V 的拟合结果
    fold_targets = [
        ("TESS True Period", TESS_PERIOD),
        ("TESS Half-Period", TESS_PERIOD / 2.0),
        ("Stokes I Best Peak", best_p_i),
        ("Stokes V Best Peak", best_p_v)
    ]

    # 图表高度自适应增加 (4.2 * 4 = 16.8)
    fig, axes = plt.subplots(len(fold_targets), 2, figsize=(14, 4.2 * len(fold_targets)), dpi=150)

    for row, (label, p_val) in enumerate(fold_targets):
        for col, (flux_data, name, col_color) in enumerate(
                [(flux_i, "Stokes I", "steelblue"), (flux_v, "Stokes V", "darkorange")]):
            ax = axes[row, col]

            # 如果算出的周期恰好是 0 (异常兜底)，则跳过折叠防止报错
            if p_val <= 0:
                ax.text(0.5, 0.5, "Invalid Period", ha='center', va='center')
                continue

            phase = (mjd % p_val) / p_val

            # 绘制散点 (绘制两周期以观察连续性)
            ax.scatter(phase, flux_data, c=col_color, s=6, alpha=0.25, edgecolors="none", rasterized=True)
            ax.scatter(phase + 1.0, flux_data, c=col_color, s=6, alpha=0.25, edgecolors="none", rasterized=True)

            # 绘制分箱平均线
            pc, pb, pe = phase_bin_by_epoch(phase, flux_data, all_sbid_id, nbins=30)
            valid_bins = ~np.isnan(pb)
            if valid_bins.any():
                ax.errorbar(pc[valid_bins], pb[valid_bins], yerr=pe[valid_bins], fmt="o", color="black", markersize=4.5,
                            linewidth=1.1, capsize=2, zorder=10, label="Binned Avg")
                ax.errorbar(pc[valid_bins] + 1.0, pb[valid_bins], yerr=pe[valid_bins], fmt="o", color="black",
                            markersize=4.5, linewidth=1.1, capsize=2, zorder=10)

            ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
            ax.set_xlim(0, 2)
            ax.set_xlabel("Phase")
            ax.set_ylabel("Detrended Flux (mJy)")
            ax.set_title(f"{name} — folded at {label} (P = {p_val:.5f} d)")
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8, loc="upper right")

            # 根据数据分布自动调整 Y 轴范围，剔除极端离群值
            lo, hi = np.percentile(flux_data[np.isfinite(flux_data)], [0.5, 99.5])
            span = hi - lo
            ax.set_ylim(lo - 0.2 * span, hi + 0.2 * span)

    fig.suptitle(f"{SOURCE_NAME} {file_suffix} | N_points = {len(mjd)}", fontsize=11, y=0.99)
    fig.tight_layout()
    out_fold = os.path.join(MASTER_OUTPUT_DIR, f"{SOURCE_NAME}{file_suffix}_Folding.png")
    fig.savefig(out_fold, dpi=150)
    plt.close(fig)

    print(f"\n[INFO] 绘图完成。\n 周期图: {out_ls}\n 折叠图: {out_fold}")


if __name__ == "__main__":
    main()
