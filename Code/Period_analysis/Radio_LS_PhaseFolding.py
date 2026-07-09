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
TESS_PERIOD = 0.1664         # TESS 测定的恒星光学周期 (天)

# 模式 1：手动指定 DS 文件列表
DS_FILES = [
     "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_localbin/Data/Ds/2MASS_J01033563-5515561_A/flare/2MASS_J01033563-5515561_A_SB68040_beam10.ds"
]

# 模式 2：批量扫描目录
BATCH_PROCESS = True
DS_FILES_DIR = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_localbin/Data/Ds/2MASS_J01033563-5515561_A/flare"
SOURCE_FILTER = ""          # 空字符串=全部; 指定 SBID 如 "68040" 则只处理含该 SBID 的 .ds

OUTPUT_BASE = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_localbin/Result/Radio_Period_Verification"

# Lomb-Scargle 周期搜索范围 (天)
PERIOD_MIN = 0.01
PERIOD_MAX = 50.0

# 相位折叠：True=复制 [0,1)→[1,2)；False=真实相位 mod 2 不复制
DOUBLE_PHASE = False


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


def phase_bin_by_epoch(phase, flux, epoch_ids, nbins=40, x_range=(0, 1)):
    """将相位数据分箱并计算平均值与标准误差。"""
    edges = np.linspace(x_range[0], x_range[1], nbins + 1)
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
    # 收集文件
    if BATCH_PROCESS:
        files = sorted(glob.glob(os.path.join(DS_FILES_DIR, "*.ds")))
    else:
        files = list(DS_FILES)
    
    if SOURCE_FILTER:
        files = [f for f in files if SOURCE_FILTER in os.path.basename(f)]
    
    if not files:
        print(f"[ERROR] 未找到匹配的 .ds 文件")
        sys.exit(1)
    
    # 从第一个文件名提取 source_name
    basename0 = os.path.basename(files[0])
    source_match = re.search(r'(.+?)_SB\d+', basename0)
    SOURCE_NAME = source_match.group(1) if source_match else os.path.splitext(basename0)[0]
    MASTER_OUTPUT_DIR = os.path.join(OUTPUT_BASE, SOURCE_NAME)
    os.makedirs(MASTER_OUTPUT_DIR, exist_ok=True)
    
    print(f"[INFO] 源: {SOURCE_NAME} | 文件: {len(files)} 个")

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
        sbid = f"SB{sbid_num}"

        # 提取 beam 编号用于文件命名
        beam_match = re.search(r'beam(\d+)', basename, re.IGNORECASE)
        beam_num = beam_match.group(1) if beam_match else "X"

        try:
            ds = DynamicSpectrum(file)
            t_hours = getattr(ds, "time", None)
            if t_hours is None: continue

            print(f"    ds.time range: [{t_hours[0]:.2f}, {t_hours[-1]:.2f}] h")

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
            freq_axis = 1 if stokes_i_data.shape[0] == t_len else 0
            flux_i = np.nanmean(stokes_i_data, axis=freq_axis)
            flux_v = np.nanmean(stokes_v_data, axis=freq_axis)

            flux_i = np.real(flux_i)
            flux_v = np.real(flux_v)

            # 每 epoch 去基线偏移（保留 flare 结构，消除 calibration 尺度差异）
            flux_i = flux_i - np.nanmedian(flux_i)
            flux_v = flux_v - np.nanmedian(flux_v)

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
    return all_mjd[valid], all_flux_i[valid], all_flux_v[valid], all_sbid_id[valid], sbid_map, beam_map, SOURCE_NAME, MASTER_OUTPUT_DIR


# ==================================
# 4. 主程序流程：周期分析与绘图
# ==================================
def main():
    mjd, flux_i, flux_v, all_sbid_id, sbid_map, beam_map, SOURCE_NAME, MASTER_OUTPUT_DIR = load_and_stitch_long_tracks()

    # 动态构建输出文件名后缀
    unique_sbids = list(sbid_map.keys())
    if len(unique_sbids) == 1:
        single_sbid = unique_sbids[0]
        single_beam = beam_map[single_sbid]
        file_suffix = f"_{single_sbid}_beam{single_beam}"
    else:
        file_suffix = f"_Stitched_{len(unique_sbids)}obs"

    # --- Lomb-Scargle 周期图计算 ---
    t_centered = mjd - np.mean(mjd)
    baseline_days = mjd[-1] - mjd[0]

    # 1. 计算 Stokes V（带符号）
    flux_v_norm = robust_normalize(flux_v)
    ls_v = LombScargle(t_centered, flux_v_norm)
    frequency, power_v = ls_v.autopower(minimum_frequency=1 / PERIOD_MAX, maximum_frequency=1 / PERIOD_MIN,
                                        samples_per_peak=15)
    periods = 1.0 / frequency

    # 2. 计算 Stokes I
    flux_i_norm = robust_normalize(flux_i)
    ls_i = LombScargle(t_centered, flux_i_norm)
    _, power_i = ls_i.autopower(minimum_frequency=1 / PERIOD_MAX, maximum_frequency=1 / PERIOD_MIN, samples_per_peak=15)

    # 3. 计算窗函数 (Window Function)
    window_power = LombScargle(t_centered, np.ones_like(t_centered), fit_mean=False, center_data=False).power(frequency)

    # 分别提取最佳周期
    best_p_v = periods[np.argmax(power_v)]
    best_p_i = periods[np.argmax(power_i)]

    # FAP
    try:
        fap_v = ls_v.false_alarm_probability(np.max(power_v))
    except Exception:
        fap_v = None
    try:
        fap_i = ls_i.false_alarm_probability(np.max(power_i))
    except Exception:
        fap_i = None

    print(f"\n[INFO] === 射电周期拟合结果 ===")
    print(f"   TESS 周期: {TESS_PERIOD:.5f} d")
    print(f"   TESS 半周期  : {TESS_PERIOD / 2.0:.5f} d")
    print(f"   Stokes V: {best_p_v:.5f} d,  FAP={fap_v:.2e}" if fap_v else f"   Stokes V: {best_p_v:.5f} d")
    print(f"   Stokes I: {best_p_i:.5f} d,  FAP={fap_i:.2e}" if fap_i else f"   Stokes I: {best_p_i:.5f} d")
    print(f"   数据基线: {baseline_days:.2f} d, V 覆盖 {baseline_days/best_p_v:.1f} 个周期")
    print(f"=====================================\n")

    # ---------------------------------------------
    # 绘图 1：Lomb-Scargle 周期图
    # ---------------------------------------------
    fig, ax1 = plt.subplots(figsize=(11, 5.5), dpi=150)

    # 多 epoch 时才画窗函数
    if len(unique_sbids) > 1:
        ax2 = ax1.twinx()
        ax2.fill_between(periods, 0, window_power, color="gray", alpha=0.12)
        ax2.plot(periods, window_power, color="gray", linewidth=0.6, alpha=0.3)
        ax2.set_ylabel("Window Function Power", color="gray", fontsize=9)
        ax2.set_ylim(0, max(1.1, np.max(window_power) * 1.2))
        ax2.tick_params(axis='y', labelcolor="gray", labelsize=8)
        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    # 主Y轴：绘制数据功率谱 (置于顶层)
    ax1.set_zorder(10)
    ax1.patch.set_visible(False)

    # 标记参考周期线
    ax1.axvline(x=TESS_PERIOD, color="#cc3333", linestyle="-.", linewidth=1.8, label=f"TESS Period = {TESS_PERIOD:.4f} d")
    ax1.axvline(x=TESS_PERIOD / 2.0, color="#cc3333", linestyle=":", linewidth=1.5,
                label=f"TESS Half Period = {TESS_PERIOD / 2.0:.4f} d")

    ax1.plot(periods, power_i, color="steelblue", linewidth=1.0, alpha=0.6, label="Radio Stokes I")
    ax1.plot(periods, power_v, color="darkorange", linewidth=1.5, label="Radio Stokes V")

    ax1.axvline(x=best_p_i, color="steelblue", linestyle="--", linewidth=1.5, label=f"Stokes I LS Peak = {best_p_i:.4f} d")
    ax1.axvline(x=best_p_v, color="darkorange", linestyle="--", linewidth=1.5, label=f"Stokes V LS Peak = {best_p_v:.4f} d")

    ax1.set_xlim(0.04, 0.5)
    ax1.set_xlabel("Period (Days)", fontweight='bold')
    ax1.set_ylabel("Lomb-Scargle Power (Data)", fontweight='bold')
    ax1.set_title(f"{SOURCE_NAME} {file_suffix} - Optimized Periodogram", fontsize=11)
    ax1.legend(fontsize=9, loc="upper right")

    # 坐标轴刻度与网格设置
    ax1.xaxis.set_major_locator(MultipleLocator(0.05))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

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
        ("TESS Period", TESS_PERIOD, "#cc3333"),
        ("TESS Half-Period", TESS_PERIOD / 2.0, "#cc3333"),
        ("Stokes I LS Peak", best_p_i, "steelblue"),
        ("Stokes V LS Peak", best_p_v, "darkorange")
    ]

    fig, axes = plt.subplots(len(fold_targets), 2, figsize=(14, 4.2 * len(fold_targets)), dpi=150)

    all_phase_ranges = []
    for row, (label, p_val, row_color) in enumerate(fold_targets):
        for col, (flux_data, name, alpha_val) in enumerate(
                [(flux_i, "Stokes I", 0.6), (flux_v, "Stokes V", 1.0)]):
            ax = axes[row, col]

            if p_val <= 0:
                ax.text(0.5, 0.5, "Invalid Period", ha='center', va='center')
                continue

            t_ref = np.min(mjd)
            if DOUBLE_PHASE:
                phase_01 = ((mjd - t_ref) % p_val) / p_val       # [0,1)
                phase_full = np.concatenate([phase_01, phase_01 + 1.0])
                flux_dup = np.concatenate([flux_data, flux_data])
                ax.scatter(phase_full, flux_dup, c=row_color, s=6, alpha=0.25 * alpha_val, edgecolors="none", rasterized=True)
                pc, pb, pe = phase_bin_by_epoch(phase_01, flux_data, all_sbid_id, nbins=30, x_range=(0, 1))
                valid = ~np.isnan(pb)
                if valid.any():
                    ax.errorbar(pc[valid], pb[valid], yerr=pe[valid], fmt="o", color="black", markersize=4.5,
                                linewidth=1.1, capsize=2, zorder=10, label="Binned Avg")
                    ax.errorbar(pc[valid] + 1.0, pb[valid], yerr=pe[valid], fmt="o", color="black",
                                markersize=4.5, linewidth=1.1, capsize=2, zorder=10)
                all_phase_ranges.append((0, 2))
            else:
                phase_full = ((mjd - t_ref) / p_val) % 2.0
                ax.scatter(phase_full, flux_data, c=row_color, s=6, alpha=0.25 * alpha_val, edgecolors="none", rasterized=True)
                pc, pb, pe = phase_bin_by_epoch(phase_full, flux_data, all_sbid_id, nbins=60, x_range=(0, 2))
                valid = ~np.isnan(pb)
                if valid.any():
                    ax.errorbar(pc[valid], pb[valid], yerr=pe[valid], fmt="o", color="black", markersize=4.5,
                                linewidth=1.1, capsize=2, zorder=10, label="Binned Avg")
                all_phase_ranges.append((np.nanmin(phase_full), np.nanmax(phase_full)))

            ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
            ax.set_xlabel("Phase")
            ax.set_ylabel("Flux (mJy)")
            ax.set_title(f"{name} — {label} (P = {p_val:.5f} d)", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8, loc="upper right")

            # 根据数据分布自动调整 Y 轴范围，剔除极端离群值
            lo, hi = np.percentile(flux_data[np.isfinite(flux_data)], [0.5, 99.5])
            span = hi - lo
            ax.set_ylim(lo - 0.2 * span, hi + 0.2 * span)

    # 统一所有面板的 X 轴范围
    if all_phase_ranges:
        global_xmin = np.floor(min(r[0] for r in all_phase_ranges))
        global_xmax = np.ceil(max(r[1] for r in all_phase_ranges))
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlim(global_xmin, global_xmax)

    fig.suptitle(f"{SOURCE_NAME} {file_suffix} | N_points = {len(mjd)}", fontsize=22, fontweight="bold", y=0.99)
    fig.tight_layout()
    out_fold = os.path.join(MASTER_OUTPUT_DIR, f"{SOURCE_NAME}{file_suffix}_Folding.png")
    fig.savefig(out_fold, dpi=150)
    plt.close(fig)

    print(f"\n[INFO] 绘图完成。\n 周期图: {out_ls}\n 折叠图: {out_fold}")


if __name__ == "__main__":
    main()
