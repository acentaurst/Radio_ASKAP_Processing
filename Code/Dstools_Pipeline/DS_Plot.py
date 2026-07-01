import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpecFromSubplotSpec
from astropy.io import fits
from dstools.dynamic_spectrum import DynamicSpectrum


def project_path(relative_path: str) -> str:
    current = os.path.abspath(os.path.dirname(__file__))
    while not (os.path.isdir(os.path.join(current, 'Code')) and
               os.path.isdir(os.path.join(current, 'Processed_Data'))):
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.join(os.getcwd(), relative_path)
        current = parent
    return os.path.join(current, relative_path)


# ==========================================
# 配置区
# ==========================================
# 批量处理开关
BATCH_PROCESS = False # True: 扫描 PIPELINE_RESULTS_DIR / False: 处理 SINGLE_DS_FILE
PIPELINE_RESULTS_DIR = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Result/DS/Proxima_Cen/DS_Results"
SINGLE_DS_FILE = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Result/DS/Proxima_Cen/DS_Results/Proxima_Cen_SB50381_beam33.ds"
MASTER_OUTPUT_DIR = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/Dynamic_Spectrum"  # Mac Local
# MASTER_OUTPUT_DIR = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Result/Dynamic Spectrum"  # Linux Sever

I_LIMIT = 20    # Stokes I 色标上限 (mJy)
V_LIMIT = 20   # Stokes V 色标上限 (mJy)
T_AVG_LC = 1     # 光变曲线时间平均
T_AVG_DS = 1    # 动态谱时间平均
F_AVG = 5        # 频率平均因子

COLOR_I = "black"
COLOR_V = "#e74c3c"

# WSClean MFS FITS 图开关
INCLUDE_WSCLEAN = False  # True: 在底部加入 WSClean MFS FITS 面板
WSCLEAN_BASE = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Result/DS"


def find_wsclean_fits(hostname, sbid, beam, wsclean_base):
    """根据 DS 文件名推断 WSClean MFS FITS 路径"""
    base = f"{hostname}_SB{sbid}_beam{beam}"
    fits_dir = os.path.join(wsclean_base, hostname,
                            f"{base}_workspace",
                            f"wsclean_model_{base}")
    fits_i = os.path.join(fits_dir, "wsclean-MFS-I-model.fits")
    fits_v = os.path.join(fits_dir, "wsclean-MFS-V-model.fits")
    return fits_i, fits_v


def process_ds_file(ds_file, output_dir):
    basename = os.path.basename(ds_file)

    match = re.search(r'(.+)_SB(\d+)_beam(\d+)\.ds$', basename, re.IGNORECASE)
    if match:
        hostname = match.group(1)
        sbid = match.group(2)
        beam = match.group(3)
    else:
        hostname = basename.replace('.ds', '')
        sb_match = re.search(r'SB(\d+)', basename, re.IGNORECASE)
        beam_match = re.search(r'beam(\d+)', basename, re.IGNORECASE)
        sbid = sb_match.group(1) if sb_match else "UNKNOWN"
        beam = beam_match.group(1) if beam_match else "UNKNOWN"

    base_name_str = f"{hostname}_SB{sbid}_beam{beam}"
    source_specific_dir = os.path.join(output_dir, hostname)
    os.makedirs(source_specific_dir, exist_ok=True)

    print("-" * 60)
    print(f"Processing: {basename}")
    print(f"Output: {source_specific_dir}/")
    print(f"Loading data...")

    # 光变曲线数据（细时间分辨率）
    try:
        ds_lc = DynamicSpectrum(ds_path=ds_file, tavg=T_AVG_LC, favg=F_AVG, trim=True)
    except Exception as e:
        print(f"Load failed: {e}")
        return

    t_hours = ds_lc.time
    stokes_i_lc = np.real(ds_lc.data.get("I"))
    stokes_v_lc = np.real(ds_lc.data.get("V"))

    flux_i = np.nanmean(stokes_i_lc, axis=1)
    flux_v = np.nanmean(stokes_v_lc, axis=1)
    flux_i_detrend = flux_i - np.nanmedian(flux_i)
    flux_v_detrend = flux_v - np.nanmedian(flux_v)
    flux_i_err = np.nanstd(stokes_i_lc, axis=1) / np.sqrt(len(ds_lc.freq))
    flux_v_err = np.nanstd(stokes_v_lc, axis=1) / np.sqrt(len(ds_lc.freq))

    # 动态谱数据（粗时间分辨率）
    try:
        ds_map = DynamicSpectrum(ds_path=ds_file, tavg=T_AVG_DS, favg=F_AVG, trim=True)
    except Exception as e:
        print(f"Load failed: {e}")
        return

    freqs = ds_map.freq
    stokes_i_2d = np.real(ds_map.data.get("I"))
    stokes_v_2d = np.real(ds_map.data.get("V"))
    t_map = ds_map.time

    duration = t_hours[-1] - t_hours[0]
    print(f"  LC: {len(t_hours)} samples")
    print(f"  DS: {len(t_map)} time, {len(freqs)} freq, duration={duration:.2f}h")
    print(f"  Freq: {freqs[0]:.1f} - {freqs[-1]:.1f} MHz")

    obs_time_str = ds_map.header.get("time_start", "unknown")
    time_label = f"Time (hours since {obs_time_str})"

    # ==========================================
    # 绘图
    # ==========================================
    if INCLUDE_WSCLEAN:
        n_rows = 5  # (可选，如果你后面没用到n_rows可以不改)
        fig = plt.figure(figsize=(15, 17), facecolor="white")  # 稍微拉长一点画布
        gs = fig.add_gridspec(5, 2, width_ratios=[20, 0.35],
                              height_ratios=[1, 1.2, 1.2, 0.3, 1.5],  # 👈 增加了 0.3 的缓冲层
                              hspace=0, wspace=0.08)
    else:
        n_rows = 3
        fig = plt.figure(figsize=(15, 12), facecolor="white")
        gs = fig.add_gridspec(3, 2, width_ratios=[20, 0.35],
                              height_ratios=[1, 1.2, 1.2],
                              hspace=0, wspace=0.08)

    title_str = f"Source: {hostname}   |   SBID: {sbid}   |   Beam: {beam}"
    fig.suptitle(title_str, fontsize=18, fontweight='bold', y=0.96)
    plt.subplots_adjust(top=0.92)  # 给主标题留出空间，防止和第一张图撞在一起

    # --- Panel 1: Stokes I/V Lightcurves ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(t_hours, flux_i_detrend, yerr=flux_i_err,
                 fmt='-', color=COLOR_I, linewidth=0.8, capsize=2,
                 alpha=0.85, label="Stokes I")
    ax1.errorbar(t_hours, flux_v_detrend, yerr=flux_v_err,
                 fmt='-', color=COLOR_V, linewidth=0.8, capsize=2,
                 alpha=0.85, label="Stokes V")
    ax1.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    ax1.set_xlim(t_map[0], t_map[-1])
    ax1.set_ylabel("Detrended Flux (mJy)", fontsize=13, color="#333333")
    ax1.text(0.02, 0.95, "Lightcurve",
             transform=ax1.transAxes, fontsize=12, fontweight="bold",
             color="white", ha="left", va="top",
             path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    ax1.legend(fontsize=10, loc="upper right", framealpha=0.8, edgecolor="#dddddd")
    ax1.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, color="#aaaaaa")
    ax1.set_axisbelow(True)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(8))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax1.tick_params(axis='x', labelbottom=False)
    for spine in ax1.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")
    ax1.tick_params(labelsize=10, colors="#555555")

    # --- Panel 2: Stokes I Dynamic Spectrum ---
    ax2 = fig.add_subplot(gs[1, 0])
    im_i = ax2.pcolormesh(t_map, freqs, stokes_i_2d.T,
                           cmap="coolwarm", shading="auto", rasterized=True)
    im_i.set_clim(-I_LIMIT, I_LIMIT)
    ax2.set_xlim(t_map[0], t_map[-1])
    ax2.set_ylim(freqs[0], freqs[-1])
    ax2.set_ylabel("Frequency (MHz)", fontsize=12, color="#333333")
    ax2.text(0.02, 0.95, "Stokes I",
             transform=ax2.transAxes, fontsize=12, fontweight="bold",
             color="white", ha="left", va="top",
             path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    cax_i = fig.add_subplot(gs[1, 1])
    fig.colorbar(im_i, cax=cax_i, label="Stokes I (mJy)")
    ax2.tick_params(axis='x', labelbottom=False)
    ax2.tick_params(labelsize=10, colors="#555555")

    # --- Panel 3: Stokes V Dynamic Spectrum ---
    ax3 = fig.add_subplot(gs[2, 0])
    im_v = ax3.pcolormesh(t_map, freqs, stokes_v_2d.T,
                           cmap="coolwarm", shading="auto", rasterized=True)
    im_v.set_clim(-V_LIMIT, V_LIMIT)
    ax3.set_xlim(t_map[0], t_map[-1])
    ax3.set_ylim(freqs[0], freqs[-1])
    ax3.set_ylabel("Frequency (MHz)", fontsize=12, color="#333333")
    ax3.text(0.02, 0.95, "Stokes V",
             transform=ax3.transAxes, fontsize=12, fontweight="bold",
             color="white", ha="left", va="top",
             path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    cax_v = fig.add_subplot(gs[2, 1])
    fig.colorbar(im_v, cax=cax_v, label="Stokes V (mJy)")
    ax3.tick_params(axis='x', labelbottom=True)
    ax3.set_xlabel(time_label, fontsize=12, color="#333333")
    ax3.tick_params(labelsize=10, colors="#555555")

    # --- Panel 4: WSClean MFS FITS ---
    if INCLUDE_WSCLEAN:
        subgs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[4, 0], wspace=0.05)
        ax4 = fig.add_subplot(subgs[0, 0])
        ax5 = fig.add_subplot(subgs[0, 1])

        fits_i_path, fits_v_path = find_wsclean_fits(hostname, sbid, beam, WSCLEAN_BASE)

        for ax, path, label, cmap in [
            (ax4, fits_i_path, "MFS I-model", "magma"),
            (ax5, fits_v_path, "MFS V-model", "RdBu_r"),
        ]:
            if os.path.exists(path):
                with fits.open(path) as hdul:
                    data = np.squeeze(hdul[0].data)
                    vmin, vmax = np.nanpercentile(data, [1, 99])
                    if cmap == "RdBu_r":
                        vlim = max(abs(vmin), abs(vmax))
                        vmin, vmax = -vlim, vlim
                    ax.imshow(data, origin='lower', cmap=cmap,
                              vmin=vmin, vmax=vmax, aspect='auto')
                ax.text(0.02, 0.95, label,
                        transform=ax.transAxes, fontsize=11, fontweight="bold",
                        color="white", ha="left", va="top",
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
            else:
                ax.text(0.5, 0.5, "FITS not found",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="#999999")
            ax.set_xticks([])
            ax.set_yticks([])

    # 保存
    out_path = os.path.join(source_specific_dir, f"{base_name_str}.png")
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches='tight')
    plt.close(fig)
    print(f" -> {out_path}")


def main():
    print("=" * 60)
    print(" ASKAP Combined Plot Pipeline")
    print("=" * 60)

    if BATCH_PROCESS:
        print("Mode: BATCH")
        ds_files = sorted(glob.glob(os.path.join(PIPELINE_RESULTS_DIR, "*.ds")))
        if not ds_files:
            print(f" No .ds files found in: {PIPELINE_RESULTS_DIR}")
            return
        print(f" Found {len(ds_files)} .ds files")
        for ds_file in ds_files:
            process_ds_file(ds_file, MASTER_OUTPUT_DIR)
    else:
        print("Mode: SINGLE")
        if not os.path.exists(SINGLE_DS_FILE):
            print(f" File not found: {SINGLE_DS_FILE}")
            return
        process_ds_file(SINGLE_DS_FILE, MASTER_OUTPUT_DIR)

    print("\n" + "=" * 60 + "\nDone.")


if __name__ == "__main__":
    main()
