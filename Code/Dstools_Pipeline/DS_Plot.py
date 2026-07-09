import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpecFromSubplotSpec
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
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
BATCH_PROCESS = False
PIPELINE_RESULTS_DIR = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Result/DS/Proxima_Cen/DS_Results"
# SINGLE_DS_FILE = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Result/DS/Proxima_Cen/DS_Results/Proxima_Cen_SB50381_beam33.ds"
SINGLE_DS_FILE = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_localbin/Data/Ds/2MASS_J01033563-5515561_A/flare/2MASS_J01033563-5515561_A_SB59565_beam22.ds"
# MASTER_OUTPUT_DIR = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Result/Dynamic Spectrum"
MASTER_OUTPUT_DIR = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_localbin/Result/Dynamic_Spectrum/2MASS_J01033563-5515561_A"
WSCLEAN_BASE = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Result/DS"

# 科学图参数
I_LIMIT = 5
V_LIMIT = 5
T_AVG_LC = 12
T_AVG_DS = 12
F_AVG = 5
SHOW_BASELINE_REMOVED = False

# QC 图参数
POL_FRAC_LIMIT = 50  # V/I 色标上限 (%)

# MFS 图开关（画好后改成 False 省时间）
INCLUDE_MFS = False    # True: 画 MFS 模型图; False: 跳过
INCLUDE_QC = True     # True: 画 QC 诊断图; False: 跳过

COLOR_I = "black"
COLOR_V = "#e74c3c"

ASKAP_CATALOGUE_CSV = project_path('Processed_Data/Catalogue/01.askap_catalogue.csv')
INPUT_CSV = project_path('Processed_Data/Catalogue/02.final_confirmed_stars_direct_1.csv')


# ==========================================
# 辅助函数
# ==========================================
def find_wsclean_fits(hostname, sbid, beam, wsclean_base):
    base = f"{hostname}_SB{sbid}_beam{beam}"
    fits_dir = os.path.join(wsclean_base, hostname,
                            f"{base}_workspace",
                            f"wsclean_model_{base}")
    return (os.path.join(fits_dir, "wsclean-MFS-I-model.fits"),
            os.path.join(fits_dir, "wsclean-MFS-V-model.fits"))


def get_target_coords(hostname, sbid):
    """从星表自动计算目标坐标（自存修正）。"""
    obs_df = pd.read_csv(ASKAP_CATALOGUE_CSV)
    obs_df.columns = obs_df.columns.str.strip()
    obs_df['sbid_clean'] = obs_df['obs_id'].apply(
        lambda x: str(int(re.search(r'(\d+)', str(x)).group(1))) if re.search(r'(\d+)', str(x)) else None)
    sbid_to_mjd = obs_df.dropna(subset=['sbid_clean']).drop_duplicates(subset=['sbid_clean']).set_index('sbid_clean')['t_min'].to_dict()

    stars_df = pd.read_csv(INPUT_CSV)
    stars_df.columns = stars_df.columns.str.strip()
    stars_df['hostname_clean'] = stars_df['hostname'].astype(str).str.strip().str.replace(' ', '_')
    star_catalog_dict = stars_df.drop_duplicates(subset=['hostname_clean']).set_index('hostname_clean').to_dict('index')

    norm_h = re.sub(r'[^a-zA-Z0-9]', '', hostname).lower()
    matched_key = None
    for k in star_catalog_dict.keys():
        if re.sub(r'[^a-zA-Z0-9]', '', k).lower() in norm_h or norm_h in re.sub(r'[^a-zA-Z0-9]', '', k).lower():
            matched_key = k
            break
    if not matched_key:
        return None, None
    star_meta = star_catalog_dict[matched_key]
    obs_mjd = sbid_to_mjd.get(sbid)
    if not obs_mjd:
        return None, None

    pmra = 0.0 if pd.isna(star_meta.get('sy_pmra', star_meta.get('pmra', 0.0))) else float(star_meta.get('sy_pmra', star_meta.get('pmra', 0.0)))
    pmdec = 0.0 if pd.isna(star_meta.get('sy_pmdec', star_meta.get('pmdec', 0.0))) else float(star_meta.get('sy_pmdec', star_meta.get('pmdec', 0.0)))
    plx_val = star_meta.get('sy_plx', star_meta.get('plx', 10.0))
    plx = 10.0 if pd.isna(plx_val) or float(plx_val) <= 0 else float(plx_val)

    star_j2015 = SkyCoord(ra=star_meta['ra'] * u.deg, dec=star_meta['dec'] * u.deg,
                          pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                          distance=(1000 / plx) * u.pc, frame='icrs', obstime=Time('J2015.5'))
    star_at_obs = star_j2015.apply_space_motion(new_obstime=Time(obs_mjd, format='mjd'))
    return round(star_at_obs.ra.deg, 7), round(star_at_obs.dec.deg, 7)


# ==========================================
# 科学图：光变曲线 + I/V 动态谱
# ==========================================
def plot_science(ds_file, output_dir, hostname, sbid, beam, base_name_str):
    print(f"  [Science] 绘制中...")

    try:
        ds_lc = DynamicSpectrum(ds_path=ds_file, tavg=T_AVG_LC, favg=F_AVG, trim=True)
    except Exception as e:
        print(f"  Load failed: {e}")
        return

    time_lc = ds_lc.time
    stokes_i_lc = np.real(ds_lc.data.get("I"))
    stokes_v_lc = np.real(ds_lc.data.get("V"))

    flux_i = np.nanmean(stokes_i_lc, axis=1)
    flux_v = np.nanmean(stokes_v_lc, axis=1)
    flux_i_err = np.nanstd(stokes_i_lc, axis=1) / np.sqrt(len(ds_lc.freq))
    flux_v_err = np.nanstd(stokes_v_lc, axis=1) / np.sqrt(len(ds_lc.freq))

    if SHOW_BASELINE_REMOVED:
        flux_i_plot = flux_i - np.nanmedian(flux_i)
        flux_v_plot = flux_v - np.nanmedian(flux_v)
    else:
        flux_i_plot = flux_i
        flux_v_plot = flux_v

    try:
        ds_map = DynamicSpectrum(ds_path=ds_file, tavg=T_AVG_DS, favg=F_AVG, trim=True)
    except Exception as e:
        print(f"  Load failed: {e}")
        return

    freqs = ds_map.freq
    stokes_i_2d = np.real(ds_map.data.get("I"))
    stokes_v_2d = np.real(ds_map.data.get("V"))
    time_map = ds_map.time

    duration = time_lc[-1] - time_lc[0]
    print(f"    I: median={np.nanmedian(flux_i):.4f} range=[{np.nanmin(flux_i):.4f}, {np.nanmax(flux_i):.4f}]")
    print(f"    V: median={np.nanmedian(flux_v):.4f} range=[{np.nanmin(flux_v):.4f}, {np.nanmax(flux_v):.4f}]")

    obs_time_str = ds_map.header.get("time_start", "unknown")
    if duration < 1:
        time_lc = time_lc * 60
        time_map = time_map * 60
        major_step = 1
        time_label = f"Time (minutes since {obs_time_str})"
    elif duration < 5:
        major_step = 0.5
        time_label = f"Time (hours since {obs_time_str})"
    else:
        major_step = 1.0
        time_label = f"Time (hours since {obs_time_str})"
    x_major_locator = ticker.MultipleLocator(major_step)
    x_minor_locator = ticker.AutoMinorLocator(5)

    fig = plt.figure(figsize=(15, 12), facecolor="white")
    gs = fig.add_gridspec(3, 2, width_ratios=[20, 0.5],
                          height_ratios=[1, 1.2, 1.2],
                          hspace=0, wspace=0.08)

    title_str = f"Source: {hostname}   |   SBID: {sbid}   |   Beam: {beam}"
    fig.suptitle(title_str, fontsize=18, fontweight='bold', y=0.98)
    fig.subplots_adjust(top=0.95, bottom=0.03)

    # Panel 1: Lightcurve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(time_lc, flux_i_plot, yerr=flux_i_err,
                 fmt='-', color=COLOR_I, linewidth=0.8, capsize=2,
                 alpha=0.85, label="Stokes I")
    ax1.errorbar(time_lc, flux_v_plot, yerr=flux_v_err,
                 fmt='-', color=COLOR_V, linewidth=0.8, capsize=2,
                 alpha=0.85, label="Stokes V")
    ax1.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    ax1.set_xlim(time_map[0], time_map[-1])
    if SHOW_BASELINE_REMOVED:
        ax1.set_ylabel("Baseline-removed Flux (mJy)", fontsize=13, color="#333333")
    else:
        ax1.set_ylabel("Flux (mJy)", fontsize=13, color="#333333")
    ax1.text(0.02, 0.95, "Lightcurve",
             transform=ax1.transAxes, fontsize=12, fontweight="bold",
             color="white", ha="left", va="top",
             path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    ax1.legend(fontsize=10, loc="upper right", framealpha=0.8, edgecolor="#dddddd")
    ax1.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, color="#aaaaaa")
    ax1.set_axisbelow(True)
    ax1.xaxis.set_major_locator(x_major_locator)
    ax1.xaxis.set_minor_locator(x_minor_locator)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax1.tick_params(axis='x', labelbottom=False)
    for spine in ax1.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")
    ax1.tick_params(labelsize=10, colors="#555555")

    # Panel 2: Stokes I DS
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    im_i = ax2.pcolormesh(time_map, freqs, stokes_i_2d.T,
                           cmap="coolwarm", shading="auto", rasterized=True)
    im_i.set_clim(-I_LIMIT, I_LIMIT)
    ax2.set_ylim(freqs[0], freqs[-1])
    ax2.set_ylabel("Frequency (MHz)", fontsize=12, color="#333333")
    ax2.text(0.02, 0.95, "Stokes I",
             transform=ax2.transAxes, fontsize=12, fontweight="bold",
             color="white", ha="left", va="top",
             path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    cax_i = fig.add_subplot(gs[1, 1])
    fig.colorbar(im_i, cax=cax_i, label="Stokes I (mJy)")
    fig.canvas.draw()
    pos_i = cax_i.get_position()
    cax_i.set_position([pos_i.x0, pos_i.y0 + pos_i.height * 0.05, pos_i.width, pos_i.height * 0.95])
    ax2.tick_params(axis='x', labelbottom=False)
    ax2.tick_params(labelsize=10, colors="#555555")

    # Panel 3: Stokes V DS
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    im_v = ax3.pcolormesh(time_map, freqs, stokes_v_2d.T,
                           cmap="coolwarm", shading="auto", rasterized=True)
    im_v.set_clim(-V_LIMIT, V_LIMIT)
    ax3.set_ylim(freqs[0], freqs[-1])
    ax3.set_ylabel("Frequency (MHz)", fontsize=12, color="#333333")
    ax3.text(0.02, 0.95, "Stokes V",
             transform=ax3.transAxes, fontsize=12, fontweight="bold",
             color="white", ha="left", va="top",
             path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    cax_v = fig.add_subplot(gs[2, 1])
    fig.colorbar(im_v, cax=cax_v, label="Stokes V (mJy)")
    fig.canvas.draw()
    pos_v = cax_v.get_position()
    cax_v.set_position([pos_v.x0, pos_v.y0 + pos_v.height * 0.05, pos_v.width, pos_v.height * 0.95])
    ax3.tick_params(axis='x', labelbottom=True)
    ax3.set_xlabel(time_label, fontsize=12, color="#333333")
    ax3.tick_params(labelsize=10, colors="#555555")

    source_specific_dir = os.path.join(output_dir, hostname)
    os.makedirs(source_specific_dir, exist_ok=True)
    out_path = os.path.join(source_specific_dir, f"{base_name_str}.png")
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches='tight')
    plt.close(fig)
    print(f"  [Science] -> {out_path}")


# ==========================================
# MFS 模型图
# ==========================================
def plot_mfs(output_dir, hostname, sbid, beam, base_name_str):
    print(f"  [MFS] 绘制中...")

    fits_i_path, fits_v_path = find_wsclean_fits(hostname, sbid, beam, WSCLEAN_BASE)

    corr_ra, corr_dec = get_target_coords(hostname, sbid)
    if corr_ra is None:
        print(f"  [MFS] 无法获取目标坐标，跳过源位置标记")
        target_coord = None
    else:
        target_coord = SkyCoord(corr_ra * u.deg, corr_dec * u.deg, frame='icrs')
        coord_str = target_coord.to_string('hmsdms', precision=2)

    fig, axes = plt.subplots(2, 1, figsize=(10, 14), facecolor="white")

    for ax, path, label, cmap in [
        (axes[0], fits_i_path, "MFS I-model", "magma"),
        (axes[1], fits_v_path, "MFS V-model", "RdBu_r"),
    ]:
        if os.path.exists(path):
            with fits.open(path) as hdul:
                data = np.squeeze(hdul[0].data)
                header = hdul[0].header
                w = WCS(header).celestial

                vmin, vmax = np.nanpercentile(data, [1, 99])
                if cmap == "RdBu_r":
                    vlim = max(abs(vmin), abs(vmax))
                    vmin, vmax = -vlim, vlim

                ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

                if target_coord:
                    px, py = w.world_to_pixel(target_coord)
                    if 0 <= px < data.shape[-1] and 0 <= py < data.shape[-2]:
                        ax.plot(px, py, 'w+', markersize=15, markeredgewidth=2, label=f"Target: {coord_str}")
                        ax.legend(fontsize=8, loc='upper right', framealpha=0.8)
                        print(f"    {label}: target at pixel ({px:.0f}, {py:.0f})")

                ax.set_xlabel("RA")
                ax.set_ylabel("DEC")

            ax.text(0.02, 0.95, label,
                    transform=ax.transAxes, fontsize=12, fontweight="bold",
                    color="white", ha="left", va="top",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
        else:
            ax.text(0.5, 0.5, f"{label} not found",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="#999999")
            print(f"    {label}: FITS not found")

    title_str = f"Source: {hostname}   |   SBID: {sbid}   |   Beam: {beam}"
    fig.suptitle(title_str, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    source_specific_dir = os.path.join(output_dir, hostname)
    os.makedirs(source_specific_dir, exist_ok=True)
    out_path = os.path.join(source_specific_dir, f"{base_name_str}_MFS.png")
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches='tight')
    plt.close(fig)
    print(f"  [MFS] -> {out_path}")


# ==========================================
# QC 诊断图：Im(I) + Im(V) + V/I
# ==========================================
def plot_qc(ds_file, output_dir, hostname, sbid, beam, base_name_str):
    print(f"  [QC] 绘制中...")

    try:
        ds = DynamicSpectrum(ds_path=ds_file, tavg=T_AVG_DS, favg=F_AVG, trim=True)
    except Exception as e:
        print(f"  Load failed: {e}")
        return

    freqs = ds.freq
    time_map = ds.time
    duration = time_map[-1] - time_map[0]

    re_i = np.real(ds.data.get("I"))
    im_i = np.imag(ds.data.get("I"))
    re_v = np.real(ds.data.get("V"))
    im_v = np.imag(ds.data.get("V"))

    i_threshold = np.nanmedian(np.abs(re_i)) * 0.1
    pol_frac = np.full_like(re_v, np.nan)
    mask = np.abs(re_i) > i_threshold
    pol_frac[mask] = np.abs(re_v[mask]) / re_i[mask] * 100

    print(f"    Re(I): mean={np.nanmean(re_i):.4f}  Im(I) rms={np.nanstd(im_i):.4f}")
    print(f"    Re(V): mean={np.nanmean(re_v):.4f}  Im(V) rms={np.nanstd(im_v):.4f}")
    print(f"    |V|/I: median={np.nanmedian(pol_frac):.2f}%")

    obs_time_str = ds.header.get("time_start", "unknown")
    if duration < 1:
        time_map = time_map * 60
        major_step = 1
        time_label = f"Time (minutes since {obs_time_str})"
    elif duration < 5:
        major_step = 0.5
        time_label = f"Time (hours since {obs_time_str})"
    else:
        major_step = 1.0
        time_label = f"Time (hours since {obs_time_str})"
    x_major_locator = ticker.MultipleLocator(major_step)
    x_minor_locator = ticker.AutoMinorLocator(5)

    fig = plt.figure(figsize=(15, 14), facecolor="white")
    gs = fig.add_gridspec(3, 2, width_ratios=[20, 0.5],
                          height_ratios=[1, 1, 1],
                          hspace=0, wspace=0.08)

    title_str = f"Data Quality: {hostname}   |   SBID: {sbid}   |   Beam: {beam}"
    fig.suptitle(title_str, fontsize=18, fontweight='bold', y=0.98)
    fig.subplots_adjust(top=0.95, bottom=0.03)

    # Panel 1: Im(Stokes I)
    ax1 = fig.add_subplot(gs[0, 0])
    vlim_i = np.nanpercentile(np.abs(im_i), 99)
    im1 = ax1.pcolormesh(time_map, freqs, im_i.T, cmap="coolwarm",
                          shading="auto", rasterized=True, vmin=-vlim_i, vmax=vlim_i)
    ax1.set_ylim(freqs[0], freqs[-1])
    ax1.set_ylabel("Frequency (MHz)", fontsize=12)
    ax1.text(0.02, 0.95, "Im(Stokes I)",
             transform=ax1.transAxes, fontsize=12, fontweight="bold",
             color="white", ha="left", va="top",
             path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    cax1 = fig.add_subplot(gs[0, 1])
    fig.colorbar(im1, cax=cax1, label="Im(I) (mJy)")
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.xaxis.set_major_locator(x_major_locator)
    ax1.xaxis.set_minor_locator(x_minor_locator)
    ax1.tick_params(labelsize=10, colors="#555555")

    # Panel 2: Im(Stokes V)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    vlim_v = np.nanpercentile(np.abs(im_v), 99)
    im2 = ax2.pcolormesh(time_map, freqs, im_v.T, cmap="coolwarm",
                          shading="auto", rasterized=True, vmin=-vlim_v, vmax=vlim_v)
    ax2.set_ylim(freqs[0], freqs[-1])
    ax2.set_ylabel("Frequency (MHz)", fontsize=12)
    ax2.text(0.02, 0.95, "Im(Stokes V)",
             transform=ax2.transAxes, fontsize=12, fontweight="bold",
             color="white", ha="left", va="top",
             path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    cax2 = fig.add_subplot(gs[1, 1])
    fig.colorbar(im2, cax=cax2, label="Im(V) (mJy)")
    ax2.tick_params(axis='x', labelbottom=False)
    ax2.tick_params(labelsize=10, colors="#555555")

    # Panel 3: V/I
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    im3 = ax3.pcolormesh(time_map, freqs, pol_frac.T, cmap="RdYlBu_r",
                          shading="auto", rasterized=True,
                          vmin=-POL_FRAC_LIMIT, vmax=POL_FRAC_LIMIT)
    ax3.set_ylim(freqs[0], freqs[-1])
    ax3.set_ylabel("Frequency (MHz)", fontsize=12)
    ax3.set_xlabel(time_label, fontsize=12)
    ax3.text(0.02, 0.95, "|V|/I Circular Polarization Fraction",
             transform=ax3.transAxes, fontsize=12, fontweight="bold",
             color="white", ha="left", va="top",
             path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    cax3 = fig.add_subplot(gs[2, 1])
    fig.colorbar(im3, cax=cax3, label="|V|/I (%)")
    ax3.tick_params(axis='x', labelbottom=True)
    ax3.tick_params(labelsize=10, colors="#555555")

    source_specific_dir = os.path.join(output_dir, hostname)
    os.makedirs(source_specific_dir, exist_ok=True)
    out_path = os.path.join(source_specific_dir, f"{base_name_str}_QC.png")
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches='tight')
    plt.close(fig)
    print(f"  [QC] -> {out_path}")


# ==========================================
# 主流程
# ==========================================
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

    print("-" * 60)
    print(f"Processing: {basename}")

    # 1. 科学图（始终画）
    plot_science(ds_file, output_dir, hostname, sbid, beam, base_name_str)

    # 2. MFS 模型图（开关控制）
    if INCLUDE_MFS:
        plot_mfs(output_dir, hostname, sbid, beam, base_name_str)
    else:
        print("  [MFS] 已跳过（INCLUDE_MFS = False）")

    # 3. QC 诊断图（开关控制）
    if INCLUDE_QC:
        plot_qc(ds_file, output_dir, hostname, sbid, beam, base_name_str)
    else:
        print("  [QC] 已跳过（INCLUDE_QC = False）")


def main():
    print("=" * 60)
    print(" ASKAP Dynamic Spectrum Plot Pipeline")
    print(f" Science: ON | MFS: {'ON' if INCLUDE_MFS else 'OFF'} | QC: {'ON' if INCLUDE_QC else 'OFF'}")
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
