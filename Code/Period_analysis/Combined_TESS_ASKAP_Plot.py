import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
import lightkurve as lk
from dstools.dynamic_spectrum import DynamicSpectrum

# ==========================================
# 1. 手动配置区
# ==========================================
PERIOD = 0.166  # 折叠周期 (天)

# DS 文件路径
DS_FILE = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/Ds/2MASS_J01033563-5515561_A/2MASS_J01033563-5515561_A_SB68040_beam10.ds"

# TESS FITS 文件路径 (手动选择)
TESS_FILE = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/TESS_Data/2MASS_J01033563-5515561/Sector_00/mastDownload/HLSP/hlsp_tess-spoc_tess_phot_0000000206502540-s0069_tess_v1_tp/hlsp_tess-spoc_tess_phot_0000000206502540-s0069_tess_v1_tp.fits"

# 输出目录
OUTPUT_DIR = "/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Result/Combined/2MASS_J01033563-5515561_A"

# 动态谱色标范围 (mJy)
I_LIMIT = 6.0
V_LIMIT = 3.5

# 时间/频率平均因子
T_AVG = 15    # 时间平均因子
F_AVG = 3     # 频率平均因子

# 调色板
COLOR_TESS = "#4a8fd4"
COLOR_MEDIAN = "#e74c3c"
COLOR_I = "black"
COLOR_V = "#e74c3c"

# ==========================================
# 2. 辅助函数
# ==========================================
def project_path(relative_path):
    current = os.path.abspath(os.path.dirname(__file__))
    while not (os.path.isdir(os.path.join(current, 'Code')) and
               os.path.isdir(os.path.join(current, 'Processed_Data'))):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.join(current, relative_path)


# ==========================================
# 3. 获取 ASKAP 观测起点 MJD (t0)
# ==========================================
sbid_match = re.search(r'SB(\d+)', os.path.basename(DS_FILE), re.IGNORECASE)
if not sbid_match:
    print("ERROR: Cannot extract SBID from DS file name")
    sys.exit(1)
target_sbid = sbid_match.group(1)
print(f"Target SBID: {target_sbid}")

catalogue_path = project_path('Processed_Data/Catalogue/01.askap_catalogue.csv')
df_cat = pd.read_csv(catalogue_path)

t0_mjd = None
for _, row in df_cat.iterrows():
    sbid_in_row = str(row['obs_id'])
    if target_sbid in sbid_in_row:
        t0_mjd = float(row['t_min'])
        break

if t0_mjd is None:
    print(f"ERROR: SBID {target_sbid} not found in catalogue")
    sys.exit(1)
print(f"ASKAP obs start MJD (t0): {t0_mjd:.6f}")


# ==========================================
# 4. 加载 TESS 数据
# ==========================================
print(f"\nLoading TESS: {os.path.basename(TESS_FILE)}")
lc = lk.read(TESS_FILE)
if hasattr(lc, "to_lightcurve"):
    lc = lc.to_lightcurve(aperture_mask="pipeline")
lc = lc.remove_nans().remove_outliers(sigma=5)
flux_median = float(np.nanmedian(lc.flux.value))
lc_norm = lc / flux_median
tess_btjd = lc_norm.time.value
tess_flux = lc_norm.flux.value
tess_mjd = tess_btjd + 2457000 - 2400000.5
print(f"  TESS data: {len(tess_btjd)} points, MJD range [{tess_mjd[0]:.4f}, {tess_mjd[-1]:.4f}]")

# ==========================================
# 5. 加载 DS 数据
# ==========================================
print(f"\nLoading DS: {os.path.basename(DS_FILE)}")
ds = DynamicSpectrum(DS_FILE, tavg=T_AVG, favg=F_AVG, trim=True)
t_hours = ds.time
freqs = ds.freq

stokes_i_2d = np.real(ds.data.get("I"))
stokes_v_2d = np.real(ds.data.get("V"))

flux_i = np.nanmean(stokes_i_2d, axis=1)
flux_v = np.nanmean(stokes_v_2d, axis=1)
flux_i_detrend = flux_i - np.nanmedian(flux_i)
flux_v_detrend = flux_v - np.nanmedian(flux_v)

flux_i_err = np.nanstd(stokes_i_2d, axis=1) / np.sqrt(len(freqs))
flux_v_err = np.nanstd(stokes_v_2d, axis=1) / np.sqrt(len(freqs))

ds_duration = t_hours[-1] - t_hours[0]
print(f"  DS: {len(t_hours)} time samples, {len(freqs)} freq channels")
print(f"  DS time: tmin={t_hours[0]:.2f}h, tmax={t_hours[-1]:.2f}h, duration={ds_duration:.2f}h")
print(f"  Freq: {freqs[0]:.1f} - {freqs[-1]:.1f} MHz")

# ==========================================
# 6. 相位计算
# ==========================================
period_hours = PERIOD * 24.0
n_phases = ds_duration / period_hours
display_max = n_phases  # DS 数据覆盖多少相位就显示多少
n_tiles = int(np.ceil(display_max))  # TESS 平铺仍需取整
print(f"\nPhase info: period = {PERIOD} d = {period_hours:.2f} h")
print(f"  DS covers {n_phases:.2f} phases, display 0 - {display_max:.2f}")

# TESS 相位 [0, 1) — 所有 TESS 数据折叠到单周期
phase_tess = (tess_mjd - t0_mjd) / PERIOD % 1.0

# TESS 观测起始时刻在相位轴上的位置
tess_start_phase = (tess_mjd[0] - t0_mjd) / PERIOD % 1.0
print(f"  TESS start phase (at t0): {tess_start_phase:.4f}")

# ASKAP 相位 — 不折叠，原始映射 t_hours → phase
phase_askap = t_hours / period_hours

# ==========================================
# 7. TESS 相位分箱 (fold to [0, 1))
# ==========================================
NBINS = 60
bins = np.linspace(0, 1, NBINS + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_medians = np.full(NBINS, np.nan)
for i in range(NBINS):
    mask = (phase_tess >= bins[i]) & (phase_tess < bins[i + 1])
    if np.sum(mask) > 5:
        bin_medians[i] = np.nanmedian(tess_flux[mask])

# 平铺到 [0, display_max)
tiled_phases = np.concatenate([bin_centers + i for i in range(n_tiles)])
tiled_flux = np.tile(bin_medians, n_tiles)
valid = ~np.isnan(tiled_flux)

# ==========================================
# 8. 绘图
# ==========================================
fig = plt.figure(figsize=(15, 14), facecolor="white")
gs = fig.add_gridspec(4, 2, width_ratios=[20, 0.6],
                      height_ratios=[1.4, 1, 1.2, 1.2],
                      hspace=0, wspace=0.05)

# --- Panel 1: TESS Phase-Folded Lightcurve ---
ax1 = fig.add_subplot(gs[0, 0])

# TESS 散点（折叠到 display_range 内）
for offset in range(n_tiles):
    ph_offset = phase_tess + offset
    mask = (ph_offset >= 0) & (ph_offset <= display_max)
    if np.sum(mask) > 0:
         ax1.scatter(ph_offset[mask], tess_flux[mask], s=4, c=COLOR_TESS,
                     alpha=0.35, rasterized=True, linewidths=0)

ax1.plot(tiled_phases[valid], tiled_flux[valid],
         color=COLOR_MEDIAN, linewidth=2.5, label="TESS folded median")
ax1.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8)
ax1.set_xlim(0, display_max)
y_min = np.nanpercentile(tess_flux, 5)
y_max = np.nanpercentile(tess_flux, 95)
y_pad = (y_max - y_min) * 0.1
ax1.set_ylim(y_min - y_pad, y_max + y_pad)
ax1.set_ylabel("Relative Flux", fontsize=13, color="#333333")
ax1.text(0.02, 0.95, f"TESS PhaseFolding  (P = {PERIOD:.4f} d)",
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

# --- Panel 2: ASKAP Stokes I/V Lightcurves ---
ax2 = fig.add_subplot(gs[1, 0])
ax2.errorbar(phase_askap, flux_i_detrend, yerr=flux_i_err,
             fmt='-', color=COLOR_I, linewidth=0.8, capsize=2,
             alpha=0.85, label="Stokes I")
ax2.errorbar(phase_askap, flux_v_detrend, yerr=flux_v_err,
             fmt='-', color=COLOR_V, linewidth=0.8, capsize=2,
             alpha=0.85, label="Stokes V")
ax2.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
ax2.set_xlim(0, display_max)
ax2.set_ylabel("Detrended Flux (mJy)", fontsize=13, color="#333333")
ax2.text(0.02, 0.95, "Lightcurve",
         transform=ax2.transAxes, fontsize=12, fontweight="bold",
         color="white", ha="left", va="top",
         path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
ax2.legend(fontsize=10, loc="upper right", framealpha=0.8, edgecolor="#dddddd")
ax2.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, color="#aaaaaa")
ax2.set_axisbelow(True)
ax2.xaxis.set_major_locator(ticker.MaxNLocator(8))
ax2.yaxis.set_major_locator(ticker.MaxNLocator(5))
ax2.tick_params(axis='x', labelbottom=False)
for spine in ax2.spines.values():
    spine.set_linewidth(0.5)
    spine.set_color("#cccccc")
ax2.tick_params(labelsize=10, colors="#555555")

# --- Panel 3: Stokes I Dynamic Spectrum ---
ax3 = fig.add_subplot(gs[2, 0])
im_i = ax3.pcolormesh(phase_askap, freqs, stokes_i_2d.T,
                       cmap="coolwarm", shading="auto", rasterized=True)
im_i.set_clim(-I_LIMIT, I_LIMIT)
ax3.set_xlim(0, display_max)
ax3.set_ylim(freqs[0], freqs[-1])
ax3.set_ylabel("Frequency (MHz)", fontsize=12, color="#333333")
ax3.text(0.02, 0.95, "Stokes I",
         transform=ax3.transAxes, fontsize=12, fontweight="bold",
         color="white", ha="left", va="top",
         path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
cax_i = fig.add_subplot(gs[2, 1])
fig.colorbar(im_i, cax=cax_i, label="Stokes I (mJy)")
ax3.tick_params(axis='x', labelbottom=False)
ax3.tick_params(labelsize=10, colors="#555555")

# --- Panel 4: Stokes V Dynamic Spectrum ---
ax4 = fig.add_subplot(gs[3, 0])
im_v = ax4.pcolormesh(phase_askap, freqs, stokes_v_2d.T,
                       cmap="coolwarm", shading="auto", rasterized=True)
im_v.set_clim(-V_LIMIT, V_LIMIT)
ax4.set_xlim(0, display_max)
ax4.set_ylim(freqs[0], freqs[-1])
ax4.set_xlabel("Phase", fontsize=13, color="#333333")
ax4.set_ylabel("Frequency (MHz)", fontsize=12, color="#333333")
ax4.text(0.02, 0.95, "Stokes V",
         transform=ax4.transAxes, fontsize=12, fontweight="bold",
         color="white", ha="left", va="top",
         path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
cax_v = fig.add_subplot(gs[3, 1])
fig.colorbar(im_v, cax=cax_v, label="Stokes V (mJy)")
ax4.tick_params(labelsize=10, colors="#555555")

# ==========================================
# 9. 保存
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

basename_no_ext = os.path.splitext(os.path.basename(DS_FILE))[0]
out_img = os.path.join(OUTPUT_DIR, f"{basename_no_ext}_P{PERIOD}_Combined.png")
fig.savefig(out_img, dpi=300, facecolor="white", edgecolor="none")
plt.close(fig)
print(f"\nImage saved: {out_img}")

out_csv = os.path.join(OUTPUT_DIR, f"{basename_no_ext}_P{PERIOD}_PhaseInfo.csv")
pd.DataFrame({
    "parameter": ["t0_mjd", "period_days", "period_hours", "ds_duration_hours",
                  "n_phases", "display_max", "n_tiles", "tess_start_phase"],
    "value": [f"{t0_mjd:.6f}", f"{PERIOD:.6f}", f"{period_hours:.3f}",
              f"{ds_duration:.3f}", f"{n_phases:.3f}", f"{display_max:.3f}",
              f"{n_tiles}", f"{tess_start_phase:.4f}"]
}).to_csv(out_csv, index=False)
print(f"Phase info saved: {out_csv}")
