import os
import re
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.time import Time
import lightkurve as lk
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


# ============================================================
# 配置区
# ============================================================
PERIOD_DAYS = 0.1664       # 折叠周期（天）
BIN_SEC = 60               # 时间分箱（秒）
XLO, XHI = 0.0, 2.0        # 双周期显示

# ── TESS ──
# TESS_FILE = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Data/TESS_Data/2MASS_J01033563-5515561_A/Sector_00/mastDownload/HLSP/hlsp_qlp_tess_ffi_s0069-0000000616014335_tess_v01_llc/hlsp_qlp_tess_ffi_s0069-0000000616014335_tess_v01_llc.fits"
INCLUDE_TESS = True         # 是否在顶部加入 TESS 相位折叠面板

# ── 射电 DS ──
DS_FILES = [
]
BATCH_PROCESS = True
DS_FILES_DIR = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Result/DS/2MASS_J01033563-5515561_A/DS_Results"
SOURCE_FILTER = ""

OUTPUT_BASE = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Result/Epoch_Comparison"

ASKAP_CATALOGUE_CSV = project_path('Processed_Data/Catalogue/01.askap_catalogue.csv')

# 绘图参数
FIG_W = 12
PANEL_H = 3.3
LINE_LW = 1.5
ERR_MS = 3
ERR_ELW = 0.8
COL_I, COL_V = "black", "#e74c3c"
COL_TESS = "#4a8fd4"
COL_TESS_MED = "#e74c3c"
TESS_ALPHA = 0.35
DPI = 300
# ============================================================


def get_sbid_mjd_map():
    if not os.path.exists(ASKAP_CATALOGUE_CSV):
        return {}
    df = pd.read_csv(ASKAP_CATALOGUE_CSV)
    df.columns = df.columns.str.strip()
    sbid_mjd = {}
    for _, row in df.iterrows():
        match = re.search(r'(\d+)', str(row.get('obs_id', '')))
        if match:
            sbid_mjd[match.group(1)] = float(row['t_min'])
    return sbid_mjd


def read_ds(ds_path):
    ds = DynamicSpectrum(ds_path=ds_path, tavg=1, favg=1, trim=True,
                         absolute_times=True, calscans=True, barycentre=True)
    I_t = np.nanmean(ds.data["I"].real, axis=1)
    V_t = np.nanmean(ds.data["V"].real, axis=1)
    t0 = Time(ds.header["time_start"], scale=str(ds.header.get("time_scale", "utc")).lower())
    t_abs = t0 + (ds.time * ds.tunit)
    bjd = t_abs.tdb.jd
    m = np.isfinite(bjd) & np.isfinite(I_t) & np.isfinite(V_t)
    return bjd[m], I_t[m], V_t[m], float(t0.tdb.jd)


def load_tess(fits_path):
    """加载 TESS FITS，返回 (mjd_tdb, flux_norm, t0_mjd_tdb)。"""
    lc = lk.read(fits_path)
    if hasattr(lc, "to_lightcurve"):
        lc = lc.to_lightcurve(aperture_mask="pipeline")
    lc = lc.remove_nans().remove_outliers(sigma=5)
    try:
        lc = lc.flatten(window_length=401)
    except Exception:
        pass
    flux_median = float(np.nanmedian(lc.flux.value))
    lc_norm = lc / flux_median
    tess_mjd = lc_norm.time.tdb.mjd
    tess_flux = lc_norm.flux.value
    t0_mjd = Time(lc_norm.time[0].tdb.mjd, format="mjd", scale="tdb").mjd
    return tess_mjd, tess_flux, t0_mjd


def bin_time_jd_multi(t_jd, y1, y2, dt_sec=60):
    t_jd = np.asarray(t_jd, float); y1 = np.asarray(y1, float); y2 = np.asarray(y2, float)
    m = np.isfinite(t_jd) & np.isfinite(y1) & np.isfinite(y2)
    t_jd, y1, y2 = t_jd[m], y1[m], y2[m]
    if t_jd.size == 0:
        return (np.array([]),) * 5
    dt = dt_sec / 86400.0
    edges = np.arange(np.nanmin(t_jd), np.nanmax(t_jd) + dt, dt)
    ind = np.digitize(t_jd, edges) - 1
    tb, y1b, e1b, y2b, e2b = [], [], [], [], []
    for k in range(len(edges) - 1):
        mk = (ind == k)
        nk = int(np.sum(mk))
        if nk <= 0:
            continue
        tb.append(np.nanmean(t_jd[mk]))
        y1b.append(np.nanmean(y1[mk]))
        y2b.append(np.nanmean(y2[mk]))
        if nk > 1:
            e1b.append(np.nanstd(y1[mk]) / np.sqrt(nk))
            e2b.append(np.nanstd(y2[mk]) / np.sqrt(nk))
        else:
            e1b.append(np.nan), e2b.append(np.nan)
    return (np.array(tb), np.array(y1b), np.array(e1b), np.array(y2b), np.array(e2b))


def fold_raw(tb, I, Ie, V, Ve, P, T0_ref):
    phi_abs = (tb - T0_ref) / P
    m = (phi_abs >= XLO) & (phi_abs <= XHI) & np.isfinite(I) & np.isfinite(V)
    phi = phi_abs[m]
    idx = np.argsort(phi)
    return phi[idx], I[m][idx], Ie[m][idx], V[m][idx], Ve[m][idx]


def fold_mod2(tb, I, Ie, V, Ve, P, T0_ref):
    phi_abs = (tb - T0_ref) / P
    phi = np.mod(phi_abs, 2.0)
    m = (phi >= XLO) & (phi < XHI) & np.isfinite(I) & np.isfinite(V)
    idx = np.argsort(phi[m])
    return phi[m][idx], I[m][idx], Ie[m][idx], V[m][idx], Ve[m][idx]


def fold_tess_to_radio(mjd, flux, P, T0_ref, nbins=60):
    """将 TESS 数据折叠到 [0,1)，平铺到 [0,2) 以对齐射电双周期。"""
    phi = np.mod((mjd - T0_ref) / P, 1.0)
    m = np.isfinite(phi) & np.isfinite(flux)
    phi, flux = phi[m], flux[m]

    edges = np.linspace(0, 1, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned = np.full(nbins, np.nan)
    errs = np.full(nbins, np.nan)
    for i in range(nbins):
        mask = (phi >= edges[i]) & (phi < edges[i + 1])
        n = mask.sum()
        if n > 3:
            binned[i] = np.nanmedian(flux[mask])
            errs[i] = np.nanstd(flux[mask]) / np.sqrt(n)

    # 平铺到 [0, 2)
    centers_2 = np.concatenate([centers, centers + 1.0])
    binned_2 = np.tile(binned, 2)
    errs_2 = np.tile(errs, 2)
    valid = ~np.isnan(binned_2)
    return centers_2[valid], binned_2[valid], errs_2[valid]


def plot_with_breaks(ax, x, y, **kwargs):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if x.size == 0:
        return
    idx = np.argsort(x); x, y = x[idx], y[idx]
    breaks = np.where(np.diff(x) > 0.2)[0]
    start = 0
    for b in np.append(breaks, x.size - 1):
        end = b + 1
        if end - start >= 2:
            ax.plot(x[start:end], y[start:end], **kwargs)
        start = end


def plot_panel_radio(ax, phi, I, Ie, V, Ve, title, break_lines=False, show_legend=True):
    ax.errorbar(phi, I, yerr=Ie, fmt='.', ms=ERR_MS, elinewidth=ERR_ELW,
                alpha=0.6, color=COL_I, label="_nolegend_")
    ax.errorbar(phi, V, yerr=Ve, fmt='.', ms=ERR_MS, elinewidth=ERR_ELW,
                alpha=0.6, color=COL_V, label="_nolegend_")

    if break_lines:
        plot_with_breaks(ax, phi, I, lw=LINE_LW, color=COL_I, label="Stokes I")
        plot_with_breaks(ax, phi, V, lw=LINE_LW, color=COL_V, label="Stokes V", zorder=10)
    else:
        ax.plot(phi, I, lw=LINE_LW, color=COL_I, label="Stokes I")
        ax.plot(phi, V, lw=LINE_LW, color=COL_V, label="Stokes V", zorder=10)

    ax.axvline(1.0, ls="--", lw=1.0, color="gray", alpha=0.6)
    ax.set_xlim(XLO, XHI)
    ax.margins(x=0)
    ax.text(0.02, 0.95, title, transform=ax.transAxes, va="top", ha="left", fontsize=11)
    ax.tick_params(axis='both', labelsize=11)
    if show_legend:
        ax.legend(loc="upper right", fontsize=10, framealpha=0.8, edgecolor="#dddddd")


def plot_panel_tess(ax, phi, flux, flux_err, title):
    """TESS 面板：散点 + 分箱中位数 + 误差带。"""
    ax.scatter(phi, flux, s=3, c=COL_TESS, alpha=TESS_ALPHA, rasterized=True, linewidths=0)
    ax.errorbar(phi, flux, yerr=flux_err, fmt='o', ms=3, color=COL_TESS_MED,
                capsize=2, lw=1.0, label="TESS folded")
    ax.axvline(1.0, ls="--", lw=1.0, color="gray", alpha=0.6)
    ax.set_xlim(XLO, XHI)
    ax.margins(x=0)
    ax.set_ylabel("Relative Flux", fontsize=12)
    ax.text(0.02, 0.95, title, transform=ax.transAxes, va="top", ha="left", fontsize=11)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.8, edgecolor="#dddddd")


def main():
    print("=" * 60)
    print(" Multi-Epoch Phase Folding (TESS + Radio)")
    print(f" Period: {PERIOD_DAYS:.6f} d ({PERIOD_DAYS*24:.4f} h)")
    print("=" * 60)

    # ── 加载 TESS ──
    tess_folded = None
    if INCLUDE_TESS and os.path.exists(TESS_FILE):
        try:
            tess_mjd, tess_flux, t0_tess = load_tess(TESS_FILE)
            print(f"TESS: {len(tess_mjd)} pts, MJD range [{tess_mjd[0]:.4f}, {tess_mjd[-1]:.4f}]")
        except Exception as e:
            print(f"[WARN] TESS 加载失败: {e}")
            tess_folded = None
    else:
        print("[INFO] 跳过 TESS")

    # ── 加载射电 ──
    if BATCH_PROCESS:
        all_ds = sorted(glob.glob(os.path.join(DS_FILES_DIR, "*.ds")))
        if SOURCE_FILTER:
            all_ds = [f for f in all_ds if SOURCE_FILTER in os.path.basename(f)]
    else:
        all_ds = DS_FILES

    if not all_ds:
        print("[ERROR] 未找到 .ds 文件"); sys.exit(1)

    print(f"找到 {len(all_ds)} 个 .ds 文件")

    epochs = []
    for ds_file in all_ds:
        basename = os.path.basename(ds_file)
        sb_match = re.search(r'SB(\d+)', basename, re.IGNORECASE)
        sbid = sb_match.group(1) if sb_match else "???"
        try:
            bjd, I_t, V_t, T0 = read_ds(ds_file)
            dur_h = (bjd[-1] - bjd[0]) * 24
            print(f"  [OK] {basename} | SB{sbid} | {bjd.size} pts | duration={dur_h:.2f}h")
            epochs.append({'sbid': sbid, 'bjd': bjd, 'I': I_t, 'V': V_t, 'T0': T0})
        except Exception as e:
            print(f"  [WARN] 读取 {basename} 失败: {e}")

    if len(epochs) < 1:
        print("[ERROR] 没有有效数据"); sys.exit(1)

    PHASE_ZERO_TIME = epochs[0]['T0']
    print(f"\nPhase zero (first ASKAP epoch start): BJD_TDB = {PHASE_ZERO_TIME:.9f}")

    P = PERIOD_DAYS
    folded_radio = []
    for i, ep in enumerate(epochs):
        tb, Ib, Ibe, Vb, Vbe = bin_time_jd_multi(ep['bjd'], ep['I'], ep['V'], dt_sec=BIN_SEC)
        phi, I_p, I_e, V_p, V_e = fold_mod2(tb, Ib, Ibe, Vb, Vbe, P, PHASE_ZERO_TIME)
        folded_radio.append({'phi': phi, 'I': I_p, 'Ie': I_e, 'V': V_p, 'Ve': V_e,
                             'title': f"SB{ep['sbid']}", 'break': True})
        if phi.size:
            print(f"  folded SB{ep['sbid']}: {phi.size} pts, phi=[{np.nanmin(phi):.3f}, {np.nanmax(phi):.3f}]")

    # ── 折叠 TESS ──
    if INCLUDE_TESS and tess_folded is None:
        try:
            phi_t, flux_b, err_b = fold_tess_to_radio(tess_mjd, tess_flux, P, PHASE_ZERO_TIME)
            tess_folded = {'phi': phi_t, 'flux': flux_b, 'err': err_b, 'title': "TESS"}
            print(f"  folded TESS: {len(phi_t)} pts")
        except Exception as e:
            print(f"  [WARN] TESS 折叠失败: {e}")
            tess_folded = None

    # ── 绘图：TESS 最上，射电依次 ──
    n_radio = len(folded_radio)
    n_tess = 1 if tess_folded is not None else 0
    n_total = n_tess + n_radio

    fig_h = PANEL_H * n_total
    fig, axes = plt.subplots(n_total, 1, figsize=(FIG_W, fig_h), sharex=True)
    if n_total == 1:
        axes = [axes]

    idx = 0
    if tess_folded is not None:
        plot_panel_tess(axes[idx], tess_folded['phi'], tess_folded['flux'],
                        tess_folded['err'], tess_folded['title'])
        axes[idx].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        idx += 1

    for i, fd in enumerate(folded_radio):
        plot_panel_radio(axes[idx], fd['phi'], fd['I'], fd['Ie'], fd['V'], fd['Ve'],
                         fd['title'], break_lines=fd['break'], show_legend=(i == 0))
        if idx < n_total - 1:
            axes[idx].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        idx += 1

    axes[-1].set_xlabel("Rotational phase (cycles)", fontsize=13)
    for ax in axes:
        # Skip ylabel for radio panels (they have I/V fluxes in mJy)
        pass

    # Set radio ylabels
    if tess_folded is not None:
        # TESS panel already has its own ylabel set in plot_panel_tess
        pass
    for ax_idx in range(n_tess, n_total):
        axes[ax_idx].set_ylabel("Flux (mJy)", fontsize=12)

    hostname_match = re.search(r'(.+?)_SB\d+', os.path.basename(all_ds[0]))
    hostname = hostname_match.group(1) if hostname_match else "Unknown"
    fig.suptitle(f"{hostname} | TESS + {n_radio} radio epochs | P={PERIOD_DAYS:.6f} d ({PERIOD_DAYS*24:.2f} h)", fontsize=14, y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_dir = os.path.join(OUTPUT_BASE, hostname)
    os.makedirs(output_dir, exist_ok=True)
    tag = "TESS_" if tess_folded else ""
    out_png = os.path.join(output_dir, f"{tag}{hostname}_MultiEpoch_P{PERIOD_DAYS}.png")
    out_pdf = os.path.join(output_dir, f"{tag}{hostname}_MultiEpoch_P{PERIOD_DAYS}.pdf")
    fig.savefig(out_png, dpi=DPI, facecolor="white", bbox_inches='tight')
    fig.savefig(out_pdf, facecolor="white", bbox_inches='tight')
    plt.close(fig)

    # 保存相位折叠数据
    for i, fd in enumerate(folded_radio):
        csv_out = os.path.join(output_dir, f"{hostname}_SB{epochs[i]['sbid']}_phase_P{PERIOD_DAYS}.csv")
        pd.DataFrame({
            "phase": fd['phi'], "I_mJy": fd['I'], "I_err": fd['Ie'],
            "V_mJy": fd['V'], "V_err": fd['Ve']
        }).dropna().to_csv(csv_out, index=False)
    if tess_folded is not None:
        csv_tess = os.path.join(output_dir, f"{hostname}_TESS_phase_P{PERIOD_DAYS}.csv")
        pd.DataFrame({
            "phase": tess_folded['phi'], "flux_norm": tess_folded['flux'], "err": tess_folded['err']
        }).dropna().to_csv(csv_tess, index=False)
    # 保存参考历元
    ref_csv = os.path.join(output_dir, f"{hostname}_reference_epoch.csv")
    pd.DataFrame({
        "parameter": ["PHASE_ZERO_BJD_TDB", "period_days", "period_hours"],
        "value": [f"{PHASE_ZERO_TIME:.9f}", f"{PERIOD_DAYS:.8f}", f"{PERIOD_DAYS*24:.6f}"]
    }).to_csv(ref_csv, index=False)

    print(f"\nSaved:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    main()
