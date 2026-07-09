"""I/V 交叉排查脚本：一键切换参数，对比不同配置的曲线效果。"""
import os, re, sys, shutil, tarfile, subprocess, glob
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import casacore.tables as pt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ======================================================================
#  排查方案：改下面参数，每改完一次跑脚本即生成一个对比数据点
# ======================================================================
TAR_PATH = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Data/Ms_Data/Proxima_Cen/50381_scienceData.VAST_1453-62.SB50381.VAST_1453-62.beam33_averaged_cal.leakage.ms.tar"

# ── 数据处理开关 ──
DO_PREPROCESS = True            # 是否执行 dstools-askap-preprocess
DO_INSERT = True                # 是否执行 dstools-insert-model
DO_SUBTRACT = True              # 是否执行 dstools-subtract-model
PREDICT_AFTER = False           # insert-model 后额外跑 wsclean -predict
DATACOLUMN = "data"             # 提取来源: "data" / "corrected"
EXTRACT_FROM = "subtracted"     # 提取对象: "subtracted" / "clean"

# ── 提取参数 ──
MINUVDIST = 500                 # -u: 最小基线(λ), 0=不切
BASELINE_AVERAGE = "averaged"   # "averaged"(-B) / "no-average"(--no-baseline-average)

# ── insert-model 参数 ──
MASK_RADIUS = 15                # 掩模半径 (asec)

# ── 诊断开关 ──
DO_RAW_DIAGNOSTIC = True        # 画原始相关积（XX/YY/XY/YX）随时间的变化
DO_LEAKAGE_ESTIMATE = True      # 估计 leakage 参数（静默期 V/I 比）
APPLY_LEAKAGE_CORRECTION = False # 对最终 .ds 应用经验 leakage 修正
LEAKAGE_ALPHA = 0.0             # I' = I + alpha * V
LEAKAGE_BETA = 0.0              # V' = V + beta * I

# ======================================================================

PIPELINE_RESULTS_BASE = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Result/DS"


def project_path(relative_path: str) -> str:
    current = os.path.abspath(os.path.dirname(__file__))
    while not (os.path.isdir(os.path.join(current, 'Code')) and os.path.isdir(os.path.join(current, 'Processed_Data'))):
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.join(os.getcwd(), relative_path)
        current = parent
    return os.path.join(current, relative_path)


def extract_sbid_and_beam(filename: str):
    sb_match = re.search(r'SB(\d+)', filename, re.IGNORECASE)
    beam_match = re.search(r'beam(\d+)', filename, re.IGNORECASE)
    sbid = str(int(sb_match.group(1))) if sb_match else None
    beam = str(int(beam_match.group(1))) if beam_match else None
    return sbid, beam


def run_cmd(cmd_str: str, cwd: str) -> None:
    conda_bin_dir = os.path.dirname(sys.executable)
    conda_lib_dir = os.path.join(os.path.dirname(conda_bin_dir), "lib")
    custom_env = os.environ.copy()
    custom_env["PATH"] = conda_bin_dir + os.pathsep + custom_env.get("PATH", "")
    custom_env["LD_LIBRARY_PATH"] = conda_lib_dir + os.pathsep + custom_env.get("LD_LIBRARY_PATH", "")
    cmd_str = f"env LD_LIBRARY_PATH={custom_env['LD_LIBRARY_PATH']} {cmd_str}"
    subprocess.run(cmd_str, shell=True, check=True, cwd=cwd, executable='/bin/bash', env=custom_env)


def raw_correlation_diagnostic(ms_path, output_png, title_prefix=""):
    """画原始 4 个相关积（XX/YY/XY/YX）的实部和虚部随时间变化，判断 I/V 问题来源。"""
    t = pt.table(ms_path, ack=False)
    d = t.getcol("DATA")
    time = t.getcol("TIME")
    t.close()

    XX, XY, YX, YY = d[:,:,0], d[:,:,1], d[:,:,2], d[:,:,3]
    t_h = (time - time[0]) / 3600

    xx = np.nanmean(XX.real, axis=1)
    yy = np.nanmean(YY.real, axis=1)
    re_xy = np.nanmean(XY.real, axis=1)
    im_xy = np.nanmean(XY.imag, axis=1)
    re_yx = np.nanmean(YX.real, axis=1)
    im_yx = np.nanmean(YX.imag, axis=1)
    I = (xx + yy) / 2
    V = (im_yx - im_xy) / 2

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    axes[0].plot(t_h, xx, 'b-', lw=0.5, label='Re(XX)')
    axes[0].plot(t_h, yy, 'r-', lw=0.5, label='Re(YY)')
    axes[0].set_ylabel('Parallel hands')
    axes[0].legend(fontsize=9)
    axes[0].axhline(0, color='gray', ls=':')

    axes[1].plot(t_h, re_xy, 'g-', lw=0.5, label='Re(XY)')
    axes[1].plot(t_h, re_yx, 'm-', lw=0.5, label='Re(YX)')
    axes[1].set_ylabel('Cross real')
    axes[1].legend(fontsize=9)
    axes[1].axhline(0, color='gray', ls=':')

    axes[2].plot(t_h, im_xy, 'g-', lw=0.5, label='Im(XY)')
    axes[2].plot(t_h, im_yx, 'm-', lw=0.5, label='Im(YX)')
    axes[2].set_ylabel('Cross imag → Stokes V')
    axes[2].legend(fontsize=9)
    axes[2].axhline(0, color='gray', ls=':')

    axes[3].plot(t_h, I, 'k-', lw=0.5, label='Stokes I')
    axes[3].plot(t_h, V, 'r-', lw=0.5, label='Stokes V')
    axes[3].set_ylabel('Stokes')
    axes[3].legend(fontsize=9)
    axes[3].axhline(0, color='gray', ls=':')

    axes[3].set_xlabel('Time (hours)')
    fig.suptitle(f'{title_prefix}Raw Correlations', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close(fig)

    print(f"\n  [原始相关积诊断] {output_png}")
    print(f"  Re(XX) mean={xx.mean():.2f}  <0: {(xx<0).sum()}/{len(xx)}")
    print(f"  Re(YY) mean={yy.mean():.2f}  <0: {(yy<0).sum()}/{len(yy)}")
    print(f"  Re(XY) mean={re_xy.mean():.4f}")
    print(f"  Re(YX) mean={re_yx.mean():.4f}")
    print(f"  Im(XY) mean={im_xy.mean():.4f}")
    print(f"  Im(YX) mean={im_yx.mean():.4f}")
    print(f"  Stokes I mean={I.mean():.2f}  I<0: {(I<0).sum()}/{len(I)}")
    print(f"  Stokes V mean={V.mean():.4f}  |V|>|I|: {(np.abs(V)>np.abs(I)).sum()}/{len(I)}")

    if xx.mean() < 0 and yy.mean() < 0:
        print("  >>> 平行手 XX/YY 均值为负 → 校准/灵敏度问题")
    elif abs(im_xy.mean()) > abs(I.mean()) * 0.1:
        print("  >>> 交叉手 Im(XY) 异常大 → leakage / XY phase 残留")
    if abs(re_xy.mean() - re_yx.mean()) > abs(re_xy.mean()) * 0.1:
        print("  >>> Re(XY) != Re(YX) → XY 相位不对称")

    return I, V


def estimate_leakage(I, V, quiet_percentile=20):
    """用静默期（I 最低的 N%）估计 leakage 参数。"""
    threshold = np.nanpercentile(I, quiet_percentile)
    quiet = I < threshold
    I_quiet = I[quiet]
    V_quiet = V[quiet]

    I_mean = np.nanmean(I_quiet)
    V_mean = np.nanmean(V_quiet)

    if abs(I_mean) < 1e-6:
        print("  [leakage] 静默期 I 均值太小，无法估计")
        return 0.0, 0.0

    beta = V_mean / I_mean
    alpha = -beta

    print(f"\n  [leakage 估计] 静默期({quiet_percentile}%): I={I_mean:.2f}, V={V_mean:.4f}")
    print(f"  beta={beta:.4f}  alpha={alpha:.4f}")
    print(f"  建议: LEAKAGE_ALPHA={alpha:.4f}, LEAKAGE_BETA={beta:.4f}")
    return alpha, beta


ASKAP_CATALOGUE_CSV = project_path('Processed_Data/Catalogue/01.askap_catalogue.csv')
INPUT_CSV = project_path('Processed_Data/Catalogue/02.final_confirmed_stars_direct_1.csv')

sbid, beam = extract_sbid_and_beam(os.path.basename(TAR_PATH))

obs_df = pd.read_csv(ASKAP_CATALOGUE_CSV)
obs_df.columns = obs_df.columns.str.strip()
obs_df['sbid_clean'] = obs_df['obs_id'].apply(
    lambda x: str(int(re.search(r'(\d+)', str(x)).group(1))) if re.search(r'(\d+)', str(x)) else None)
sbid_to_mjd = obs_df.dropna(subset=['sbid_clean']).drop_duplicates(subset=['sbid_clean']).set_index('sbid_clean')['t_min'].to_dict()
obs_mjd = sbid_to_mjd[sbid]

stars_df = pd.read_csv(INPUT_CSV)
stars_df.columns = stars_df.columns.str.strip()
stars_df['hostname_clean'] = stars_df['hostname'].astype(str).str.strip().str.replace(' ', '_')
star_catalog_dict = stars_df.drop_duplicates(subset=['hostname_clean']).set_index('hostname_clean').to_dict('index')

dir_name = os.path.basename(os.path.dirname(TAR_PATH))
norm_dir = re.sub(r'[^a-zA-Z0-9]', '', dir_name).lower()
matched_key = None
for k in star_catalog_dict.keys():
    norm_k = re.sub(r'[^a-zA-Z0-9]', '', k).lower()
    if norm_k in norm_dir or norm_dir in norm_k:
        matched_key = k
        break
clean_hostname = matched_key
star_meta = star_catalog_dict[clean_hostname]

pmra_val = star_meta.get('sy_pmra', star_meta.get('pmra', 0.0))
pmdec_val = star_meta.get('sy_pmdec', star_meta.get('pmdec', 0.0))
pmra = 0.0 if pd.isna(pmra_val) else float(pmra_val)
pmdec = 0.0 if pd.isna(pmdec_val) else float(pmdec_val)
plx_val = star_meta.get('sy_plx', star_meta.get('plx', 10.0))
plx = 10.0 if pd.isna(plx_val) or float(plx_val) <= 0 else float(plx_val)

star_j2015 = SkyCoord(ra=star_meta['ra'] * u.deg, dec=star_meta['dec'] * u.deg,
                      pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                      distance=(1000 / plx) * u.pc, frame='icrs', obstime=Time('J2015.5'))
obs_time = Time(obs_mjd, format='mjd')
star_at_obs = star_j2015.apply_space_motion(new_obstime=obs_time)
corr_ra, corr_dec = round(star_at_obs.ra.deg, 7), round(star_at_obs.dec.deg, 7)

star_results_dir = os.path.join(PIPELINE_RESULTS_BASE, clean_hostname)
ds_results_dir = os.path.join(star_results_dir, "DS_Results")
workspace_name = f"{clean_hostname}_SB{sbid}_beam{beam}_workspace"
workspace_dir = os.path.join(star_results_dir, workspace_name)
os.makedirs(workspace_dir, exist_ok=True)
os.makedirs(ds_results_dir, exist_ok=True)

with tarfile.open(TAR_PATH, 'r') as tar:
    top_dirs = {n.split('/')[0] for n in tar.getnames() if n.strip()}
extracted_folder_name = min(top_dirs, key=len)
name_parts = extracted_folder_name.split('.')
field_name = name_parts[1] if len(name_parts) > 1 else "UnknownField"
clean_ms_name = f"SB{sbid}.{field_name}.beam{beam}.ms"
subtracted_ms_name = f"SB{sbid}.{field_name}.beam{beam}.subtracted.ms"

# 为每次测试生成唯一的 DS 文件名
tag_parts = []
if not DO_PREPROCESS: tag_parts.append("nopre")
if not DO_INSERT: tag_parts.append("noins")
if not DO_SUBTRACT: tag_parts.append("nosub")
if PREDICT_AFTER: tag_parts.append("pred")
tag_parts.append(f"col{DATACOLUMN}")
tag_parts.append(f"from{EXTRACT_FROM}")
tag_parts.append(f"uv{MINUVDIST}" if MINUVDIST > 0 else "uv0")
if BASELINE_AVERAGE == "no-average": tag_parts.append("noavg")
tag_parts.append(f"r{MASK_RADIUS}")
tag = "_".join(tag_parts)
final_ds_name = f"{clean_hostname}_SB{sbid}_beam{beam}_{tag}.ds"

wsclean_model_dir_name = f"wsclean_model_{clean_hostname}_SB{sbid}_beam{beam}"

print("=" * 60)
print(f" 诊断方案: {tag}")
print(f" PREPROCESS={DO_PREPROCESS} INSERT={DO_INSERT} SUBTRACT={DO_SUBTRACT} PREDICT={PREDICT_AFTER}")
print(f" DATACOLUMN={DATACOLUMN} EXTRACT_FROM={EXTRACT_FROM}")
print(f" MINUVDIST={MINUVDIST} BASELINE={BASELINE_AVERAGE} MASK_RADIUS={MASK_RADIUS}")
print("=" * 60)

# 清理旧 MS
for f in glob.glob(os.path.join(workspace_dir, "SB*.*.ms")):
    shutil.rmtree(f, ignore_errors=True)
for f in glob.glob(os.path.join(workspace_dir, "*.ds")):
    os.remove(f)

os.chdir(workspace_dir)

# 解压 + rename
with tarfile.open(TAR_PATH, 'r') as tar:
    tar.extractall(path=workspace_dir)
os.rename(os.path.join(workspace_dir, extracted_folder_name), os.path.join(workspace_dir, clean_ms_name))

if DO_PREPROCESS:
    run_cmd(f"dstools-askap-preprocess {clean_ms_name}", cwd=workspace_dir)

if DO_INSERT:
    run_cmd(
        f"dstools-insert-model -p {corr_ra} {corr_dec} -r {MASK_RADIUS} {wsclean_model_dir_name} {clean_ms_name}",
        cwd=workspace_dir)

if DO_INSERT and PREDICT_AFTER:
    run_cmd(
        f"wsclean -predict -name {wsclean_model_dir_name}/wsclean -channels-out 8 -pol iv {clean_ms_name}",
        cwd=workspace_dir)

if DO_SUBTRACT:
    run_cmd(f"dstools-subtract-model -S {clean_ms_name}", cwd=workspace_dir)

# 决定提取对象
if EXTRACT_FROM == "subtracted" and DO_SUBTRACT:
    extract_from = subtracted_ms_name
elif EXTRACT_FROM == "clean":
    extract_from = clean_ms_name
else:
    extract_from = clean_ms_name  # fallback

# 构建 extract 命令
uv_flag = f"-u {MINUVDIST}" if MINUVDIST > 0 else ""
bl_flag = "--no-baseline-average" if BASELINE_AVERAGE == "no-average" else "-B"
extract_cmd = (
    f"dstools-extract-ds -p {corr_ra} {corr_dec} -v {uv_flag} {bl_flag} "
    f"-d {DATACOLUMN} {extract_from} {final_ds_name}"
)
print(f"Run: {extract_cmd}")
run_cmd(extract_cmd, cwd=workspace_dir)

shutil.move(os.path.join(workspace_dir, final_ds_name), os.path.join(ds_results_dir, final_ds_name))

ds_out = os.path.join(ds_results_dir, final_ds_name)
print(f"\n完成: {ds_out}")

# ── 原始相关积诊断 ──
if DO_RAW_DIAGNOSTIC:
    diag_ms = os.path.join(workspace_dir, clean_ms_name)
    diag_png = os.path.join(ds_results_dir, f"{clean_hostname}_SB{sbid}_beam{beam}_raw_corr.png")
    I_raw, V_raw = raw_correlation_diagnostic(diag_ms, diag_png, title_prefix=f"{clean_hostname} SB{sbid} Bm{beam} | ")

    if DO_LEAKAGE_ESTIMATE:
        alpha, beta = estimate_leakage(I_raw, V_raw)

# ── 经验 leakage 修正 ──
if APPLY_LEAKAGE_CORRECTION:
    print(f"\n应用 leakage 修正: alpha={LEAKAGE_ALPHA}, beta={LEAKAGE_BETA}")
    from dstools.dynamic_spectrum import DynamicSpectrum
    ds_obj = DynamicSpectrum(ds_path=ds_out)
    I_data = ds_obj.data['I'].real
    V_data = ds_obj.data['V'].real
    ds_obj.data['I'] = I_data + LEAKAGE_ALPHA * V_data
    ds_obj.data['V'] = V_data + LEAKAGE_BETA * I_data
    # 重新画图
    from dstools.plotting import plot_lightcurve
    fig, ax = plot_lightcurve(ds_obj, stokes="IV", imag=False)
    corrected_png = ds_out.replace('.ds', '_corrected.png')
    fig.savefig(corrected_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"修正后图: {corrected_png}")

# 自动画图
print("\n绘图...")
run_cmd(
    f"dstools-plot-ds -d {ds_out} -l -s IV -t 6 -f 8",
    cwd=ds_results_dir)
