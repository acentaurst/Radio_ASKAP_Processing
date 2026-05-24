import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats  # 替换回更稳健的 sigma clipping
from astropy.modeling import models, fitting
from astropy.time import Time
from matplotlib.ticker import SymmetricalLogLocator

import warnings

warnings.simplefilter('ignore')


def project_path(relative_path):
    current = os.path.abspath(os.path.dirname(__file__))
    while not (
        os.path.isdir(os.path.join(current, 'Code')) and
        os.path.isdir(os.path.join(current, 'Processed_Data'))
    ):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.join(current, relative_path)

# 1. 全局参数与配置
base_dir = project_path('Downloading_Data/fits_images')
time_info_file = project_path('Processed_Data/Catalogue/01.askap_catalogue.csv')
output_dir = project_path('Processed_Data/Fits_Data/Filtered')
os.makedirs(output_dir, exist_ok=True)

# 控制区
target_safe_name = 'Proxima_Cen'  # 指定源文件夹名称
use_log_scale = True  # 纵坐标对数开关：True (对数), False (线性)

#  时间截取开关
use_time_zoom = True  # 改为 True 开启时间缩放，改为 False 则读取全部时间
zoom_mjd_start = 59800
zoom_mjd_end = 61100

target_original_name = target_safe_name.replace('_', ' ')
fit_box_size = 5

color_I, color_V, color_VI = '#C8102E', '#0033A0', '#007749'
ghost_alpha, err_band_alpha = 0.35, 0.25


# 2. 核心提取函数 (加入 NaN 过滤与参数边界约束机制)
def extract_flux_at_center(data_2d, box=5):
    data_2d = np.squeeze(data_2d)
    y_center, x_center = data_2d.shape[0] // 2, data_2d.shape[1] // 2
    half_box = box // 2
    cutout = data_2d[y_center - half_box: y_center + half_box + 1, x_center - half_box: x_center + half_box + 1]

    # 如果整个提取框全都是 NaN，直接返回无效值
    if np.all(np.isnan(cutout)):
        return np.nan

    y_grid, x_grid = np.mgrid[:box, :box]

    # 获取局部像素的真实波动范围
    max_val = np.nanmax(cutout)
    abs_max = np.nanmax(np.abs(cutout))

    g_init = models.Gaussian2D(amplitude=max_val, x_mean=half_box, y_mean=half_box, x_stddev=1, y_stddev=1)

    # 约束条件
    g_init.x_mean.bounds = (0, box - 1)
    g_init.y_mean.bounds = (0, box - 1)
    g_init.x_stddev.bounds = (0.5, 3.0)
    g_init.y_stddev.bounds = (0.5, 3.0)
    g_init.amplitude.bounds = (-abs_max * 2, abs_max * 2)

    fitter = fitting.LevMarLSQFitter()

    # 添加 filter_non_finite=True，自动忽略NaN和Inf像素
    g_fit = fitter(g_init, x_grid, y_grid, cutout, filter_non_finite=True)
    return g_fit.amplitude.value


# 3. 定向提取数据
print(f" 开始处理 [筛选-纯图版]: {target_safe_name} ...")

stokes_I_path = os.path.join(base_dir, target_safe_name, "StokesI")
stokes_V_path = os.path.join(base_dir, target_safe_name, "StokesV")

all_I_files = glob.glob(os.path.join(stokes_I_path, "*.fits"))
sbid_dict = {}
for f in all_I_files:
    match = re.search(r'(ASKAP-\d+)', os.path.basename(f))
    if match: sbid_dict.setdefault(match.group(1), []).append(f)

t_info = pd.read_csv(time_info_file)
plot_data = []

for sbid, i_file_list in sbid_dict.items():
    t_min = t_info[t_info['obs_id'] == sbid].iloc[0]['t_min'] if sbid in t_info['obs_id'].values else 60000

    #  时间开关拦截逻辑
    if use_time_zoom:
        if (zoom_mjd_start and t_min < zoom_mjd_start) or (zoom_mjd_end and t_min > zoom_mjd_end):
            continue

    # ------ 寻找最佳 Stokes I 图像 ------
    best_file_I, min_rms_I = None, float('inf')
    for f in i_file_list:
        try:
            with fits.open(f) as hdul:
                data = np.squeeze(hdul[0].data)
                valid_pixels = data[(data != 0.0) & (~np.isnan(data))]
                if len(valid_pixels) < 100: continue

                # 条件一：迭代剔除>3σ的恒星与旁瓣信号，获取真实Local RMS
                _, _, rms = sigma_clipped_stats(valid_pixels, sigma=3.0, maxiters=5)

                # 条件二：ASKAP 物理底线，拒绝异常平滑废图
                if 1e-5 < rms < min_rms_I:
                    min_rms_I, best_file_I = rms, f
        except:
            continue
    if not best_file_I: continue

    # ------ 寻找最佳 Stokes V 图像 ------
    best_file_V, min_rms_V = None, float('inf')
    for vf in glob.glob(os.path.join(stokes_V_path, f"*{sbid}*.fits")):
        try:
            with fits.open(vf) as hdul:
                data = np.squeeze(hdul[0].data)
                valid_pixels = data[(data != 0.0) & (~np.isnan(data))]
                if len(valid_pixels) < 100: continue

                _, _, rms = sigma_clipped_stats(valid_pixels, sigma=3.0, maxiters=5)

                if 1e-5 < rms < min_rms_V:
                    min_rms_V, best_file_V = rms, vf
        except:
            continue
    if not best_file_V: continue

    try:
        obs_date = Time(t_min, format='mjd').to_datetime().strftime('%Y.%m.%d')
        with fits.open(best_file_I) as hdul_I:
            flux_I = extract_flux_at_center(hdul_I[0].data, fit_box_size)
        with fits.open(best_file_V) as hdul_V:
            flux_V = extract_flux_at_center(hdul_V[0].data, fit_box_size)

        plot_data.append({
            'SBID': sbid, 'MJD': t_min, 'Date': obs_date,
            'Flux_I': flux_I * 1000, 'Flux_V': flux_V * 1000,
            'RMS_I': min_rms_I * 1000, 'RMS_V': min_rms_V * 1000
        })
    except Exception as e:
        print(f"    ->  提取出错 ({sbid}): {e}")
        continue

if not plot_data: exit(f"️ 未提取到有效数据。")

# 4. 数据筛选
df_plot = pd.DataFrame(plot_data).sort_values(by='MJD').reset_index(drop=True)
v_over_i_raw = (df_plot['Flux_V'] / df_plot['Flux_I']) * 100
v_over_i_abs = np.abs(v_over_i_raw)
rms_vi = v_over_i_abs * np.sqrt(
    (df_plot['RMS_V'] / df_plot['Flux_V']) ** 2 + (df_plot['RMS_I'] / df_plot['Flux_I']) ** 2)
df_plot = df_plot.assign(
    V_over_I_raw=v_over_i_raw,
    V_over_I_abs=v_over_i_abs,
    RMS_VI=rms_vi,
)

good_data_mask = (np.abs(df_plot['Flux_V']) <= df_plot['Flux_I'] * 1.2) & \
                 (df_plot['RMS_I'] < 1.0) & \
                 (df_plot['Flux_I'] >= 3.0 * df_plot['RMS_I'])

df_plot = df_plot.assign(Data_Quality=np.where(good_data_mask, 'Valid', 'Excluded'))
df_valid, df_invalid = df_plot[good_data_mask], df_plot[~good_data_mask]

# 5. 绘制光变曲线
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

title_scale = "(Log Scale)" if use_log_scale else "(Linear Scale)"
title_time = f"(MJD {zoom_mjd_start} - {zoom_mjd_end})" if use_time_zoom else "(All Epochs)"
fig.suptitle(f'Filtered Lightcurve: {target_original_name}\n{title_scale} {title_time}', fontsize=17, fontweight='bold',
             y=1.05)

if not df_valid.empty:
    ax1.fill_between(df_valid['MJD'], df_valid['Flux_I'] - df_valid['RMS_I'], df_valid['Flux_I'] + df_valid['RMS_I'],
                     color=color_I, alpha=err_band_alpha, zorder=2)
    ax1.fill_between(df_valid['MJD'], df_valid['Flux_V'] - df_valid['RMS_V'], df_valid['Flux_V'] + df_valid['RMS_V'],
                     color=color_V, alpha=err_band_alpha * 0.7, zorder=1)
    ax1.plot(df_valid['MJD'], df_valid['Flux_I'], '-o', color=color_I, linewidth=2, markersize=8,
             markeredgecolor='white', label='Stokes I (Valid)', zorder=5)
    ax1.plot(df_valid['MJD'], df_valid['Flux_V'], '-s', color=color_V, linewidth=2, markersize=8,
             markeredgecolor='white', label='Stokes V (Valid)', zorder=6)
    ax2.errorbar(df_valid['MJD'], df_valid['V_over_I_abs'], yerr=df_valid['RMS_VI'], fmt='-^', color=color_VI,
                 ecolor=color_VI, capsize=3, linewidth=2, markersize=9, markeredgecolor='white', label='|V/I| Ratio',
                 zorder=5)

if not df_invalid.empty:
    ax1.errorbar(df_invalid['MJD'], df_invalid['Flux_I'], yerr=df_invalid['RMS_I'], fmt='x', color=color_I,
                 ecolor=color_I, capsize=3, alpha=ghost_alpha, zorder=4)
    ax1.errorbar(df_invalid['MJD'], df_invalid['Flux_V'], yerr=df_invalid['RMS_V'], fmt='x', color=color_V,
                 ecolor=color_V, capsize=3, alpha=ghost_alpha, zorder=4)
    ax1.plot([], [], 'x', color='gray', label='Excluded (Low SNR)')

ax1.axhline(0, color='#757575', linewidth=1.5, linestyle='--')
ax2.axhline(50, color='#9E9E9E', linewidth=1.5, linestyle=':')
ax2.axhline(0, color='#757575', linewidth=1.5, linestyle='--')


# 上图: 流量（可选线性或对数）
if use_log_scale:
    ax1.set_yscale('symlog', linthresh=0.5)
    ax1.yaxis.set_major_locator(SymmetricalLogLocator(linthresh=0.5, base=10))
else:
    ax1.set_yscale('linear')

# 2. 下图: 偏振度比例始终为线性
ax2.set_yscale('linear')

# 下图纵轴范围（不需要可注释）
if not df_plot.empty:
    max_vi = df_plot['V_over_I_abs'].max()
    ax2.set_ylim(-10, 120)

ax1.set_ylabel('Flux Density (mJy)', fontsize=14, fontweight='bold')
ax2.set_ylabel('|V / I| (%)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (MJD)', fontsize=14, fontweight='bold')
ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1, frameon=False)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)

# 动态后缀
time_suffix = f"MJD_{zoom_mjd_start}_to_{zoom_mjd_end}" if use_time_zoom else "FullTime"
scale_suffix = "Log" if use_log_scale else "Linear"
file_suffix = f"_{scale_suffix}_{time_suffix}"

plt.savefig(os.path.join(output_dir, f'{target_safe_name}_Lightcurve{file_suffix}.png'), dpi=300, bbox_inches='tight')
df_plot.to_csv(os.path.join(output_dir, f'{target_safe_name}_Measurements{file_suffix}.csv'), index=False)
plt.close('all')

print(f" 筛选版处理完毕！已保存至: {output_dir} \n (文件名后缀: {file_suffix})")
