import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
from astropy.time import Time

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

# 1. 全局参数与路径配置
base_dir = project_path('Downloading_Data/fits_images')
time_info_file = project_path('Processed_Data/Catalogue/01.askap_catalogue.csv')
output_dir = project_path('Processed_Data/Fits_Data/Filtered')
os.makedirs(output_dir, exist_ok=True)
print(f"📁 所有结果将保存在: {output_dir}\n")
box_size = 5  # 提取 5x5 的中心区域

# --- 色彩定义区 ---
color_I = '#C8102E'  # Stokes I
color_V = '#0033A0'  # Stokes V
color_VI = '#007749'  # |V/I|
ghost_alpha = 0.35  # 幽灵点的透明度
err_band_alpha = 0.25  # 误差飘带的透明度


# 2. 核心提取函数：中心点高斯拟合 (加入 NaN 过滤与参数边界约束机制)
def extract_flux_at_center(data_2d, box=5):
    data_2d = np.squeeze(data_2d)
    y_center, x_center = data_2d.shape[0] // 2, data_2d.shape[1] // 2
    half_box = box // 2

    cutout = data_2d[y_center - half_box: y_center + half_box + 1,
    x_center - half_box: x_center + half_box + 1]

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


# 3. 批量循环与数据提取
target_list = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
target_list.sort()

print(f" 扫描到 {len(target_list)} 个目标文件夹，开始批量处理...\n" + "=" * 50)

for target_safe_name in target_list:
    target_original_name = target_safe_name.replace('_', ' ')

    stokes_I_path = os.path.join(base_dir, target_safe_name, "StokesI")
    stokes_V_path = os.path.join(base_dir, target_safe_name, "StokesV")

    if not os.path.exists(stokes_I_path) or not os.path.exists(stokes_V_path):
        continue

    all_I_files = glob.glob(os.path.join(stokes_I_path, "*.fits"))
    if not all_I_files:
        continue

    print(f" 正在处理目标: {target_safe_name} ...")

    sbid_dict = {}
    for f in all_I_files:
        match = re.search(r'(ASKAP-\d+)', os.path.basename(f))
        if match:
            sbid = match.group(1)
            sbid_dict.setdefault(sbid, []).append(f)

    plot_data = []

    for sbid, i_file_list in sbid_dict.items():

        # ------ 寻找最佳 Stokes I 图像 ------
        best_file_I = None
        min_rms_I = float('inf')

        for f in i_file_list:
            try:
                with fits.open(f) as hdul:
                    data = np.squeeze(hdul[0].data)
                    valid_pixels = data[(data != 0.0) & (~np.isnan(data))]
                    if len(valid_pixels) < 100: continue

                    # 迭代剔除>3σ的恒星与旁瓣信号，获取真实Local RMS
                    _, _, rms = sigma_clipped_stats(valid_pixels, sigma=3.0, maxiters=5)

                    if 1e-5 < rms < min_rms_I:
                        min_rms_I, best_file_I = rms, f
            except Exception:
                continue

        if best_file_I is None:
            continue

        # ------ 寻找最佳 Stokes V 图像 ------
        file_V_pattern = os.path.join(stokes_V_path, f"*{sbid}*.fits")
        v_file_list = glob.glob(file_V_pattern)

        best_file_V = None
        min_rms_V = float('inf')

        for vf in v_file_list:
            try:
                with fits.open(vf) as hdul:
                    data = np.squeeze(hdul[0].data)
                    valid_pixels = data[(data != 0.0) & (~np.isnan(data))]
                    if len(valid_pixels) < 100: continue

                    _, _, rms = sigma_clipped_stats(valid_pixels, sigma=3.0, maxiters=5)

                    if 1e-5 < rms < min_rms_V:
                        min_rms_V, best_file_V = rms, vf
            except Exception:
                continue

        if best_file_V is None:
            continue

        # ------ 提取数据 ------
        try:
            t_info = pd.read_csv(time_info_file)
            if sbid in t_info['obs_id'].values:
                t_min = t_info[t_info['obs_id'] == sbid].iloc[0]['t_min']
            else:
                t_min = 60000

            obs_date = Time(t_min, format='mjd').to_datetime().strftime('%Y.%m.%d')

            with fits.open(best_file_I) as hdul_I:
                data_I = np.squeeze(hdul_I[0].data)
                flux_I_mJy = extract_flux_at_center(data_I, box=box_size) * 1000

            with fits.open(best_file_V) as hdul_V:
                data_V = np.squeeze(hdul_V[0].data)
                flux_V_mJy = extract_flux_at_center(data_V, box=box_size) * 1000

            plot_data.append({
                'SBID': sbid,
                'MJD': t_min,
                'Date': obs_date,
                'Flux_I': flux_I_mJy,
                'Flux_V': flux_V_mJy,
                'RMS_I': min_rms_I * 1000,
                'RMS_V': min_rms_V * 1000
            })
        except Exception as e:
            continue

    # 4. 数据清洗、误差传递与 CSV 导出
    if not plot_data:
        print(f"  -> ️ {target_safe_name} 未提取到有效数据，跳过画图。")
        continue

    df_plot = pd.DataFrame(plot_data)
    df_plot = df_plot.sort_values(by='MJD').reset_index(drop=True)

    v_over_i_raw = (df_plot['Flux_V'] / df_plot['Flux_I']) * 100
    v_over_i_abs = np.abs(v_over_i_raw)

    # 使用误差传递公式计算 |V/I| 的误差百分比 (RMS_VI)
    rms_vi = v_over_i_abs * np.sqrt(
        (df_plot['RMS_V'] / df_plot['Flux_V']) ** 2 +
        (df_plot['RMS_I'] / df_plot['Flux_I']) ** 2
    )
    df_plot = df_plot.assign(
        V_over_I_raw=v_over_i_raw,
        V_over_I_abs=v_over_i_abs,
        RMS_VI=rms_vi,
    )

    valid_physics_mask = np.abs(df_plot['Flux_V']) <= (df_plot['Flux_I'] * 1.2)
    valid_rms_mask = df_plot['RMS_I'] < 1.0
    valid_snr_mask = df_plot['Flux_I'] >= (3.0 * df_plot['RMS_I'])

    good_data_mask = valid_physics_mask & valid_rms_mask & valid_snr_mask

    df_plot = df_plot.assign(Data_Quality=np.where(good_data_mask, 'Valid', 'Excluded'))
    df_valid = df_plot[good_data_mask].copy()
    df_invalid = df_plot[~good_data_mask].copy()

    # 导出 CSV 包含了新增的 RMS_VI
    csv_columns = ['SBID', 'MJD', 'Date', 'Flux_I', 'Flux_V', 'V_over_I_raw', 'V_over_I_abs', 'RMS_I', 'RMS_V',
                   'RMS_VI', 'Data_Quality']
    csv_filename = f'{target_safe_name}_Measurements.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    df_plot[csv_columns].to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 5. 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    fig.suptitle(f'Radio Lightcurve of {target_original_name} (ASKAP)',
                 fontsize=18, fontweight='bold', y=1.06, fontfamily='serif')

    # 图 1: Stokes I 和 Stokes V
    # A. 绘制【未剔除数据】：半透明飘带 + 实线连点
    if not df_valid.empty:
        # Stokes I (红飘带)
        ax1.fill_between(df_valid['MJD'],
                         df_valid['Flux_I'] - df_valid['RMS_I'],
                         df_valid['Flux_I'] + df_valid['RMS_I'],
                         color=color_I, alpha=err_band_alpha, edgecolor='none', zorder=2)
        # Stokes V (蓝飘带)
        ax1.fill_between(df_valid['MJD'],
                         df_valid['Flux_V'] - df_valid['RMS_V'],
                         df_valid['Flux_V'] + df_valid['RMS_V'],
                         color=color_V, alpha=err_band_alpha * 0.7, edgecolor='none', zorder=1)

        # 正常连线 (去掉了点上的十字误差棒)
        ax1.plot(df_valid['MJD'], df_valid['Flux_I'], '-o', color=color_I, linewidth=2, markersize=8,
                 markeredgecolor='white', markeredgewidth=1.5, label='Stokes I (Valid)', zorder=5)
        ax1.plot(df_valid['MJD'], df_valid['Flux_V'], '-s', color=color_V, linewidth=2, markersize=8,
                 markeredgecolor='white', markeredgewidth=1.5, label='Stokes V (Valid)', zorder=6)

    # B. 绘制【被剔除数据】：只画独立的十字误差棒 (不连线，不画飘带)
    if not df_invalid.empty:
        # 剔除的 Stokes I (带帽误差棒)
        ax1.errorbar(df_invalid['MJD'], df_invalid['Flux_I'], yerr=df_invalid['RMS_I'],
                     fmt='x', color=color_I, ecolor=color_I, elinewidth=1.5, capsize=3,
                     markersize=8, alpha=ghost_alpha, zorder=4)
        # 剔除的 Stokes V (带帽误差棒)
        ax1.errorbar(df_invalid['MJD'], df_invalid['Flux_V'], yerr=df_invalid['RMS_V'],
                     fmt='x', color=color_V, ecolor=color_V, elinewidth=1.5, capsize=3,
                     markersize=8, alpha=ghost_alpha, zorder=4)

        ax1.plot([], [], 'x', color='gray', markersize=8, label='Excluded (Low SNR)')

    ax1.axhline(0, color='#757575', linewidth=1.5, linestyle='--', alpha=0.8, zorder=1)
    ax1.set_ylabel('Flux Density (mJy)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=11, frameon=False)

    # 图 2: |V / I| (%) 绝对圆偏振率
    # A. 绘制【未剔除的 V/I】：使用带帽误差棒 + 连线
    if not df_valid.empty:
        ax2.errorbar(df_valid['MJD'], df_valid['V_over_I_abs'], yerr=df_valid['RMS_VI'],
                     fmt='-^', color=color_VI, ecolor=color_VI, elinewidth=1.5, capsize=3,
                     linewidth=2, markersize=9, markeredgecolor='white', markeredgewidth=1.5,
                     label='|V/I| Ratio', zorder=5)

    ax2.axhline(50, color='#9E9E9E', linewidth=1.5, linestyle=':', alpha=0.9, zorder=1, label='50% Polarization')
    ax2.axhline(0, color='#757575', linewidth=1.5, linestyle='--', alpha=0.8, zorder=1)

    ax2.set_ylabel('|V / I| (%)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Time (MJD)', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=11, frameon=False)

    # 全局细节美化
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', linewidth=1, alpha=0.6, color='#CFD8DC', zorder=0)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.5, length=6, color='#424242')
        ax.get_xaxis().get_major_formatter().set_useOffset(False)

    img_filename = f'{target_safe_name}_Lightcurve.png'
    img_path = os.path.join(output_dir, img_filename)
    plt.savefig(img_path, dpi=300, bbox_inches='tight', facecolor='white')

    plt.close(fig)
    print(f"  ->  完成: {img_filename} 和 {csv_filename}")
