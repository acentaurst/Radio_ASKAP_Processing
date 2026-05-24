import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import mad_std, sigma_clipped_stats
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

# 统一存储所有输出结果的文件夹
output_dir = project_path('Processed_Data/Fits_Data/Unfiltered')
os.makedirs(output_dir, exist_ok=True)
print(f" 批量无筛选结果将保存在: {output_dir}\n")

# 提取参数
fit_box_size = 5  # 拟合流量
visual_box_size = 31  # 画廊切片宽度

# --- 色彩定义区 ---
color_I = '#C8102E'
color_V = '#0033A0'
color_VI = '#007749'
err_band_alpha = 0.25


# 2. 2D高斯拟合提取数据与图像切片
def extract_flux_and_cutout(data_2d, fit_box=5, visual_box=31):
    data_2d = np.squeeze(data_2d)
    y_center, x_center = data_2d.shape[0] // 2, data_2d.shape[1] // 2

    # 提取用于画廊切片的大框 (Visual Cutout)
    half_vbox = visual_box // 2
    cutout_visual = data_2d[y_center - half_vbox: y_center + half_vbox + 1,
    x_center - half_vbox: x_center + half_vbox + 1]

    # 提取用于2D高斯拟合的小框 (Fit Cutout)
    half_fbox = fit_box // 2
    cutout_fit = data_2d[y_center - half_fbox: y_center + half_fbox + 1,
    x_center - half_fbox: x_center + half_fbox + 1]

    # 如果整个提取框全都是 NaN，说明目标在图像外，直接返回无效值
    if np.all(np.isnan(cutout_fit)):
        return np.nan, cutout_visual

    y_grid, x_grid = np.mgrid[:fit_box, :fit_box]

    # 获取局部像素的真实波动范围
    max_val = np.nanmax(cutout_fit)
    abs_max = np.nanmax(np.abs(cutout_fit))

    g_init = models.Gaussian2D(amplitude=max_val, x_mean=half_fbox, y_mean=half_fbox, x_stddev=1, y_stddev=1)

    # 约束条件
    g_init.x_mean.bounds = (0, fit_box - 1)
    g_init.y_mean.bounds = (0, fit_box - 1)
    g_init.x_stddev.bounds = (0.5, 3.0)
    g_init.y_stddev.bounds = (0.5, 3.0)
    g_init.amplitude.bounds = (-abs_max * 2, abs_max * 2)

    fitter = fitting.LevMarLSQFitter()

    # 添加 filter_non_finite=True，让 Astropy 自动忽略 NaN 和 Inf 像素进行拟合
    g_fit = fitter(g_init, x_grid, y_grid, cutout_fit, filter_non_finite=True)

    # 返回拟合峰作为通量，以及用于画图的大框数据
    return g_fit.amplitude.value, cutout_visual


# 3. 批量遍历所有源目录
target_list = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
target_list.sort()

print(f" 扫描到 {len(target_list)} 个目标文件夹，开始批量处理...\n" + "=" * 50)

# 提前读取星表时间
t_info = pd.read_csv(time_info_file)

for target_safe_name in target_list:
    target_original_name = target_safe_name.replace('_', ' ')

    stokes_I_path = os.path.join(base_dir, target_safe_name, "StokesI")
    stokes_V_path = os.path.join(base_dir, target_safe_name, "StokesV")

    if not os.path.exists(stokes_I_path) or not os.path.exists(stokes_V_path):
        print(f" 跳过 {target_safe_name}: 缺少必要文件夹。")
        continue

    all_I_files = glob.glob(os.path.join(stokes_I_path, "*.fits"))
    if not all_I_files:
        continue

    print(f" 正在处理目标: {target_safe_name} ...")

    sbid_dict = {}
    for f in all_I_files:
        match = re.search(r'(ASKAP-\d+)', os.path.basename(f))
        if match:
            sbid_dict.setdefault(match.group(1), []).append(f)

    plot_data = []

    for sbid, i_file_list in sbid_dict.items():
        if sbid in t_info['obs_id'].values:
            t_min = t_info[t_info['obs_id'] == sbid].iloc[0]['t_min']
        else:
            t_min = 60000

        # ------ 寻找最佳 Stokes I 图像 ------
        best_file_I, min_rms_I = None, float('inf')
        for f in i_file_list:
            try:
                with fits.open(f) as hdul:
                    data = np.squeeze(hdul[0].data)

                    # 剔除无效零值
                    valid_pixels = data[(data != 0.0) & (~np.isnan(data))]
                    if len(valid_pixels) < 100: continue

                    #  条件一 (Sigma Clipping)：迭代剔除>3σ的恒星与旁瓣信号，获取Local RMS
                    _, _, rms = sigma_clipped_stats(valid_pixels, sigma=3.0, maxiters=5)

                    #  条件二 (物理底线)：假设ASKAP真实RMS不能低于 0.01 mJy (1e-5)
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

        # 提取数据
        try:
            obs_date = Time(t_min, format='mjd').to_datetime().strftime('%Y.%m.%d')

            with fits.open(best_file_I) as hdul_I:
                data_I = np.squeeze(hdul_I[0].data)
                flux_I_mJy, cutout_I = extract_flux_and_cutout(data_I, fit_box_size, visual_box_size)
                flux_I_mJy *= 1000

            with fits.open(best_file_V) as hdul_V:
                data_V = np.squeeze(hdul_V[0].data)
                flux_V_mJy, cutout_V = extract_flux_and_cutout(data_V, fit_box_size, visual_box_size)
                flux_V_mJy *= 1000

            plot_data.append({
                'SBID': sbid, 'MJD': t_min, 'Date': obs_date,
                'Flux_I': flux_I_mJy, 'Flux_V': flux_V_mJy,
                'RMS_I': min_rms_I * 1000, 'RMS_V': min_rms_V * 1000,
                'Cutout_I': cutout_I * 1000, 'Cutout_V': cutout_V * 1000
            })
        except Exception as e:
            # 修改点：将错误信息显式打印出来，方便排查后续其他异常
            print(f"    ->  提取数据出错 ({sbid}): {e}")
            continue

    if not plot_data:
        print(f"  -> ✖️ {target_safe_name} 未提取到有效数据，跳过。")
        continue

    # 4. 数据整理与导出
    df_plot = pd.DataFrame(plot_data)
    df_plot = df_plot.sort_values(by='MJD').reset_index(drop=True)

    v_over_i_raw = (df_plot['Flux_V'] / df_plot['Flux_I']) * 100
    v_over_i_abs = np.abs(v_over_i_raw)
    rms_vi = v_over_i_abs * np.sqrt(
        (df_plot['RMS_V'] / df_plot['Flux_V']) ** 2 + (df_plot['RMS_I'] / df_plot['Flux_I']) ** 2)
    df_plot = df_plot.assign(
        V_over_I_raw=v_over_i_raw,
        V_over_I_abs=v_over_i_abs,
        RMS_VI=rms_vi,
        Data_Quality='Valid',  # 全部不过滤，统统视为有效
    )

    # 导出 CSV
    csv_cols = ['SBID', 'MJD', 'Date', 'Flux_I', 'Flux_V', 'V_over_I_raw', 'V_over_I_abs', 'RMS_I', 'RMS_V', 'RMS_VI']
    csv_filename = f'{target_safe_name}_Measurements_Unfiltered.csv'
    df_plot[csv_cols].to_csv(os.path.join(output_dir, csv_filename), index=False, encoding='utf-8-sig')

    # 5. 绘制光变曲线 (纯误差棒版本)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f'Unfiltered Lightcurve of {target_original_name}', fontsize=19, fontweight='bold', y=1.06,
                 fontfamily='serif')

    # --- 图 1: Stokes I 和 V  ---
    # Stokes I (红线 + 圆点 + 误差棒)
    ax1.errorbar(df_plot['MJD'], df_plot['Flux_I'], yerr=df_plot['RMS_I'],
                 fmt='-o', color=color_I, ecolor=color_I, elinewidth=1.5, capsize=3,
                 linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                 label='Stokes I (All)')

    # Stokes V (蓝线 + 方块 + 误差棒)
    ax1.errorbar(df_plot['MJD'], df_plot['Flux_V'], yerr=df_plot['RMS_V'],
                 fmt='-s', color=color_V, ecolor=color_V, elinewidth=1.5, capsize=3,
                 linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                 label='Stokes V (All)')

    ax1.axhline(0, color='#757575', linewidth=1.5, linestyle='--')
    ax1.set_ylabel('Flux Density (mJy)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=12, frameon=False)

    # --- 图 2: V/I Ratio (保持误差棒) ---
    ax2.errorbar(df_plot['MJD'], df_plot['V_over_I_abs'], yerr=df_plot['RMS_VI'],
                 fmt='-^', color=color_VI, ecolor=color_VI, elinewidth=1.5, capsize=3,
                 linewidth=2, markersize=9, markeredgecolor='white', markeredgewidth=1.5,
                 label='|V/I| Ratio')

    ax2.axhline(50, color='#9E9E9E', linewidth=1.5, linestyle=':')
    ax2.axhline(0, color='#757575', linewidth=1.5, linestyle='--')
    ax2.set_ylabel('|V / I| (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (MJD)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1, fontsize=12, frameon=False)

    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', linewidth=1, alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().get_major_formatter().set_useOffset(False)

    lc_path = os.path.join(output_dir, f'{target_safe_name}_Lightcurve_Unfiltered.png')
    plt.savefig(lc_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # 6. 生成 Cutout 画廊
    num_epochs = len(df_plot)
    # 控制长图高度，防止数据太多时图片过长崩溃，单行 3.5 英寸
    fig_cut, axes_cut = plt.subplots(num_epochs, 2, figsize=(10, 3.5 * num_epochs))
    if num_epochs == 1:
        axes_cut = np.array([axes_cut])

    for idx, row in df_plot.iterrows():
        ax_I, ax_V = axes_cut[idx, 0], axes_cut[idx, 1]

        # 动态范围设定：以噪声水平为基准，让肉眼更容易分辨信号
        im_I = ax_I.imshow(row['Cutout_I'], origin='lower', cmap='magma', vmin=-2 * row['RMS_I'], vmax=5 * row['RMS_I'])
        im_V = ax_V.imshow(row['Cutout_V'], origin='lower', cmap='RdBu_r', vmin=-4 * row['RMS_V'],
                           vmax=4 * row['RMS_V'])

        center = visual_box_size // 2
        for ax in [ax_I, ax_V]:
            ax.axhline(center, color='white', linestyle='--', alpha=0.4)
            ax.axvline(center, color='white', linestyle='--', alpha=0.4)
            ax.set_xticks([])
            ax.set_yticks([])

        snr_I = row['Flux_I'] / row['RMS_I']
        snr_V = row['Flux_V'] / row['RMS_V']

        ax_I.set_title(
            f"SBID: {row['SBID']} | MJD: {row['MJD']:.2f}\nStokes I Flux: {row['Flux_I']:.2f} mJy (SNR: {snr_I:.1f})",
            fontsize=11)
        ax_V.set_title(f" \nStokes V Flux: {row['Flux_V']:.2f} mJy (SNR: {snr_V:.1f})", fontsize=11)

        fig_cut.colorbar(im_I, ax=ax_I, fraction=0.046, pad=0.04)
        fig_cut.colorbar(im_V, ax=ax_V, fraction=0.046, pad=0.04)

    fig_cut.suptitle(f'{target_original_name} Visual Cutouts', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()

    cut_path = os.path.join(output_dir, f'{target_safe_name}_Cutouts.png')
    plt.savefig(cut_path, dpi=200, bbox_inches='tight', facecolor='white')

    # 内存释放
    plt.close('all')

    print(f"  -> 完成: {target_safe_name}")

print("\n 全部源无筛选批量处理完成！请查看文件夹:", output_dir)
