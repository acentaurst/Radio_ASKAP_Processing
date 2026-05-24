import os
import glob
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


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

# 单位设置
FLUX_UNIT = 'mJy'

# 路径设置
official_csv_path = project_path('Processed_Data/Catalogue/02.final_confirmed_stars_direct_1.csv')
measurements_dir = project_path('Processed_Data/Fits_Data/Unfiltered')

output_csv = project_path('Processed_Data/Catalogue/04.Flux_Comparison_Result.csv')
output_plot_2d = project_path('Processed_Data/Catalogue/04.Flux_2D_Scatter.png')

OFFICIAL_STAR_NAME_COL = 'hostname'
OFFICIAL_SBID_COL = 'sbid_clean'
OFFICIAL_FLUX_COL = 'col_flux_peak'

MEASURED_SBID_COL = 'SBID'
MEASURED_FLUX_COL = 'Flux_I'


def extract_5digit_sbid(sbid_str):
    match = re.search(r'(\d{5})', str(sbid_str))
    if match:
        return match.group(1)
    return None


def load_official_catalogue():
    df_official = pd.read_csv(official_csv_path).copy()
    return df_official.assign(
        Cleaned_Source_Name=df_official[OFFICIAL_STAR_NAME_COL].astype(str).str.replace(' ', '_'),
        Match_SBID=df_official[OFFICIAL_SBID_COL].map(extract_5digit_sbid),
    )


def load_measurements():
    measured_frames = []
    pattern = os.path.join(measurements_dir, '*_Measurements_Unfiltered.csv')

    for file_path in glob.glob(pattern):
        source_name = os.path.basename(file_path).replace('_Measurements_Unfiltered.csv', '')
        df_measured = pd.read_csv(file_path)

        if MEASURED_SBID_COL not in df_measured.columns or MEASURED_FLUX_COL not in df_measured.columns:
            continue

        cleaned_df = (
            df_measured[[MEASURED_SBID_COL, MEASURED_FLUX_COL]]
            .copy()
            .assign(
                Cleaned_Source_Name=source_name,
                Match_SBID=lambda df: df[MEASURED_SBID_COL].map(extract_5digit_sbid),
                My_Flux_I=lambda df: pd.to_numeric(df[MEASURED_FLUX_COL], errors='coerce'),
            )
            .dropna(subset=['Match_SBID'])
            [['Cleaned_Source_Name', 'Match_SBID', 'My_Flux_I']]
        )
        measured_frames.append(cleaned_df)

    if not measured_frames:
        return pd.DataFrame(columns=['Cleaned_Source_Name', 'Match_SBID', 'My_Flux_I'])

    return pd.concat(measured_frames, ignore_index=True)


def main():
    print("正在加载和清理数据...")
    try:
        df_official = load_official_catalogue()
    except FileNotFoundError:
        print(f"找不到官方文件: {official_csv_path}")
        return

    df_my_measurements = load_measurements()

    print("正在进行数据交叉比对...")
    df_result = pd.merge(
        df_official,
        df_my_measurements,
        on=['Cleaned_Source_Name', 'Match_SBID'],
        how='left'
    ).copy()

    official_flux = pd.to_numeric(df_result[OFFICIAL_FLUX_COL], errors='coerce')
    measured_flux = pd.to_numeric(df_result['My_Flux_I'], errors='coerce')
    flux_diff = measured_flux - official_flux
    safe_official_flux = official_flux.replace(0, np.nan)

    df_result = df_result.assign(
        **{
            OFFICIAL_FLUX_COL: official_flux,
            'My_Flux_I': measured_flux,
            'Flux_Diff': flux_diff,
            'Relative_Error(%)': (flux_diff / safe_official_flux) * 100,
            'Measurement_Status': np.where(measured_flux.isna(), 'Missing', 'Matched'),
        }
    )

    if df_result.empty:
        print("\n没有可导出的结果。")
        return

    cols_to_export = [
        OFFICIAL_STAR_NAME_COL, 'Cleaned_Source_Name', OFFICIAL_SBID_COL, 'Match_SBID',
        OFFICIAL_FLUX_COL, 'My_Flux_I', 'Flux_Diff', 'Relative_Error(%)', 'Measurement_Status'
    ]
    final_output = df_result.loc[:, cols_to_export].copy()
    final_output.to_csv(output_csv, index=False)

    # 绘图：二维残差散点图
    matched_data = final_output[
        final_output['Measurement_Status'].eq('Matched')
    ].dropna(subset=[OFFICIAL_FLUX_COL, 'My_Flux_I'])

    if matched_data.empty:
        print("\n没有有效的匹配数据来生成散点图！")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x_data = matched_data[OFFICIAL_FLUX_COL]
    diff_data = matched_data['Flux_Diff']
    rel_err_data = matched_data['Relative_Error(%)']

    # 图1：绝对通量差值 vs 官方通量
    ax1.scatter(x_data, diff_data, alpha=0.7, edgecolors='k', color='royalblue')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.set_title('Absolute Flux Difference vs. Official Flux')
    ax1.set_xlabel(f'Official Flux Peak ({FLUX_UNIT})')
    ax1.set_ylabel(f'Flux Difference: Mine - Official ({FLUX_UNIT})')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend()

    # 图2：相对误差百分比 vs 官方通量
    ax2.scatter(x_data, rel_err_data, alpha=0.7, edgecolors='k', color='darkorange')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_title('Relative Error (%) vs. Official Flux')
    ax2.set_xlabel(f'Official Flux Peak ({FLUX_UNIT})')
    ax2.set_ylabel('Relative Error (%)')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()

    plt.suptitle(f'ASKAP Photometry Validation (Unit: {FLUX_UNIT})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_plot_2d, bbox_inches='tight')
    plt.close()

    print(f"\n================ 误差分析完成 ================")
    print(f"成功匹配的观测数量: {len(matched_data)}")
    print(f"二维残差散点图已生成并保存至: {output_plot_2d}")


if __name__ == "__main__":
    main()
