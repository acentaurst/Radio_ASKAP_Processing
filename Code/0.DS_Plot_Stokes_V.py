import matplotlib
# 强制使用后台渲染，防止 Linux 报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import os

from dstools.dynamic_spectrum import DynamicSpectrum
from dstools.plotting import plot_ds


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

def main():
    # 1. 设置文件路径 
    ds_file = project_path('Pipeline_Results/Proxima_Cen/DS_Results/Proxima_Cen_SB50381_beam33.ds')
    
    print(f" 正在加载数据: {ds_file}")
    
    # 2. 加载数据
    ds = DynamicSpectrum(ds_path=ds_file, tavg=3, favg=3, trim=True)

    print(" 数据加载完成，正在渲染 Stokes V...")
    v_limit = 20

    # 3. 绘制 Stokes V
    fig, ax = plot_ds(ds, stokes='V', cmax=v_limit, imag=False)

    # 获取刚刚画出来的那张动态谱图
    if len(ax.images) > 0:
        im = ax.images[0]
        # 强制设置色标为红蓝发散色 (RdBu_r)
        im.set_cmap('coolwarm')
        # 强制设置颜色显示范围为绝对对称 (-5 到 +5)
        im.set_clim(-v_limit, v_limit)

    # 4. 标题
    ax.set_title(f"ASKAP Dynamic Spectrum - Stokes V (Range: +/- {v_limit} mJy)", fontsize=14, pad=15)
    
    # 5. 保存
    output_image = project_path('Downloading_Data/ms_data/Proxima_Cen/Proxima_Cen_V_50381.png')
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    
    print(f" Stokes V 图像已保存至: {output_image}")

if __name__ == "__main__":
    main()
