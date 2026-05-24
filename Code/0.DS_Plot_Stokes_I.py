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
    ds_file = project_path('Downloading_Data/ms_data/Proxima_Cen/Proxima_Cen_50381_r10.ds')

    print(f"⏳ 正在加载数据: {ds_file}")

    # 2. 加载数据
    # Stokes V 的信号通常比 Stokes I 弱很多，建议适当增加平均因子(tavg/favg)以压低噪声
    # trim=True 会切除频带边缘的无效数据
    ds = DynamicSpectrum(ds_path=ds_file, tavg=3, favg=3, trim=True)

    print("📊 数据加载完成，正在渲染 Stokes I...")

    # ==========================
    # 核心修改区域
    # ==========================

    # 设置显示范围 (单位: mJy)
    # Stokes V 信号通常很弱。如果你的 Stokes I 峰值是 100 mJy，Stokes V 可能只有 5-10 mJy。
    # 这里我们设为 +/- 5 mJy。如果图全白，把它改小；如果爆掉（全红/全蓝），把它改大。
    v_limit = 20

    # 3. 绘制 Stokes I
    # stokes='I' : 指定画圆极化
    # cmap='RdBu_r' : 使用"红-白-蓝"发散色阶 (Red-Blue reversed)。通常 蓝=正, 红=负
    # cmin, cmax : 强制对称，确保 0 值（无极化）是白色的
    fig, ax = plot_ds(ds, stokes='I', cmax=v_limit, imag=False)

    # 获取刚刚画出来的那张动态谱图
    if len(ax.images) > 0:
        im = ax.images[0]
        # 强制设置色标为红蓝发散色 (RdBu_r)
        im.set_cmap('coolwarm')
        # 强制设置颜色显示范围为绝对对称 (-5 到 +5)
        im.set_clim(-v_limit, v_limit)

    # 4. 标题与标注
    ax.set_title(f"ASKAP Dynamic Spectrum - Stokes I (Range: +/- {v_limit} mJy)", fontsize=14, pad=15)

    # 5. 保存图片
    output_image = project_path('Downloading_Data/ms_data/Proxima_Cen/Proxima_Cen_I_50381.png')
    plt.savefig(output_image, dpi=300, bbox_inches='tight')

    print(f"✅ Stokes I 图像已保存至: {output_image}")


if __name__ == "__main__":
    main()
