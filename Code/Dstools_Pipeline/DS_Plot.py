import os
import re
import matplotlib

# 强制使用后台渲染，防止 Linux 报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dstools.dynamic_spectrum import DynamicSpectrum
from dstools.plotting import plot_ds

# 路径与参数
def project_path(relative_path: str) -> str:
    """自适应项目根目录定位"""
    current = os.path.abspath(os.path.dirname(__file__))
    while not (os.path.isdir(os.path.join(current, 'Code')) and os.path.isdir(os.path.join(current, 'Processed_Data'))):
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.join(os.getcwd(), relative_path)
        current = parent
    return os.path.join(current, relative_path)


# 图像输出的全局根目录
MASTER_OUTPUT_DIR = project_path('Processed_Data/Dynamic_Spectrum')

# 控制面板
# 请在这里填入你要单独出图的那个 .ds 文件的相对/绝对位置
SINGLE_DS_FILE = project_path('Pipeline_Results/GJ_4274/DS_Results/GJ_4274_SB36105_beam28.ds')

# 统一设置绘图界限 (mJy)
V_LIMIT = 18

# 设置平均因子 (相当于官方指令的 -t 和 -f)
T_AVG = 3  # 时间平均因子
F_AVG = 3  # 频率平均因子

# 🎨 核心出图逻辑

def process_ds_file(ds_file, output_dir):
    """处理单个 .ds 文件，利用原生 Python API 画图并分发到专属画廊"""
    basename = os.path.basename(ds_file)

    # 提取源名、SBid和Beam
    match = re.search(r'(.+)_SB(\d+)_beam(\d+)\.ds$', basename, re.IGNORECASE)
    if match:
        hostname = match.group(1)
        sbid = match.group(2)
        beam = match.group(3)
    else:
        # 容错提取
        hostname = basename.replace('.ds', '')
        sb_match = re.search(r'SB(\d+)', basename, re.IGNORECASE)
        beam_match = re.search(r'beam(\d+)', basename, re.IGNORECASE)
        sbid = sb_match.group(1) if sb_match else "UNKNOWN"
        beam = beam_match.group(1) if beam_match else "UNKNOWN"

    # 组合出规范的输出文件前缀
    base_name_str = f"{hostname}_SB{sbid}_beam{beam}"

    #  动态创建专属文件夹
    source_specific_dir = os.path.join(output_dir, hostname)
    os.makedirs(source_specific_dir, exist_ok=True)

    print("-" * 60)
    print(f" 正在处理数据: {basename}")
    print(f" 图像将收纳至: {source_specific_dir}/")
    print(f" 正在加载数据...")

    try:
        ds = DynamicSpectrum(ds_path=ds_file, tavg=T_AVG, favg=F_AVG, trim=True)
    except Exception as e:
        print(f" 数据加载失败，请检查文件: {e}")
        return

    # 任务 1: 渲染 Stokes I
    out_i = os.path.join(source_specific_dir, f"{base_name_str}_StokesI.png")
    print(" 正在渲染 Stokes I...")
    fig_i, ax_i = plot_ds(ds, stokes='I', cmax=V_LIMIT, imag=False)

    # 调整色标和范围
    if len(ax_i.images) > 0:
        im_i = ax_i.images[0]
        im_i.set_cmap('coolwarm')
        im_i.set_clim(-V_LIMIT, V_LIMIT)
    ax_i.set_title(f"ASKAP Dynamic Spectrum (SB{sbid}) - Stokes I", fontsize=14, pad=15)
    plt.savefig(out_i, dpi=300, bbox_inches='tight')
    plt.close(fig_i)
    print(f" Stokes I 已保存: {os.path.basename(out_i)}")

    # 任务 2: 渲染 Stokes V
    out_v = os.path.join(source_specific_dir, f"{base_name_str}_StokesV.png")
    print(" 正在渲染 Stokes V...")
    fig_v, ax_v = plot_ds(ds, stokes='V', cmax=V_LIMIT, imag=False)

    # 调整色标和范围
    if len(ax_v.images) > 0:
        im_v = ax_v.images[0]
        im_v.set_cmap('coolwarm')
        im_v.set_clim(-V_LIMIT, V_LIMIT)

    #  标题去掉了 Range，加上了 SBid
    ax_v.set_title(f"ASKAP Dynamic Spectrum (SB{sbid}) - Stokes V", fontsize=14, pad=15)
    plt.savefig(out_v, dpi=300, bbox_inches='tight')
    plt.close(fig_v)
    print(f" Stokes V 已保存: {os.path.basename(out_v)}")


def main():
    print("  ASKAP 单文件画图管线")

    if not os.path.exists(SINGLE_DS_FILE):
        print(f" 指定的单文件不存在，请检查路径: \n{SINGLE_DS_FILE}")
        return

    # 核心调用
    process_ds_file(SINGLE_DS_FILE, MASTER_OUTPUT_DIR)

    print("\n" + "=" * 60 + f"\n 图像生成完毕！")
    print(f" 请前往画廊查看: {MASTER_OUTPUT_DIR}")


if __name__ == "__main__":
    main()