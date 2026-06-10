import sys
import os
import shutil
import runpy
import re
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


# 管线产生 .ds 成果的目录
PIPELINE_RESULTS_DIR = project_path('Pipeline_Results/2MASS_J01033563-5515561_A/DS_Results')

# 图像输出的全局根目录 (按源建立专属画廊)
MASTER_OUTPUT_DIR = project_path('Processed_Data/Dynamic_Spectrum')


# 控制面板
# 批量处理开关
# True: 自动扫描 PIPELINE_RESULTS_DIR 下所有的 .ds 文件并批量出图
# False: 仅处理下方指定的 SINGLE_DS_FILE
BATCH_PROCESS = False

# 如果关闭了批量处理 (设为False)，请在这里填入你要单独出图的那个 .ds 文件的相对/绝对位置
SINGLE_DS_FILE = project_path('Pipeline_Results/2MASS_J01033563-5515561_A/DS_Results/2MASS_J01033563-5515561_A_SB84261_beam12.ds')


def run_official_and_force_save(args, output_prefix):
    print(f" 执行指令: {' '.join(args)}")

    # 拦截官方脚本底层的 plt.show()，并捕获所有生成的独立画板
    def patched_show(*a, **kw):
        fignums = plt.get_fignums()

        if not fignums:
            print(" ⚠️ 警告：官方指令未生成任何图像。")
            return

        print(f"👀 发现官方指令在内存中生成了 {len(fignums)} 张画板，正在自动匹配命名...")

        # 按照官方出图的物理顺序进行精准命名（已去除 _official 后缀）
        for i, fignum in enumerate(fignums):
            plt.figure(fignum)  # 激活切换到对应的画板

            if i == 0:
                suffix = "_StokesI_Official.png"  # 第1张：Stokes I 动态谱
            elif i == 1:
                suffix = "_StokesV_Official.png"  # 第2张：Stokes V 动态谱
            elif i == 2:
                suffix = "_Lightcurve_Official.png"  # 第3张：光变曲线
            else:
                suffix = f"_ExtraPart{i + 1}.png"

            current_out = output_prefix + suffix
            plt.savefig(current_out, dpi=300, bbox_inches='tight')
            print(f"✅ 成功提取并覆盖保存: {os.path.basename(current_out)}")

    original_show = plt.show
    plt.show = patched_show

    # 获取环境变量中官方指令的真实路径
    cli_path = shutil.which("dstools-plot-ds")
    if not cli_path:
        print(" 找不到 dstools-plot-ds 指令，请确认 conda 环境已激活。")
        return

    # 伪装命令行输入
    sys.argv = args

    try:
        # 在当前 Python 进程中直接运行官方 CLI 脚本
        runpy.run_path(cli_path, run_name='__main__')
    except SystemExit as e:
        if e.code != 0 and e.code is not None:
            print(f" 官方指令异常退出，退出代码: {e.code}")
    except Exception as e:
        print(f" 运行报错: {e}")
    finally:
        # 恢复案发现场，关闭画板防内存泄漏
        plt.show = original_show
        plt.close('all')


def process_ds_file(ds_file, output_dir):
    """处理单个 .ds 文件并生成 [动态谱 + 光变曲线] 综合图"""
    basename = os.path.basename(ds_file)

    # 提取源名、SBid和Beam
    match = re.search(r'(.+)_SB(\d+)_beam(\d+)\.ds$', basename, re.IGNORECASE)
    if match:
        hostname = match.group(1)
        sbid = match.group(2)
        beam = match.group(3)
    else:
        hostname = basename.replace('.ds', '')
        sb_match = re.search(r'SB(\d+)', basename, re.IGNORECASE)
        beam_match = re.search(r'beam(\d+)', basename, re.IGNORECASE)
        sbid = sb_match.group(1) if sb_match else "UNKNOWN"
        beam = beam_match.group(1) if beam_match else "UNKNOWN"

    base_name_str = f"{hostname}_SB{sbid}_beam{beam}"

    # 在 Dynamic_Spectrum 下动态创建该恒星的专属文件夹
    source_specific_dir = os.path.join(output_dir, hostname)
    os.makedirs(source_specific_dir, exist_ok=True)

    print("-" * 60)
    print(f"🎯 正在处理数据: {basename}")
    print(f"📂 图像将收纳至: {source_specific_dir}/")

    # 指令区 (带 -l 开关，直接生成并覆盖)
    # [大图模式] Stokes I 和 V 的动态谱 + 它们的光变曲线
    args_combined = [
        "dstools-plot-ds", "-d", ds_file,
        "-s", "IV", "-l",
        "-t", "30", "-f", "3",
        "-I", "2",  # 控制 Stokes I 的辐射通量范围上限
        "-V", "2"  # 控制 Stokes V 的辐射通量范围上限
    ]
    # 名字（源_SBid_beam）：
    out_prefix = os.path.join(source_specific_dir, base_name_str)

    print("📊 开始渲染综合图 (Stokes I/V 动态谱 + 光变曲线)...")
    run_official_and_force_save(args_combined, out_prefix)


def main():
    print("======================================================")
    print(" 🎨 ASKAP 官方画图管线")
    print("======================================================")

    if BATCH_PROCESS:
        print("💡 当前模式：【全量批量出图】")
        # 自动深层寻找所有的 .ds 成果文件
        ds_files = glob.glob(os.path.join(PIPELINE_RESULTS_DIR, '**', '*.ds'), recursive=True)

        if not ds_files:
            print(f" ❌ 在 {PIPELINE_RESULTS_DIR} 下未找到任何 .ds 文件，请检查路径。")
            return

        print(f"✅ 共发现 {len(ds_files)} 个 .ds 文件，准备出图！\n")

        # 开始遍历并处理每一个 .ds 文件
        for ds_file in ds_files:
            process_ds_file(ds_file, MASTER_OUTPUT_DIR)

    else:
        print("💡 当前模式：【单文件出图】")
        if not os.path.exists(SINGLE_DS_FILE):
            print(f" ❌ 指定的单文件不存在，请检查路径: \n{SINGLE_DS_FILE}")
            return

        process_ds_file(SINGLE_DS_FILE, MASTER_OUTPUT_DIR)

    print("\n" + "=" * 60 + f"\n🎉 图像生成完毕！")
    print(f"📁 请前往 {MASTER_OUTPUT_DIR} 查看")


if __name__ == "__main__":
    main()