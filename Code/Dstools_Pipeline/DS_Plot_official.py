import sys
import os
import shutil
import runpy
import glob
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 📂 路径与参数 (自适应项目根目录机制)
def project_path(relative_path: str) -> str:
    """自适应项目根目录定位"""
    current = os.path.abspath(os.path.dirname(__file__))
    while not (os.path.isdir(os.path.join(current, 'Code')) and os.path.isdir(os.path.join(current, 'Processed_Data'))):
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.join(os.getcwd(), relative_path)
        current = parent
    return os.path.join(current, relative_path)


PIPELINE_RESULTS_DIR = project_path('Pipeline_Results/GJ_4274/DS_Results')
# 图像输出的全局根目录
MASTER_OUTPUT_DIR = project_path('Processed_Data/Dynamic_Spectrum')


# 控制面板

# 批量处理开关
# True: 自动扫描 PIPELINE_RESULTS_DIR 下所有的 .ds 文件并批量出图
# False: 仅处理下方指定的 SINGLE_DS_FILE
BATCH_PROCESS = False

# 如果关闭了批量处理，请在这里填入你要单独出图的那个 .ds 文件的相对位置
SINGLE_DS_FILE = project_path('Pipeline_Results/GJ_4274/DS_Results/GJ_4274_SB80734_beam35.ds')


def run_official_and_force_save(args, output_path):
    print(f" 执行指令: {' '.join(args)}")

    # 拦截官方脚本底层的 plt.show()，强制篡改为 plt.savefig()
    def patched_show(*a, **kw):
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f" 成功拦截内存图像并保存至: {os.path.basename(output_path)}\n")

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
        # 官方脚本跑完通常会 sys.exit(0)，捕获它防止我们的外层脚本中断
        if e.code != 0 and e.code is not None:
            print(f" 官方指令异常退出，退出代码: {e.code}")
    except Exception as e:
        print(f" 运行报错: {e}")
    finally:
        # 恢复案发现场，关闭画板防内存泄漏
        plt.show = original_show
        plt.close('all')


def process_ds_file(ds_file, output_dir):
    """处理单个 .ds 文件并分发到专属画廊的逻辑封装"""
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

    # 组合出规范的输出文件前缀 (包含源、SBid、Beam)
    base_name_str = f"{hostname}_SB{sbid}_beam{beam}"

    # 在 Dynamic_Spectrum 下创建专属文件夹
    source_specific_dir = os.path.join(output_dir, hostname)
    os.makedirs(source_specific_dir, exist_ok=True)

    print("-" * 60)
    print(f"🎯 正在处理数据: {basename}")
    print(f"📂 图像将收纳至: {source_specific_dir}/")

    # 任务 1: 单独生成 Stokes I 的动态谱 (加上 _official 后缀)
    args_i = ["dstools-plot-ds", "-d", ds_file, "-s", "I", "-t", "3", "-f", "3"]
    out_i = os.path.join(source_specific_dir, f"{base_name_str}_StokesI_official.png")
    if not os.path.exists(out_i):
        run_official_and_force_save(args_i, out_i)
    else:
        print(f" ⏭️ {os.path.basename(out_i)} 已存在，自动跳过。")

    # 任务 2: 单独生成 Stokes V 的动态谱 (加上 _official 后缀)
    args_v = ["dstools-plot-ds", "-d", ds_file, "-s", "V", "-t", "3", "-f", "3"]
    out_v = os.path.join(source_specific_dir, f"{base_name_str}_StokesV_official.png")
    if not os.path.exists(out_v):
        run_official_and_force_save(args_v, out_v)
    else:
        print(f" ⏭️ {os.path.basename(out_v)} 已存在，自动跳过。")

    # 任务 3: 生成包含光变曲线的综合图 (加上 _official 后缀)
    args_lc = ["dstools-plot-ds", "-d", ds_file, "-s", "IV", "-l", "-t", "15", "-f", "15"]
    out_lc = os.path.join(source_specific_dir, f"{base_name_str}_Lightcurve_official.png")
    if not os.path.exists(out_lc):
        run_official_and_force_save(args_lc, out_lc)
    else:
        print(f" ⏭️ {os.path.basename(out_lc)} 已存在，自动跳过。")


def main():
    print("======================================================")
    print(" ASKAP 官方画图管线")
    print("======================================================")

    if BATCH_PROCESS:
        print("💡 当前模式：【全量批量出图】")
        # 自动寻找所有的 .ds 成果文件
        ds_files = glob.glob(os.path.join(PIPELINE_RESULTS_DIR, '**', '*.ds'), recursive=True)

        if not ds_files:
            print(f" ❌ 在 {PIPELINE_RESULTS_DIR} 下未找到任何 .ds 文件，请检查路径。")
            return

        print(f"✅ 共发现 {len(ds_files)} 个 .ds 文件，准备出图！\n")

        for ds_file in ds_files:
            process_ds_file(ds_file, MASTER_OUTPUT_DIR)

    else:
        print(" 当前模式：【单ds文件出图】")
        if not os.path.exists(SINGLE_DS_FILE):
            print(f" ❌ 指定的ds文件不存在，请检查路径: \n{SINGLE_DS_FILE}")
            return
        process_ds_file(SINGLE_DS_FILE, MASTER_OUTPUT_DIR)

    print("\n" + "=" * 60 + f"\n🎉 图像生成并分发完毕！")
    print(f"📁 请前往{MASTER_OUTPUT_DIR}查看 ")


if __name__ == "__main__":
    main()