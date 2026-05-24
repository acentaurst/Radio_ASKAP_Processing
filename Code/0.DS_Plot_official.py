import sys
import os
import shutil
import runpy
import matplotlib

matplotlib.use('Agg')
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


def main():
    # 1. 路径设置
    ds_file = project_path('Pipeline_Results/Proxima_Cen/DS_Results/Proxima_Cen_SB50381_beam33.ds')
    work_dir = os.path.dirname(ds_file)

    print("开始批量生成官方图像 (强制挂载保存引擎)...\n" + "=" * 40)

    # 任务 1: 单独生成 Stokes I 的动态谱
    args_i = ["dstools-plot-ds", "-d", ds_file, "-s", "I", "-t", "3", "-f", "3"]
    out_i = os.path.join(work_dir, "Proxima_Cen_StokesI.png")
    run_official_and_force_save(args_i, out_i)

    # 任务 2: 单独生成 Stokes V 的动态谱
    args_v = ["dstools-plot-ds", "-d", ds_file, "-s", "V", "-t", "3", "-f", "3"]
    out_v = os.path.join(work_dir, "Proxima_Cen_StokesV.png")
    run_official_and_force_save(args_v, out_v)

    # 任务 3: 生成包含光变曲线的综合图 (Stokes IV + Lightcurves)
    args_lc = ["dstools-plot-ds", "-d", ds_file, "-s", "IV", "-l", "-t", "1", "-f", "1"]
    out_lc = os.path.join(work_dir, "Proxima_Cen_Lightcurve.png")
    run_official_and_force_save(args_lc, out_lc)

    print("=" * 40 + "\n 所有图像保存完毕！")


if __name__ == "__main__":
    main()
