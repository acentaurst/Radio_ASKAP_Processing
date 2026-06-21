import os
import glob

# 填写管线结果主目录
PIPELINE_RESULTS_BASE = "/mnt/home/hst/project/ASKAP_Stellar_with_Exoplanet_Serverbin/Result/DS"


def smart_clean_wsclean_models(results_dir):
    print("🧹 启动智能清理：基于 .ds 状态判断是否粉碎中间文件...")
    total_deleted_files = 0
    total_size_freed = 0

    workspaces = glob.glob(os.path.join(results_dir, "*", "*_workspace"))

    if not workspaces:
        print("未找到任何 workspace 文件夹。")
        return

    for ws in workspaces:
        ws_name = os.path.basename(ws)

        # 1. 拼接预期的最终 .ds 文件路径
        base_name = ws_name.replace("_workspace", "")
        expected_ds_name = f"{base_name}.ds"
        source_dir = os.path.dirname(ws)
        ds_path = os.path.join(source_dir, "DS_Results", expected_ds_name)

        # 2. 状态判定：若无 .ds 成果，说明未跑完，直接跳过保护断点
        if not os.path.exists(ds_path):
            print(f"⏸️  [跳过保护] {ws_name} (.ds 未生成)")
            continue

        print(f"🟢 [安全清理] {ws_name} (成果已存在，清理冗余 FITS)")

        # 3. 执行清理：仅删除 wsclean_model 下非 MFS 的 fits 文件
        model_dirs = glob.glob(os.path.join(ws, "wsclean_model*"))
        for model_dir in model_dirs:
            for file_path in glob.glob(os.path.join(model_dir, "*.fits")):
                filename = os.path.basename(file_path)

                if "MFS" not in filename:
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        total_deleted_files += 1
                        total_size_freed += file_size
                    except Exception:
                        pass

                        # 4. 统计与输出
    freed_gb = total_size_freed / (1024 ** 3)
    print("\n" + "=" * 50)
    print(f"✅ 清理完成！")
    print(f"💣 删除分通道 FITS 文件: {total_deleted_files} 个")
    print(f"🚀 释放存储空间: {freed_gb:.2f} GB")
    print("=" * 50)


if __name__ == "__main__":
    smart_clean_wsclean_models(PIPELINE_RESULTS_BASE)