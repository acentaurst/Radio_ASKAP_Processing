import numpy as np
import pandas as pd
from astroquery.casda import Casda
from astroquery.utils.tap.core import TapPlus
import os
import sys
import time


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

# 0.USER CONFIGURATION
OPAL_USER = "acentauri_huangst@163.com"
DOWNLOAD_DIR = project_path('Downloading_Data/ASKAP_Catalogue')
FAILED_CSV = os.path.join(DOWNLOAD_DIR, "failed_downloads.csv")
START_FROM_NUMBER = 1  # 从第几个文件开始
BATCH_SIZE = 3
MAX_RETRY = 3  # staging / download 都最多试 3 次
SLEEP_BETWEEN_RETRY = 5  # 秒

# 1.LOGIN
casda = Casda()
try:
    casda.login(username=OPAL_USER, store_password=True)
    print(f" Logged in as {OPAL_USER}")
except Exception as e:
    print(f" Login failed: {e}")
    sys.exit(1)

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# 2.CSV FAILURE LOGGER
def log_failure(record: dict):
    """Append failure info to CSV immediately"""
    df = pd.DataFrame([record])
    header = not os.path.exists(FAILED_CSV)
    df.to_csv(FAILED_CSV, mode="a", index=False, header=header)


# 3.QUERY CASDA
print("\n🔍 Querying CASDA ivoa.obscore ...")
tap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")
job = tap.launch_job_async(
    ("SELECT TOP 50000 * FROM ivoa.obscore where(dataproduct_subtype = 'catalogue.continuum.component')"))
r = job.get_results()
data = r[(r["quality_level"] == "GOOD") | (r["quality_level"] == "UNCERTAIN")]
unique_files = np.unique(data["filename"])
total_files = len(unique_files)
start_index = START_FROM_NUMBER - 1

print(f"\n Total files: {total_files}")
print(f" Resume from: {START_FROM_NUMBER}/{total_files}")
print("-" * 60)

# 4.DOWNLOAD LOOP (WITH RETRY & SKIP EXISTING)
urls_to_download = []


def download_batch(urls):
    """辅助函数：处理批量下载并包含重试逻辑"""
    if not urls:
        return []

    for attempt in range(1, MAX_RETRY + 1):
        print(f"    Download attempt {attempt} ({len(urls)} files)")
        try:
            casda.download_files(urls, savedir=DOWNLOAD_DIR)
            print("    Batch download success")
            return []  # 下载成功，清空列表

        except Exception as e:
            print(f"    Download failed: {e}")
            if attempt == MAX_RETRY:
                # 达到最大重试次数，记录失败并清空列表以防卡死
                for url in urls:
                    log_failure({
                        "filename": os.path.basename(url),
                        "stage_or_download": "download",
                        "attempt": attempt,
                        "error_message": str(e)
                    })
                return []
            time.sleep(SLEEP_BETWEEN_RETRY)
    return []


for i in range(start_index, total_files):
    filename = unique_files[i]
    local_filepath = os.path.join(DOWNLOAD_DIR, filename)

    # 检查本地文件是否存在
    if os.path.exists(local_filepath):
        print(f"\n [{i + 1}/{total_files}] ⏭ Skipped: {filename} (Already exists in local dir)")
        continue

    print(f"\n [{i + 1}/{total_files}]  Processing: {filename}")

    pdata = data[data["filename"] == filename]

    # ---------------- STAGING (RETRY) ----------------
    staged = False
    for attempt in range(1, MAX_RETRY + 1):
        try:
            urls = casda.stage_data(pdata)
            urls_to_download.extend(u for u in urls if u not in urls_to_download)
            staged = True
            break
        except Exception as e:
            print(f"   ️ Staging attempt {attempt} failed: {e}")
            log_failure({
                "filename": filename,
                "stage_or_download": "stage",
                "attempt": attempt,
                "error_message": str(e)
            })
            time.sleep(SLEEP_BETWEEN_RETRY)

    if not staged:
        print("    Staging failed after max retries.")
        continue

    # ---------------- DOWNLOAD (BATCH) ----------------
    if len(urls_to_download) >= BATCH_SIZE:
        urls_to_download = download_batch(urls_to_download)

# ---------------- 处理最后剩余的未满 BATCH_SIZE 的文件 ----------------
if len(urls_to_download) > 0:
    print("\n▶ Processing final remaining batch...")
    urls_to_download = download_batch(urls_to_download)

# 5.FINISH
print("\n" + "=" * 60)
print(" SCRIPT FINISHED")
print(f" Failure log saved continuously at:\n{FAILED_CSV}")
print(f" Files in directory: {len(os.listdir(DOWNLOAD_DIR))}")
