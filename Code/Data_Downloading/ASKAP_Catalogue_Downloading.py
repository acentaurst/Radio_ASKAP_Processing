"""Batch-download ASKAP catalogue files with retry logging and resumable progress."""

import numpy as np
import pandas as pd
from astroquery.casda import Casda
from astroquery.utils.tap.core import TapPlus
import os
import sys
import time
from pathlib import Path




# 0.USER CONFIGURATION
OPAL_USER = "acentauri_huangst@163.com"
DOWNLOAD_DIR = '/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/ASKAP_Catalogue'
FAILED_CSV = os.path.join(DOWNLOAD_DIR, "failed_downloads.csv")
START_FROM_NUMBER = 7600  # 从第几个文件开始
BATCH_SIZE = 3
MAX_RETRY = 3  # staging / download 都最多试 3 次
SLEEP_BETWEEN_RETRY = 5  # 秒

# 1.LOGIN
casda = Casda()

def main() -> None:
    try:
        casda.login(username=OPAL_USER, store_password=False)
        print(f" Logged in as {OPAL_USER}")
    except Exception as e:
        print(f" Login failed: {e}")
        sys.exit(1)

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)


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
                failure_record = {
                    "filename": filename,
                    "stage_or_download": "stage",
                    "attempt": attempt,
                    "error_message": str(e)
                }
                failure_df = pd.DataFrame([failure_record])
                failure_header = not os.path.exists(FAILED_CSV)
                failure_df.to_csv(
                    FAILED_CSV, mode="a", index=False, header=failure_header, encoding="utf-8"
                )
                time.sleep(SLEEP_BETWEEN_RETRY)

        if not staged:
            print("    Staging failed after max retries.")
            continue

        # ---------------- DOWNLOAD (BATCH) ----------------
        if len(urls_to_download) >= BATCH_SIZE:
            batch_urls = urls_to_download
            urls_to_download = []
            for attempt in range(1, MAX_RETRY + 1):
                print(f"    Download attempt {attempt} ({len(batch_urls)} files)")
                try:
                    casda.download_files(batch_urls, savedir=DOWNLOAD_DIR)
                    print("    Batch download success")
                    break
                except Exception as e:
                    print(f"    Download failed: {e}")
                    if attempt == MAX_RETRY:
                        for url in batch_urls:
                            failure_record = {
                                "filename": os.path.basename(url),
                                "stage_or_download": "download",
                                "attempt": attempt,
                                "error_message": str(e),
                            }
                            failure_df = pd.DataFrame([failure_record])
                            failure_header = not os.path.exists(FAILED_CSV)
                            failure_df.to_csv(
                                FAILED_CSV, mode="a", index=False,
                                header=failure_header, encoding="utf-8"
                            )
                    else:
                        time.sleep(SLEEP_BETWEEN_RETRY)

    # ---------------- 处理最后剩余的未满 BATCH_SIZE 的文件 ----------------
    if len(urls_to_download) > 0:
        print("\n▶ Processing final remaining batch...")
        batch_urls = urls_to_download
        urls_to_download = []
        for attempt in range(1, MAX_RETRY + 1):
            print(f"    Download attempt {attempt} ({len(batch_urls)} files)")
            try:
                casda.download_files(batch_urls, savedir=DOWNLOAD_DIR)
                print("    Batch download success")
                break
            except Exception as e:
                print(f"    Download failed: {e}")
                if attempt == MAX_RETRY:
                    for url in batch_urls:
                        failure_record = {
                            "filename": os.path.basename(url),
                            "stage_or_download": "download",
                            "attempt": attempt,
                            "error_message": str(e),
                        }
                        failure_df = pd.DataFrame([failure_record])
                        failure_header = not os.path.exists(FAILED_CSV)
                        failure_df.to_csv(
                            FAILED_CSV, mode="a", index=False,
                            header=failure_header, encoding="utf-8"
                        )
                else:
                    time.sleep(SLEEP_BETWEEN_RETRY)

    # 5.FINISH
    print("\n" + "=" * 60)
    print(" SCRIPT FINISHED")
    print(f" Failure log saved continuously at:\n{FAILED_CSV}")
    print(f" Files in directory: {len(os.listdir(DOWNLOAD_DIR))}")

if __name__ == "__main__":
    main()
