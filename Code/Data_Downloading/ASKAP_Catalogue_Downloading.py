"""Batch-download ASKAP catalogue files with retry logging and resumable progress."""

import numpy as np
import pandas as pd
from astroquery import log as astroquery_log
from astroquery.casda import Casda
from astroquery.utils.tap.core import TapPlus
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse




# 0.USER CONFIGURATION
OPAL_USER = "acentauri_huangst@163.com"
DOWNLOAD_DIR = '/Volumes/HST/Research/ASKAP_Stellar_with_Planet_Localbin/Data/ASKAP_Catalogue'
FAILED_CSV = os.path.join(DOWNLOAD_DIR, "failed_downloads.csv")
START_FROM_NUMBER = 1  # 从第几个文件开始
BATCH_SIZE = 3
MAX_RETRY = 3  # staging / download 都最多试 3 次
SLEEP_BETWEEN_RETRY = 5  # 秒

# 1.LOGIN
casda = Casda()


def _error_summary(error: Exception) -> str:
    """Keep the exception reason while hiding credentials in URL queries."""
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    summary = type(error).__name__
    if status is not None:
        summary += f" HTTP {status}"
    body = getattr(response, "text", "") or ""
    code = re.search(r"<Code>([A-Za-z0-9_]+)</Code>", body)
    if code:
        summary += f" {code.group(1)}"
    detail = str(error).strip()
    detail = re.sub(r"\?[^\s'\"<>]+", " [URL query redacted]", detail)
    detail = re.sub(r"(https?://)[^/@\s]+@", r"\1[credentials redacted]@", detail)
    detail = re.sub(
        r"(?i)\b(?:x-amz-signature|x-amz-credential|awsaccesskeyid|access_token|token|authorization|password|client_secret|api[_-]?key)\s*[:=]\s*[^\s,'\"<>]+",
        "[sensitive value redacted]",
        detail,
    )
    if detail:
        summary += f": {detail}"
    return summary


def _record_failure(filename: str, phase: str, attempt: int, message: str) -> None:
    record = {
        "filename": filename,
        "stage_or_download": phase,
        "attempt": attempt,
        "error_message": message,
    }
    pd.DataFrame([record]).to_csv(
        FAILED_CSV,
        mode="a",
        index=False,
        header=not os.path.exists(FAILED_CSV),
        encoding="utf-8",
    )


def main() -> None:
    try:
        casda.login(username=OPAL_USER, store_password=False)
        print(f" Logged in as {OPAL_USER}")
    except Exception as e:
        print(f" Login failed: {_error_summary(e)}")
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
    pending_groups = []  # (catalogue filename, its table, its staged URLs)

    # The sentinel iteration flushes the final batch even if the last file was skipped.
    for i in range(start_index, total_files + 1):
        if i < total_files:
            filename = unique_files[i]
            local_filepath = os.path.join(DOWNLOAD_DIR, filename)

            if os.path.exists(local_filepath):
                print(f"\n [{i + 1}/{total_files}] ⏭ Skipped: {filename} (Already exists in local dir)")
                continue

            print(f"\n [{i + 1}/{total_files}]  Processing: {filename}")
            pdata = data[data["filename"] == filename]
            staged = False
            for attempt in range(1, MAX_RETRY + 1):
                try:
                    urls = casda.stage_data(pdata)
                    if not urls:
                        raise ValueError("CASDA staging returned no URLs")
                    staged = True
                    break
                except Exception as error:
                    summary = _error_summary(error)
                    print(f"    Staging attempt {attempt} failed: {summary}")
                    _record_failure(filename, "stage", attempt, summary)
                    if attempt < MAX_RETRY:
                        time.sleep(SLEEP_BETWEEN_RETRY)

            if not staged:
                print("    Staging failed after max retries.")
                continue

            new_urls = []
            for url in urls:
                if url not in urls_to_download and url not in new_urls:
                    new_urls.append(url)
            if new_urls:
                urls_to_download.extend(new_urls)
                pending_groups.append((filename, pdata, new_urls))
            if len(urls_to_download) < BATCH_SIZE:
                continue
        elif not pending_groups:
            break
        else:
            print("\n▶ Processing final remaining batch...")

        batch_groups = pending_groups
        pending_groups = []
        urls_to_download = []

        # Astroquery downloads URLs serially; grouping by catalogue isolates errors.
        for filename, pdata, original_urls in batch_groups:
            current_urls = original_urls
            for attempt in range(1, MAX_RETRY + 1):
                if attempt > 1:
                    try:
                        current_urls = casda.stage_data(pdata)
                        if not current_urls:
                            raise ValueError("CASDA restaging returned no URLs")
                    except Exception as error:
                        summary = _error_summary(error)
                        print(f"    Restaging attempt {attempt} failed for {filename}: {summary}")
                        _record_failure(filename, "stage", attempt, summary)
                        if attempt < MAX_RETRY:
                            time.sleep(SLEEP_BETWEEN_RETRY)
                        continue

                print(f"    Download attempt {attempt} ({len(current_urls)} files): {filename}")
                previous_log_level = astroquery_log.level
                astroquery_log.setLevel("WARNING")
                try:
                    casda.download_files(current_urls, savedir=DOWNLOAD_DIR)
                    print(f"    Download success: {filename}")
                    break
                except Exception as error:
                    summary = _error_summary(error)
                    print(f"    Download failed: {filename}: {summary}")
                    if attempt == MAX_RETRY:
                        for url in current_urls:
                            safe_name = unquote(os.path.basename(urlparse(url).path))
                            _record_failure(safe_name, "download", attempt, summary)
                    else:
                        time.sleep(SLEEP_BETWEEN_RETRY)
                finally:
                    astroquery_log.setLevel(previous_log_level)

    # 5.FINISH
    print("\n" + "=" * 60)
    print(" SCRIPT FINISHED")
    print(f" Failure log saved continuously at:\n{FAILED_CSV}")
    print(f" Files in directory: {len(os.listdir(DOWNLOAD_DIR))}")

if __name__ == "__main__":
    main()
