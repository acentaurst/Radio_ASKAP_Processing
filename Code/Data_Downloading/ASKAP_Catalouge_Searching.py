"""Query CASDA for ASKAP observations and save the project catalogue table."""

import os
import sys
import tempfile
import time
from astroquery.casda import Casda
import pandas as pd
from astroquery.utils.tap.core import TapPlus
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from science_utils import safe_download_error  # noqa: E402




# 1. Setup paths
output_dir = PROJECT_ROOT / "Processed_Data" / "Catalogue"
output_filename = "01.askap_catalogue.csv"
output_path = os.path.join(output_dir, output_filename)

def main() -> None:
    os.makedirs(output_dir, exist_ok=True)

    # 2. Login (This will now prompt for your password in the terminal)
    casda = Casda()
    try:
        casda.login(username='acentauri_huangst@163.com')
    except Exception as error:
        raise RuntimeError(f"CASDA 登录失败: {safe_download_error(error)}") from None

    # 3. Connect to CASDA TAP service
    # Use the authenticated session from the casda object if needed
    tap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")

    # 4. Launch Query
    print("正在启动 TAP 异步查询作业...")
    query = "SELECT TOP 50000 * FROM ivoa.obscore WHERE dataproduct_subtype = 'catalogue.continuum.component'"
    for attempt in range(1, 4):
        try:
            job = tap.launch_job_async(query)
            result = job.get_results()
            break
        except Exception as error:
            summary = safe_download_error(error)
            print(f"TAP 查询失败 {attempt}/3: {summary}")
            if attempt == 3:
                raise RuntimeError(f"CASDA TAP 查询失败: {summary}") from None
            time.sleep(5)

    # 5. Get and Save Results
    df = result.to_pandas()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix=".askap_catalogue_",
            dir=output_dir, delete=False, encoding="utf-8", newline="",
        ) as handle:
            temporary_path = handle.name
            df.to_csv(handle, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)

    print(f"CSV 文件已保存：{output_path}")

if __name__ == "__main__":
    main()
