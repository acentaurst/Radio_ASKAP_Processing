"""Query CASDA for ASKAP observations and save the project catalogue table."""

import os
from astroquery.casda import Casda
import pandas as pd
from astroquery.utils.tap.core import TapPlus
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]




# 1. Setup paths
output_dir = PROJECT_ROOT / "Processed_Data" / "Catalogue"
output_filename = "01.askap_catalogue.csv"
output_path = os.path.join(output_dir, output_filename)

def main() -> None:
    os.makedirs(output_dir, exist_ok=True)

    # 2. Login (This will now prompt for your password in the terminal)
    casda = Casda()
    casda.login(username='acentauri_huangst@163.com')

    # 3. Connect to CASDA TAP service
    # Use the authenticated session from the casda object if needed
    tap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")

    # 4. Launch Query
    print("正在启动 TAP 异步查询作业...")
    query = "SELECT TOP 50000 * FROM ivoa.obscore WHERE dataproduct_subtype = 'catalogue.continuum.component'"
    job = tap.launch_job_async(query)

    # 5. Get and Save Results
    result = job.get_results()
    df = result.to_pandas()
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"CSV 文件已保存：{output_path}")

if __name__ == "__main__":
    main()
