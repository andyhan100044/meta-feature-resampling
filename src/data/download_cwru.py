"""
CWRU轴承数据集下载脚本
数据集来源: Case Western Reserve University Bearing Database
"""
import os
import requests
from pathlib import Path

CWRU_DATA_URLS = {
    "12k_DE": "https://engineering.case.edu/bearing/databases/12k%20Drive%20End%20Bearing%20Fault%20Data.zip",
    "12k_FE": "https://engineering.case.edu/bearing/databases/12k%20Fan%20End%20Bearing%20Fault%20Data.zip",
    "48k_DE": "https://engineering.case.edu/bearing/databases/48k%20Drive%20End%20Bearing%20Fault%20Data.zip",
    "normal": "https://engineering.case.edu/bearing/databases/Normal%20Baseline%20Data.zip"
}

def download_cwru(data_dir="data/raw"):
    """
    下载CWRU数据集

    注意: CWRU数据集需要注册后下载，此脚本提供下载链接和校验信息
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CWRU轴承数据集下载")
    print("=" * 60)
    print("\n请访问以下网址手动下载数据集:")
    print("https://engineering.case.edu/bearing/databases/")
    print("\n需要下载的文件:")
    for name, url in CWRU_DATA_URLS.items():
        print(f"  - {name}: {url}")
    print("\n下载后解压到 data/raw/cwru/ 目录")
    print("预期目录结构:")
    print("  data/raw/cwru/")
    print("    ├── 12k_Drive_End_Bearing_Fault_Data/")
    print("    ├── 12k_Fan_End_Bearing_Fault_Data/")
    print("    ├── 48k_Drive_End_Bearing_Fault_Data/")
    print("    └── Normal_Baseline_Data/")
    return data_dir

if __name__ == "__main__":
    download_cwru()