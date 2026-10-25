"""Create random DEMO csv files in data/demo/ (NOT real UNSW-NB15 data). Run: python make_demo_data.py"""
from pathlib import Path
from src.demo_data import write_demo_files

folder = write_demo_files(Path(__file__).resolve().parent / "data" / "demo")
print("Demo files written to data/demo/ (random data - not real traffic).")
print("Use data/demo/demo_upload_sample.csv to test the upload page.")
