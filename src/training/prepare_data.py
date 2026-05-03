from pathlib import Path
import argparse
import shutil
import zipfile

import pandas as pd
import requests


DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
ZIP_PATH = DATA_DIR / "complaints.csv.zip"
CSV_PATH = DATA_DIR / "complaints.csv"
CFPB_ZIP_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
TWITTER_TARGET_PATH = DATA_DIR / "twitter_customer_support.csv"
COMBINED_PATH = DATA_DIR / "combined_text_sources.csv"


def download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with target_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_obj.write(chunk)


def extract_csv(zip_path: Path, extract_to: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def merge_sources(twitter_csv_path: Path | None) -> None:
    if not CSV_PATH.exists():
        return

    source_frames = []
    cfpb_df = pd.read_csv(CSV_PATH, usecols=["Consumer complaint narrative"], low_memory=False)
    cfpb_df = cfpb_df.rename(columns={"Consumer complaint narrative": "text"})
    cfpb_df["source"] = "cfpb"
    source_frames.append(cfpb_df.dropna(subset=["text"]))

    if twitter_csv_path is not None and twitter_csv_path.exists():
        shutil.copy2(twitter_csv_path, TWITTER_TARGET_PATH)
        twitter_df = pd.read_csv(TWITTER_TARGET_PATH, low_memory=False)
        text_col = None
        for candidate in ["text", "tweet", "author_text", "message"]:
            if candidate in twitter_df.columns:
                text_col = candidate
                break
        if text_col is not None:
            twitter_df = twitter_df[[text_col]].rename(columns={text_col: "text"})
            twitter_df["source"] = "twitter"
            source_frames.append(twitter_df.dropna(subset=["text"]))
            print(f"Twitter dataset included: {TWITTER_TARGET_PATH}")
        else:
            print("Twitter CSV found, but no text-like column was detected. Skipping merge.")

    combined = pd.concat(source_frames, ignore_index=True)
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 0]
    combined.to_csv(COMBINED_PATH, index=False)
    print(f"Saved merged text sources: {COMBINED_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--twitter-csv",
        type=str,
        default="",
        help="Optional path to Twitter customer support CSV for multi-source text analysis.",
    )
    args = parser.parse_args()

    if not CSV_PATH.exists():
        if not ZIP_PATH.exists():
            print("Downloading CFPB complaints dataset...")
            download_file(CFPB_ZIP_URL, ZIP_PATH)
            print(f"Downloaded: {ZIP_PATH}")

        print("Extracting dataset...")
        extract_csv(ZIP_PATH, DATA_DIR)
        print(f"Done. CSV path: {CSV_PATH}")
    else:
        print(f"Dataset already exists: {CSV_PATH}")

    twitter_csv = Path(args.twitter_csv).expanduser().resolve() if args.twitter_csv else None
    merge_sources(twitter_csv)


if __name__ == "__main__":
    main()
