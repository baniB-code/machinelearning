from pathlib import Path
import argparse

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "dataset" / "complaints.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
CHARTS_DIR = ROOT_DIR / "outputs" / "charts"

TEXT_COLUMN = "Consumer complaint narrative"
TARGET_COLUMN = "Issue"


def normalize_target_labels(series: pd.Series, target_column: str) -> pd.Series:
    """Merge known near-duplicate labels to improve label consistency."""
    if target_column != "Product":
        return series

    return series.replace(
        {
            "Credit reporting or other personal consumer reports": "Credit reporting",
            "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
        }
    )


def prepare_dataframe(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    # Keep only text + target columns and remove rows without useful text/label.
    work_df = df[[TEXT_COLUMN, target_column]].copy()
    work_df = work_df.dropna(subset=[TEXT_COLUMN, target_column])
    work_df[TEXT_COLUMN] = work_df[TEXT_COLUMN].astype(str).str.strip()
    work_df[target_column] = work_df[target_column].astype(str).str.strip()
    work_df[target_column] = normalize_target_labels(work_df[target_column], target_column)
    work_df = work_df[work_df[TEXT_COLUMN].str.len() > 20]
    work_df = work_df[work_df[target_column].str.len() > 0]
    return work_df


def load_training_dataframe(
    sample_size: int, chunksize: int, random_state: int, target_column: str
) -> pd.DataFrame:
    """Load data in chunks so very large CSV files do not exhaust memory."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. "
            "Run `python src/training/prepare_data.py` first."
        )

    if sample_size == 0:
        # Full-load mode (kept for compatibility), only safe on high-memory machines.
        df = pd.read_csv(DATASET_PATH, low_memory=False)
        return prepare_dataframe(df, target_column=target_column)

    rng_seed = random_state
    chunks = []
    rows_collected = 0

    # Read only columns required for this model to reduce memory pressure.
    for chunk in pd.read_csv(
        DATASET_PATH,
        usecols=[TEXT_COLUMN, target_column],
        chunksize=chunksize,
        low_memory=False,
    ):
        print(f"Loaded chunk rows: {len(chunk)}")
        clean_chunk = prepare_dataframe(chunk, target_column=target_column)
        if clean_chunk.empty:
            continue

        rows_left = sample_size - rows_collected
        if rows_left <= 0:
            break

        if len(clean_chunk) > rows_left:
            clean_chunk = clean_chunk.sample(n=rows_left, random_state=rng_seed)

        chunks.append(clean_chunk)
        rows_collected += len(clean_chunk)
        print(f"Collected cleaned rows: {rows_collected}/{sample_size}")
        rng_seed += 1

        if rows_collected >= sample_size:
            break

    if not chunks:
        raise ValueError("No usable rows found in dataset after cleaning.")

    df = pd.concat(chunks, ignore_index=True)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
    return df


def save_report(report_text: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Saved report: {report_path}")


def save_model_comparison(comparison_text: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = REPORTS_DIR / "model_comparison.txt"
    comparison_path.write_text(comparison_text, encoding="utf-8")
    print(f"Saved model comparison: {comparison_path}")


def save_confusion_matrix(y_true, y_pred, labels, class_names, target_column: str) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        cmap="Blues",
        annot=False,
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {target_column} Classification")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    chart_path = CHARTS_DIR / "confusion_matrix.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved chart: {chart_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-size",
        type=int,
        default=180000,
        help=(
            "Target number of cleaned rows to train on. "
            "Set 0 to load full cleaned dataset (memory-heavy)."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Chunk size for memory-safe CSV loading.",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=200,
        help="Minimum rows per label required to keep a class.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="Product",
        help='Target column to predict (e.g., "Issue" or "Product").',
    )
    parser.add_argument(
        "--top-n-classes",
        type=int,
        default=5,
        help="Keep only top-N most frequent classes. Set 0 to disable.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling and train/test split.",
    )
    args = parser.parse_args()

    print("Loading and cleaning dataset...")
    target_column = args.target_column
    df = load_training_dataframe(
        sample_size=args.sample_size,
        chunksize=args.chunksize,
        random_state=args.random_state,
        target_column=target_column,
    )
    print(f"Rows after load/clean: {len(df)}")

    # Keep classes with enough examples to make training stable.
    value_counts = df[target_column].value_counts()
    valid_labels = value_counts[value_counts >= args.min_class_count].index
    df = df[df[target_column].isin(valid_labels)].copy()

    if args.top_n_classes > 0:
        top_labels = df[target_column].value_counts().head(args.top_n_classes).index
        df = df[df[target_column].isin(top_labels)].copy()
        valid_labels = top_labels

    print(f"Rows after class filtering: {len(df)}")
    print(f"Class count kept: {len(valid_labels)}")

    X = df[[TEXT_COLUMN]]
    y = df[target_column]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y_encoded,
    )

    text_transformer = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=30000,
                    min_df=3,
                ),
                TEXT_COLUMN,
            )
        ]
    )

    logistic_model = Pipeline(
        steps=[
            ("features", text_transformer),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=None,
                ),
            ),
        ]
    )
    nb_model = Pipeline(
        steps=[
            ("features", text_transformer),
            ("classifier", MultinomialNB()),
        ]
    )

    print("Training model A (TF-IDF + Logistic Regression)...")
    logistic_model.fit(x_train, y_train)
    logistic_pred = logistic_model.predict(x_test)

    print("Training model B (TF-IDF + Naive Bayes)...")
    nb_model.fit(x_train, y_train)
    nb_pred = nb_model.predict(x_test)

    logistic_acc = accuracy_score(y_test, logistic_pred)
    nb_acc = accuracy_score(y_test, nb_pred)
    logistic_f1 = f1_score(y_test, logistic_pred, average="weighted", zero_division=0)
    nb_f1 = f1_score(y_test, nb_pred, average="weighted", zero_division=0)

    if logistic_f1 >= nb_f1:
        selected_name = "LogisticRegression"
        model = logistic_model
        y_pred = logistic_pred
    else:
        selected_name = "NaiveBayes"
        model = nb_model
        y_pred = nb_pred

    comparison_text = (
        "Model comparison results\n"
        f"- LogisticRegression: accuracy={logistic_acc:.4f}, weighted_f1={logistic_f1:.4f}\n"
        f"- NaiveBayes: accuracy={nb_acc:.4f}, weighted_f1={nb_f1:.4f}\n"
        f"- Selected model: {selected_name}\n"
    )
    print(comparison_text)
    save_model_comparison(comparison_text)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    print(report)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "issue_classifier.joblib"
    encoder_path = MODELS_DIR / "label_encoder.joblib"
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    print(f"Saved model ({selected_name}): {model_path}")
    print(f"Saved encoder: {encoder_path}")

    save_report(report)
    labels = list(range(len(label_encoder.classes_)))
    class_names = list(label_encoder.classes_)
    save_confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
        class_names=class_names,
        target_column=target_column,
    )


if __name__ == "__main__":
    main()
