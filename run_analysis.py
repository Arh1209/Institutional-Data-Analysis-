from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "all_institutions.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PLOTS_DIR = ROOT / "outputs" / "plots"
RESULTS_DIR = ROOT / "outputs" / "model_results"
for folder in [PROCESSED_DIR, PLOTS_DIR, RESULTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 70)
    print("INSTITUTION DATASET: CLEANING, EDA, CORRELATION & ML COMPARISON")
    print("=" * 70)

    # 1. Read CSV
    print("\n[1/8] Reading CSV...")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"Original shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\nColumns:", df.columns.tolist())

    # 2. Initial data audit and cleaning
    print("\n[2/8] Data audit and cleaning...")
    audit = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_percent": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    audit.to_csv(RESULTS_DIR / "data_quality_report.csv", index=False)
    df = df.drop_duplicates().copy()

    # Normalize text fields, without changing case globally (names may use styling)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        df[col] = df[col].replace("", np.nan)

    # Parse year-like columns numerically
    for col in ["yearOfEstablishment", "uploadedYear", "addedYear"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop columns that are completely empty; record them for transparency
    all_empty_cols = [c for c in df.columns if df[c].isna().all()]
    if all_empty_cols:
        print("Dropping fully empty columns:", all_empty_cols)
        df = df.drop(columns=all_empty_cols)

    df.to_csv(PROCESSED_DIR / "institutions_cleaned.csv", index=False)
    print(f"Cleaned shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Exact duplicate rows removed: {0}")  # original duplicate count is calculated below from raw file
    raw_for_dupes = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"Exact duplicate rows in original file: {int(raw_for_dupes.duplicated().sum()):,}")
    print("Cleaned dataset saved to data/processed/institutions_cleaned.csv")

    # 3. Descriptive statistics
    print("\n[3/8] Descriptive statistics...")
    desc = df.describe(include="all").transpose()
    desc.to_csv(RESULTS_DIR / "descriptive_statistics.csv")
    numeric = df.select_dtypes(include=np.number)
    if not numeric.empty:
        numeric_stats = pd.DataFrame({
            "mean": numeric.mean(numeric_only=True),
            "median": numeric.median(numeric_only=True),
            "mode": [numeric[c].mode(dropna=True).iloc[0] if not numeric[c].mode(dropna=True).empty else np.nan for c in numeric.columns],
            "std": numeric.std(numeric_only=True),
            "variance": numeric.var(numeric_only=True),
            "min": numeric.min(numeric_only=True),
            "q1": numeric.quantile(0.25),
            "q3": numeric.quantile(0.75),
            "max": numeric.max(numeric_only=True),
            "missing": numeric.isna().sum()
        })
        numeric_stats.to_csv(RESULTS_DIR / "numeric_statistics_mean_median_mode.csv")
        print(numeric_stats.to_string())
    else:
        print("No numeric columns available for numeric descriptive statistics.")

    # 4. EDA plots
    print("\n[4/8] Generating EDA plots...")
    sns.set_theme()
    plt.figure(figsize=(10, 6))
    df["stateName"].value_counts().head(15).sort_values().plot(kind="barh")
    plt.title("Top 15 States by Institution Count")
    plt.xlabel("Number of institutions")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "institutions_by_state.png", dpi=160)
    plt.close()

    if "institutionCategory" in df.columns:
        plt.figure(figsize=(9, 5))
        df["institutionCategory"].value_counts(dropna=False).plot(kind="bar")
        plt.title("Institution Category Distribution")
        plt.xlabel("Institution category")
        plt.ylabel("Count")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "institution_category_distribution.png", dpi=160)
        plt.close()

    if "institutionType" in df.columns:
        plt.figure(figsize=(11, 7))
        df["institutionType"].value_counts().head(15).sort_values().plot(kind="barh")
        plt.title("Top 15 Institution Types")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "top_institution_types.png", dpi=160)
        plt.close()

    if "yearOfEstablishment" in df.columns:
        years = df["yearOfEstablishment"].dropna()
        years = years[(years >= 1800) & (years <= pd.Timestamp.now().year)]
        if not years.empty:
            plt.figure(figsize=(10, 5))
            sns.histplot(years, bins=35, kde=True)
            plt.title("Distribution of Institution Establishment Years")
            plt.xlabel("Year of establishment")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "establishment_year_distribution.png", dpi=160)
            plt.close()

    # 5. Correlation heatmap with values
    print("\n[5/8] Correlation analysis and annotated heatmap...")
    # Original dataset has very few meaningful numeric columns. Add transparent binary indicators
    # so the heatmap can also show associations with missingness/presence patterns.
    corr_df = df.select_dtypes(include=np.number).copy()
    for col in ["webSite", "manegement", "universityName", "districtName", "address1"]:
        if col in df.columns:
            corr_df[f"{col}_available"] = df[col].notna().astype(int)
    if not corr_df.empty and corr_df.shape[1] >= 2:
        corr = corr_df.corr(numeric_only=True)
        corr.to_csv(RESULTS_DIR / "correlation_matrix.csv")
        plt.figure(figsize=(max(9, 1.0 * len(corr.columns)), max(7, 0.8 * len(corr.columns))))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, square=False, annot_kws={"size": 8})
        plt.title("Correlation Heatmap (Pearson r; values annotated)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=180)
        plt.close()
        print(corr.round(3).to_string())
    else:
        print("Not enough numeric/engineered columns to calculate correlation matrix.")

    # 6. ML classification setup
    print("\n[6/8] Preparing ML dataset...")
    target = "institutionCategory"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is not present in this CSV.")

    # Exclude identifiers/free-text fields that are not useful for this starter classification.
    # institutionType is retained as a predictor; inspect potential overlap with category when interpreting.
    feature_candidates = [
        "stateName", "districtName", "manegement", "yearOfEstablishment",
        "institutionType", "universityType", "location", "institutionCategory"
    ]
    features = [c for c in feature_candidates if c in df.columns and c != target]
    ml_df = df[features + [target]].copy()
    ml_df = ml_df.dropna(subset=[target])
    # Remove rare classes with fewer than 2 samples only if stratified splitting cannot support them.
    counts = ml_df[target].value_counts()
    valid_classes = counts[counts >= 2].index
    ml_df = ml_df[ml_df[target].isin(valid_classes)].copy()
    X = ml_df[features]
    y = ml_df[target].astype(str)

    if y.nunique() < 2:
        raise ValueError("Need at least two target categories for classification.")
    stratify_y = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify_y
    )
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", encoder)
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ])

    # 7. Train and compare models
    print("\n[7/8] Training and evaluating classification models...")
    models = {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=18, min_samples_leaf=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=22, min_samples_leaf=3,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    }
    results = []
    fitted_models = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "precision_weighted": precision_score(y_test, preds, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, preds, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, preds, average="weighted", zero_division=0)
        }
        results.append(row)
        fitted_models[name] = pipe
        print(f"{name}: accuracy={row['accuracy']:.4f}, F1={row['f1_weighted']:.4f}")

    results_df = pd.DataFrame(results).sort_values("f1_weighted", ascending=False)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    joblib.dump(best_model, RESULTS_DIR / "best_institution_category_model.joblib")
    with open(RESULTS_DIR / "best_model_name.txt", "w", encoding="utf-8") as f:
        f.write(best_name)

    # Comparison plot
    plot_df = results_df.melt(id_vars="model", value_vars=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
                              var_name="metric", value_name="score")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric")
    plt.title("Classification Model Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=160)
    plt.close()

    # Classification report and confusion matrix for best model
    best_preds = best_model.predict(X_test)
    report = classification_report(y_test, best_preds, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(RESULTS_DIR / "best_model_classification_report.csv")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, best_preds, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xlabel("Predicted category")
    plt.ylabel("Actual category")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "best_model_confusion_matrix.png", dpi=160)
    plt.close()

    # Save sample test predictions
    prediction_output = X_test.copy()
    prediction_output["actual_category"] = y_test
    prediction_output["predicted_category"] = best_preds
    prediction_output.to_csv(RESULTS_DIR / "best_model_test_predictions.csv", index=False)

    print("\n[8/8] Done.")
    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model by weighted F1: {best_name}")
    print(f"Outputs saved under: {ROOT / 'outputs'}")
    print("\nNote: this is classification of institutionCategory, NOT cutoff percentile prediction.")

if __name__ == "__main__":
    main()
