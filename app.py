from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "institutions_cleaned.csv"
RAW_PATH = ROOT / "data" / "raw" / "all_institutions.csv"
OUT = ROOT / "outputs"
PLOTS = OUT / "plots"
RESULTS = OUT / "model_results"

st.set_page_config(page_title="Institution Data Insights", page_icon="📊", layout="wide")
st.title("🇮🇳 Institution Data Analysis & ML Dashboard")
st.caption("Exploratory analysis and institution-category classification using the provided all_institutions.csv dataset.")

@st.cache_data
def load_data():
    path = DATA_PATH if DATA_PATH.exists() else RAW_PATH
    return pd.read_csv(path, low_memory=False)

df = load_data()
st.sidebar.header("Dashboard Filters")
states = sorted(df["stateName"].dropna().astype(str).unique()) if "stateName" in df.columns else []
selected_states = st.sidebar.multiselect("State", states, default=[])
cats = sorted(df["institutionCategory"].dropna().astype(str).unique()) if "institutionCategory" in df.columns else []
selected_cats = st.sidebar.multiselect("Institution category", cats, default=[])

filtered = df.copy()
if selected_states:
    filtered = filtered[filtered["stateName"].astype(str).isin(selected_states)]
if selected_cats:
    filtered = filtered[filtered["institutionCategory"].astype(str).isin(selected_cats)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Institutions", f"{len(filtered):,}")
c2.metric("States/UTs", f"{filtered['stateName'].nunique():,}" if "stateName" in filtered else "N/A")
c3.metric("Institution categories", f"{filtered['institutionCategory'].nunique():,}" if "institutionCategory" in filtered else "N/A")
c4.metric("Institution types", f"{filtered['institutionType'].nunique():,}" if "institutionType" in filtered else "N/A")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Descriptive Stats", "Correlation", "Model Comparison", "Explore Data"])

with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("Institutions by State")
        if "stateName" in filtered:
            counts = filtered["stateName"].value_counts().head(15).sort_values()
            fig, ax = plt.subplots(figsize=(8, 6))
            counts.plot(kind="barh", ax=ax)
            ax.set_xlabel("Institution count")
            st.pyplot(fig)
            plt.close(fig)
    with right:
        st.subheader("Institution Category")
        if "institutionCategory" in filtered:
            fig, ax = plt.subplots(figsize=(8, 5))
            filtered["institutionCategory"].value_counts().plot(kind="bar", ax=ax)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=35)
            st.pyplot(fig)
            plt.close(fig)
    if "institutionType" in filtered:
        st.subheader("Top 15 Institution Types")
        st.bar_chart(filtered["institutionType"].value_counts().head(15))

with tab2:
    st.subheader("Descriptive statistics")
    st.write("Numeric columns: mean, median, mode, standard deviation, variance, quartiles, min and max.")
    numeric = filtered.select_dtypes(include=np.number)
    if not numeric.empty:
        stats = pd.DataFrame({
            "mean": numeric.mean(),
            "median": numeric.median(),
            "mode": [numeric[c].mode().iloc[0] if not numeric[c].mode().empty else np.nan for c in numeric],
            "std": numeric.std(),
            "variance": numeric.var(),
            "min": numeric.min(),
            "25%": numeric.quantile(.25),
            "75%": numeric.quantile(.75),
            "max": numeric.max(),
            "missing": numeric.isna().sum()
        })
        st.dataframe(stats, use_container_width=True)
    st.subheader("Pandas describe()")
    st.dataframe(filtered.describe(include="all").transpose(), use_container_width=True)

with tab3:
    st.subheader("Pearson Correlation Heatmap")
    corr_data = filtered.select_dtypes(include=np.number).copy()
    for col in ["webSite", "manegement", "universityName", "districtName", "address1"]:
        if col in filtered.columns:
            corr_data[f"{col}_available"] = filtered[col].notna().astype(int)
    if corr_data.shape[1] >= 2:
        corr = corr_data.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(max(8, len(corr.columns)), max(6, len(corr.columns)*0.7)))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(corr.round(3), use_container_width=True)
    else:
        st.info("There are not enough numeric variables for a useful correlation matrix. The pipeline adds numeric availability indicators where appropriate.")

with tab4:
    st.subheader("Classification Model Comparison")
    st.info("The provided file has no percentile/merit-number target. These results classify institutionCategory; they do not predict engineering cutoffs.")
    result_path = RESULTS / "model_comparison.csv"
    if result_path.exists():
        results = pd.read_csv(result_path)
        st.dataframe(results, use_container_width=True)
        plot_path = PLOTS / "model_comparison.png"
        if plot_path.exists():
            st.image(str(plot_path), caption="Accuracy, precision, recall and F1 comparison")
        best_name_path = RESULTS / "best_model_name.txt"
        if best_name_path.exists():
            st.success("Best model by weighted F1: " + best_name_path.read_text(encoding="utf-8").strip())
        cm_path = PLOTS / "best_model_confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Best model confusion matrix")
    else:
        st.warning("Model results not found yet. In VS Code terminal run: python run_analysis.py")

with tab5:
    st.subheader("Filtered records")
    st.write(f"Showing {len(filtered):,} records after filters.")
    search = st.text_input("Search institution name")
    view = filtered
    if search and "name" in view:
        view = view[view["name"].fillna("").str.contains(search, case=False, regex=False)]
    st.dataframe(view.head(2000), use_container_width=True)
    st.caption("For performance, the table shows at most the first 2,000 matching records.")
    st.download_button("Download filtered CSV", data=view.to_csv(index=False).encode("utf-8"), file_name="filtered_institutions.csv", mime="text/csv")
