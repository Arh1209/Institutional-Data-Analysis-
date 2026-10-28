# Institution Data Science Project (VS Code)
<img width="1803" height="813" alt="image" src="https://github.com/user-attachments/assets/e8c2877d-a37b-4b0b-ab39-60edd02c2349" />


## Dataset note
The supplied `all_institutions.csv` contains institution-level records (72,703 rows and 17 columns in the supplied copy). It does **not** contain engineering cutoff percentile or merit number. Therefore, this project performs a classification task to predict `institutionCategory`, rather than cutoff regression.

## 1. Open in VS Code
1. Extract this project ZIP.
2. Open the extracted `Institution_Data_Science_VSCode_Project` folder in VS Code.
3. Open Terminal → New Terminal.

## 2. Create a virtual environment (Windows)
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, use Command Prompt terminal:
```bat
.venv\Scripts\activate.bat
```
If `py -3` is not installed, use the Python version installed on your computer and make sure it is Python 3.10+.

## 3. Run the full analysis
```powershell
python run_analysis.py
```
This reads `data/raw/all_institutions.csv` and creates:
- `data/processed/institutions_cleaned.csv`
- `outputs/model_results/data_quality_report.csv`
- `outputs/model_results/descriptive_statistics.csv`
- `outputs/model_results/numeric_statistics_mean_median_mode.csv`
- `outputs/model_results/correlation_matrix.csv`
- `outputs/model_results/model_comparison.csv`
- `outputs/model_results/best_model_classification_report.csv`
- `outputs/model_results/best_model_test_predictions.csv`
- `outputs/model_results/best_institution_category_model.joblib`
- EDA, correlation heatmap, model comparison and confusion matrix plots under `outputs/plots/`

## 4. Run the Streamlit dashboard
After `python run_analysis.py` completes:
```powershell
streamlit run app.py
```
Streamlit should open the dashboard in your browser. If not, follow the localhost URL printed in the terminal.

## Models and metrics
- Dummy baseline
- Logistic Regression
- Decision Tree
- Random Forest

Metrics: accuracy, weighted precision, weighted recall, weighted F1. For classification, these are more suitable than MAE/MSE/RMSE/R². Those latter metrics are for a numeric regression target such as percentile, which this file does not have.

## Important interpretation
- The correlation heatmap includes available numeric columns and clearly named binary availability indicators for selected fields. Availability correlations reflect data completeness patterns, not necessarily real-world causal relationships.
- `institutionType` may be strongly associated with `institutionCategory`; inspect the results and be careful not to overstate generalization.
- Use this as a starter academic workflow; verify the outputs and explain limitations in the final report.
