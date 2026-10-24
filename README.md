# Smart Data Science Studio

A Streamlit application for uploading CSV/Excel files, exploring datasets, visualizing numeric distributions, and training a Random Forest regression or classification model.

## 1. Requirements

- Python 3.10–3.12 recommended
- Internet connection for the initial package installation
- No GPU required

## 2. Run on Windows 10/11

Open Command Prompt or the VS Code terminal in this project folder.

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
py -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

If `py` is not recognized, install Python from https://www.python.org/downloads/ and enable **Add Python to PATH** during setup.

## 3. Use the app

1. Upload a `.csv`, `.xlsx`, or `.xls` file from the left sidebar.
2. Select **Exploratory Analysis** to inspect shape, missing values, summary statistics, distributions, and correlations.
3. Select **Predictive Model** to choose a target, prediction type, input features, and test size.
4. Click **Train Model**. Review the evaluation metrics and download test predictions.

## 4. Notes and limitations

- The sample data is illustrative; model quality depends on data quality and feature selection.
- Random Forest is a useful baseline, not a guarantee of accurate predictions.
- The app performs basic imputation and categorical encoding. It does not automatically detect data leakage, time-series constraints, or causal relationships.
- For very large files, memory and training time depend on your computer. This starter app is intended for local/portfolio use; production deployments need upload limits, authentication, resource controls, monitoring, and security review.
- Excel support is provided by `openpyxl` (`.xlsx`) and `xlrd` (`.xls`).

## Project structure

```text
smart_data_science_studio/
├── app.py
├── requirements.txt
├── README.md
└── sample_sales_data.csv
```
