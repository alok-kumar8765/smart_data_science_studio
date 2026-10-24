from __future__ import annotations

import io
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Smart Data Science Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Visual styling inspired by the supplied screenshot: pale-blue sidebar,
# compact dataset table, and a clean white analysis workspace.
st.markdown(
    """
    <style>
      .stApp { background: #ffffff; color: #202733; }
      section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #edf3f9 0%, #e8f0f7 100%);
        border-right: 1px solid #dce5ed;
      }
      section[data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem;
      }
      h1 { font-size: 2rem !important; font-weight: 750 !important; letter-spacing: -.035em; }
      h2, h3 { color: #202733; font-weight: 700 !important; }
      div[data-testid="stDataFrame"] { border: 1px solid #e5e9ef; border-radius: 8px; }
      div[data-testid="stMetric"] {
        background: #f7f9fc; border: 1px solid #e8edf3;
        padding: 12px 14px; border-radius: 10px;
      }
      .success-note {
        background: #d8f4df; color: #176534; padding: 12px 14px;
        border-radius: 8px; font-size: .92rem; margin: 8px 0 22px 0;
      }
      .muted { color: #64748b; font-size: .92rem; }
      .stButton > button { border-radius: 8px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Smart Data Science Studio")

with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="Upload a .csv, .xlsx, or .xls dataset.",
    )
    st.caption("Supported formats: CSV, XLSX, XLS")

    df: pd.DataFrame | None = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                # Try common encodings to avoid failing on non-UTF8 CSV files.
                raw = uploaded_file.getvalue()
                try:
                    df = pd.read_csv(io.BytesIO(raw))
                except UnicodeDecodeError:
                    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
            else:
                df = pd.read_excel(uploaded_file)
            # Normalize column names without silently changing the displayed names.
            if df.empty:
                st.error("The uploaded file has no rows. Please choose another dataset.")
                df = None
            elif len(df.columns) == 0:
                st.error("No columns were found in this file.")
                df = None
            else:
                st.markdown(
                    f'<div class="success-note">✓ Data Loaded Successfully<br>'
                    f'<span style="font-size:.82rem">{uploaded_file.name} · '
                    f'{uploaded_file.size / 1024:.1f} KB</span></div>',
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")
            st.info("Check that the file is a valid CSV or Excel workbook.")

    st.header("2. Select Task")
    task = st.radio(
        "Choose what you want to do",
        ["Exploratory Analysis", "Predictive Model"],
        index=0,
        label_visibility="collapsed",
    )

if df is None:
    st.subheader("Dataset Preview")
    st.info("Upload a CSV or Excel file in the left panel to get started.")
    st.markdown("### Explore your data")
    st.write(
        "Preview your dataset, inspect missing values, visualize numeric columns, "
        "and train a machine-learning model without writing code."
    )
    st.markdown(
        """
        **Included tools**
        - CSV and Excel upload
        - Automated dataset summary and missing-value checks
        - Interactive numeric distributions and correlation heatmap
        - Random Forest regression or classification
        - Evaluation metrics and downloadable predictions
        """
    )
    st.stop()

# Remove completely empty columns, while keeping a note of the original dataset.
empty_columns = [str(c) for c in df.columns[df.isna().all()]]
if empty_columns:
    st.warning("Entirely empty columns detected: " + ", ".join(empty_columns))

#st.subheader("Dataset Preview")
#st.dataframe(df.head(10), use_container_width=True, height=220)

st.subheader("Dataset Preview")

if df is None:
    st.warning("No dataset is available.")
else:
    st.write("Dataset loaded successfully.")
    st.write("Dataset shape:", df.shape)
    st.write("Column names:", df.columns.tolist())

    st.write("First 10 rows:")
    st.dataframe(
        df.head(10),
        use_container_width=True,
        height=300
    )

    if df.empty:
        st.error("The dataset contains no rows.")
    elif len(df.columns) == 1:
        st.warning(
            "Only one column was detected. "
            "Check whether your CSV uses commas or semicolons "
            "to separate columns."
        )

if task == "Exploratory Analysis":
    st.subheader("Exploratory Data Analysis")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{df.shape[0]:,}")
    metric_cols[1].metric("Columns", f"{df.shape[1]:,}")
    metric_cols[2].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    metric_cols[3].metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    st.markdown("**Shape:** " + str(df.shape))
    st.markdown("**Missing Values:**")
    missing = df.isnull().sum().rename("Missing Values").to_frame()
    missing.index.name = "Column"
    st.dataframe(missing, use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.subheader("Numeric Columns Distribution")
        selected_col = st.selectbox("Select Column", numeric_cols)
        values = pd.to_numeric(df[selected_col], errors="coerce").dropna()
        if values.empty:
            st.warning("This column has no valid numeric values to plot.")
        else:
            fig, ax = plt.subplots(figsize=(9, 4.2))
            sns.histplot(values, kde=(len(values) > 1 and values.nunique() > 1),
                         ax=ax, color="#83c5d8", edgecolor="white", linewidth=.5)
            ax.set_xlabel(str(selected_col))
            ax.set_ylabel("Count")
            ax.grid(axis="y", alpha=.22)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with st.expander("View numeric summary statistics"):
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

        if len(numeric_cols) >= 2:
            with st.expander("View numeric correlation heatmap"):
                corr = df[numeric_cols].corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(9, max(4, len(numeric_cols) * .45)))
                sns.heatmap(corr, annot=len(numeric_cols) <= 8, cmap="Blues",
                            fmt=".2f", linewidths=.4, ax=ax)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
    else:
        st.info("No numeric columns were detected, so numeric distribution charts are unavailable.")

    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if categorical_cols:
        with st.expander("Explore a categorical column"):
            cat_col = st.selectbox("Select category", categorical_cols, key="cat_column")
            counts = df[cat_col].astype("string").fillna("(Missing)").value_counts().head(20)
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.barplot(x=counts.values, y=counts.index.astype(str), color="#83c5d8", ax=ax)
            ax.set_xlabel("Count")
            ax.set_ylabel(str(cat_col))
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

else:
    st.subheader("Train a Machine Learning Model")
    st.caption(
        "Choose a target column. The app automatically imputes missing values and "
        "converts categorical features into model-ready columns."
    )

    usable_cols = [c for c in df.columns if df[c].notna().any()]
    if len(usable_cols) < 2:
        st.error("The dataset needs at least two non-empty columns to train a model.")
        st.stop()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    default_target_index = len(numeric_cols) - 1 if numeric_cols else len(usable_cols) - 1
    target_options = usable_cols
    target = st.selectbox(
        "Select Target Column (what should the model predict?)",
        target_options,
        index=target_options.index(numeric_cols[-1]) if numeric_cols else default_target_index,
    )

    target_series = df[target]
    inferred_classification = (
        target_series.dtype == "object"
        or str(target_series.dtype) == "category"
        or target_series.dtype == "bool"
    )
    model_type = st.radio(
        "Prediction type",
        ["Classification", "Regression"],
        index=0 if inferred_classification else 1,
        horizontal=True,
    )

    feature_options = [c for c in usable_cols if c != target]
    default_features = feature_options
    selected_features = st.multiselect(
        "Select input features",
        options=feature_options,
        default=default_features,
        help="Leave all selected to use every other non-empty column.",
    )
    test_size = st.slider("Test data size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)

    if st.button("Train Model", type="primary", use_container_width=False):
        if not selected_features:
            st.error("Select at least one input feature.")
            st.stop()

        model_df = df[selected_features + [target]].copy()
        model_df = model_df.dropna(subset=[target])
        if len(model_df) < 5:
            st.error("At least 5 rows with a known target value are needed.")
            st.stop()

        X = model_df[selected_features]
        y = model_df[target]

        if model_type == "Regression":
            y_numeric = pd.to_numeric(y, errors="coerce")
            valid_mask = y_numeric.notna()
            X = X.loc[valid_mask]
            y = y_numeric.loc[valid_mask]
            if len(y) < 5:
                st.error("The selected target must contain at least 5 valid numeric values for regression.")
                st.stop()
        else:
            # Classification labels must be present and sufficiently represented.
            y = y.astype(str)
            class_counts = y.value_counts()
            if len(class_counts) < 2:
                st.error("Classification requires at least two distinct target classes.")
                st.stop()
            if class_counts.min() < 2:
                st.warning("Some target classes have only one row; a stratified split is not possible.")

        if len(X) < 5:
            st.error("Not enough usable rows remain after cleaning.")
            st.stop()

        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        transformers = []
        if numeric_features:
            transformers.append((
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ))
        if categorical_features:
            transformers.append((
                "categorical",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        if model_type == "Regression":
            estimator = RandomForestRegressor(
                n_estimators=150, random_state=int(random_state), n_jobs=-1
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=150, random_state=int(random_state), n_jobs=-1,
                class_weight="balanced"
            )

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        stratify = None
        if model_type == "Classification":
            counts = y.value_counts()
            if counts.min() >= 2 and int(len(y) * test_size) >= len(counts):
                stratify = y

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=int(random_state),
                stratify=stratify,
            )
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            st.success("Model trained successfully.")
            st.subheader("Model Evaluation")
            if model_type == "Regression":
                col1, col2, col3 = st.columns(3)
                col1.metric("R² score", f"{r2_score(y_test, predictions):.3f}")
                col2.metric("Mean absolute error", f"{mean_absolute_error(y_test, predictions):,.3f}")
                col3.metric("Root mean squared error", f"{np.sqrt(mean_squared_error(y_test, predictions)):,.3f}")
            else:
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.1%}")
                col2.metric("Test rows", f"{len(y_test):,}")
                with st.expander("Classification report"):
                    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
                    st.dataframe(pd.DataFrame(report).T, use_container_width=True)
                with st.expander("Confusion matrix"):
                    labels = sorted(pd.Series(y_test).astype(str).unique().tolist())
                    matrix = confusion_matrix(y_test.astype(str), pd.Series(predictions).astype(str), labels=labels)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                                xticklabels=labels, yticklabels=labels, ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

            results = X_test.copy()
            results.insert(0, "Actual", np.asarray(y_test))
            results["Predicted"] = predictions
            st.subheader("Predictions Preview")
            st.dataframe(results.head(100), use_container_width=True)
            st.download_button(
                "Download test predictions as CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="model_predictions.csv",
                mime="text/csv",
            )

            try:
                importances = pipeline.named_steps["model"].feature_importances_
                feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances,
                }).sort_values("Importance", ascending=False).head(20)
                with st.expander("Top feature importances"):
                    fig, ax = plt.subplots(figsize=(9, max(3.5, len(importance_df) * .3)))
                    sns.barplot(data=importance_df, x="Importance", y="Feature",
                                color="#83c5d8", ax=ax)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            except Exception:
                st.caption("Feature importance is not available for this model configuration.")
        except Exception as exc:
            st.error(f"Model training failed: {exc}")
            st.info(
                "Try choosing a different target, reducing the number of features, "
                "or checking that the target column contains suitable values."
            )
