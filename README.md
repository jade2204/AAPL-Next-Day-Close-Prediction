# Apple Stock Price Prediction Using Linear Regression

## Project Overview

This project develops a machine learning model to predict the next trading day's closing price of Apple Inc. (AAPL) using historical stock market data.

The project applies data preprocessing, exploratory data analysis (EDA), feature engineering, feature selection, time-series cross-validation, and regression evaluation to build and assess a Linear Regression model.

The trained model is saved for future use, together with metadata describing the selected features and model evaluation results.

## Project Objectives

- Analyze historical AAPL stock price and trading volume data.
- Explore relationships between market variables through data visualization.
- Engineer technical features from historical price and volume data.
- Select relevant features using Recursive Feature Elimination with Cross-Validation (RFECV).
- Train a Linear Regression model to predict the next trading day's closing price.
- Evaluate model performance using regression metrics.
- Compare the trained model against a simple baseline prediction.
- Export the trained model and its metadata for future inference.

## Dataset Information

The dataset contains historical stock market data for Apple Inc. (AAPL).

Typical columns include:

- `Date`: Trading date.
- `Open`: Opening stock price.
- `High`: Highest stock price during the trading session.
- `Low`: Lowest stock price during the trading session.
- `Close`: Closing stock price.
- `Volume`: Number of shares traded.

The dataset is expected to be available locally at the path configured in the Python script.

**Note:** This project does not necessarily download the dataset automatically. Ensure that the input CSV file exists and that `DATA_FILE` points to the correct location.

## Feature Engineering

The project creates additional features from historical market data, including:

| Feature | Description |
|---|---|
| `Return_1d` | One-day percentage return |
| `Range_pct` | Daily high-low range relative to the closing price |
| `Open_Close_pct` | Difference between opening and closing prices |
| `MA_5` | Five-day moving average of closing prices |
| `MA_20` | Twenty-day moving average of closing prices |
| `Volatility_5` | Five-day rolling volatility of daily returns |
| `Volume_MA_5` | Five-day moving average of trading volume |
| `Volume_change` | Change in trading volume |
| `Target_Next_Close` | Closing price of the next trading day |

The target variable is the next trading day's closing price.

## Methodology

### 1. Data Preprocessing
- Load the historical stock dataset.
- Clean the data and handle missing values.
- Prepare the date and numerical columns for analysis.

### 2. Exploratory Data Analysis
- Visualize stock price trends.
- Examine the distribution of numerical variables.
- Detect potential outliers.
- Analyze correlations between features.

### 3. Feature Engineering
Create technical indicators and historical market features to represent price movements, volatility, and trading activity.

### 4. Feature Selection
Use mutual information for exploratory feature analysis and Recursive Feature Elimination with Cross-Validation (RFECV) to select features for the regression model.

### 5. Model Training
Train a Linear Regression model using a scikit-learn pipeline that includes feature scaling.

The dataset is split chronologically to preserve the temporal order of observations.

### 6. Model Evaluation
Evaluate predictions using:

- **MAE:** Mean Absolute Error.
- **RMSE:** Root Mean Squared Error.
- **R²:** Coefficient of Determination.
- **MAPE:** Mean Absolute Percentage Error.
- **Directional Accuracy:** Percentage of predictions that correctly capture the direction of price movement.

The model is also compared with a baseline that predicts the next closing price using the current closing price.

### 7. Model Export
Save the trained model using Joblib and export metadata containing information such as the model version, selected features, and evaluation metrics.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Project Structure

```text
AAPL-Stock-Price-Prediction/
├── data/
│   └── AAPL.csv
├── main.py
├── README.md
├── requirements.txt
├── aapl_linear_regression.joblib
└── model_metadata.json
```

The filenames and directory structure above are examples. Adjust them to match the actual files and paths used in your project. The dataset and generated model artifacts do not need to be committed to GitHub.

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AAPL-Stock-Price-Prediction
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can use Command Prompt to activate the environment instead:

```bat
venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the dataset

Place the AAPL historical stock data CSV file in the expected location and update `DATA_FILE` in the Python script if necessary.

### 5. Run the project

```bash
python main.py
```

Replace `main.py` with the actual Python filename if it differs.

## Output

Depending on the configuration, the project generates:

- Exploratory data analysis plots.
- Feature correlation and selection results.
- Model evaluation metrics.
- Actual-versus-predicted price visualizations.
- A serialized model artifact.
- JSON metadata containing model information and evaluation results.

## Limitations

Stock prices are affected by market conditions, company announcements, macroeconomic factors, and other variables not necessarily represented in this dataset.

A Linear Regression model captures linear relationships and may not adequately represent complex market behavior. Historical performance does not guarantee future predictive accuracy.

The model's performance should be assessed against a baseline and on unseen chronological test data. Care should also be taken to prevent temporal data leakage during feature engineering and cross-validation.

## Disclaimer

This project is for educational and research purposes only. It does not constitute financial or investment advice. Model predictions should not be used as the sole basis for investment decisions.

## Author

Developed as a machine learning practice project focused on regression, feature engineering, and time-series-aware model evaluation.