# Final Year Project – Multimodal Machine Learning for Stock Recommendation

This repository contains the code and datasets used in my Final-Year Project for the BSc (Hons) in Data Science at University of Greenwich. The project explores a **multimodal machine learning approach** to evaluate publicly listed companies by combining **tabular analysis of financial metrics** with **time series modelling of historical stock prices**.

The project 

The methodology is inspired by value investing principles from *The Intelligent Investor* by Benjamin Graham. It seeks to determine whether a stock is **overvalued**, **fairly valued**, or **undervalued** based on both its historical performance and its current financial condition.

## Skills I gained from this project:

- Complete end-to-end implementation: from project ideation and data acquisition to model training, optimisation, and deployment
- Demonstrates skills in time series forecasting, financial data analysis, and modern ensemble learning (CatBoost)
- Designed with modular, scalable code architecture and reproducibility in mind




---

## 📁 Project Structure

```
Final_Year_Project/
├── data/
│   ├── fin_metrics_pipeline/
│   ├── stock_prices_pipeline/
│   ├── ticker_finder_pipeline/
│   └── readable-issuer_list_archive_2025-02-03.csv
│
├── models/
│   ├── catboost/
│   ├── lstm/
│   └── metamodel/
│
└── README.md
```

* **`catboost/`**: Gradient boosting model for classifying company valuation using financial metrics
* **`lstm/`**: LSTM model for predicting future stock prices using 10 years of monthly average historical data
* **`data/`**: Contains preprocessed datasets used in training and evaluation
* **`stock_prices_pipeline/`**: Scripts to download and preprocess stock price data from Yahoo Finance
* **`fin_metrics_pipeline/`**: Scripts to download and preprocess the most recent financial metrics of each company
* **`ticker_finder/`**: mapping companies to their stock tickers to later find their data on Yahoo Finance
* **`readable-issuer_list_archive_2025-02-03.csv`**: list of all LSE-domiciled companies from London Stock Exchange Group

---

## ⚙️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Arslan2003/Final_Year_Project.git
   cd Final_Year_Project
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate     # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

> **Note**: You will need GPU-compatible versions of PyTorch for optimal LSTM training. CatBoost can also benefit from a GPU if available.

---

## ▶️ Usage

### 📈 Train the CatBoost classifier

To train the model that classifies companies as **undervalued**, **fairly valued**, or **overvalued** based on financial metrics:

```bash
cd catboost
python catboost_main.py
```

This will:

* Load preprocessed training, validation, and test data from `../../data/fin_metrics_pipeline/`
* Train the CatBoostClassifier
* Save the model and evaluation reports in the `catboost_model.cbm` file

---

### 📉 Train the LSTM stock price predictor

To train the model that forecasts future stock prices using 10 years of historical monthly data:

```bash
cd lstm
python lstm_main.py
```

This will:

* Load average monthly prices from `../../data/monthly_avg_stock_prices.csv`
* Train the model on the 9 years of historical monthly prices, and make predictions for the next 12 months
* Save prediction plots and metrics in the `lstm_model.pt` file

---

## 📊 Results

The results were evaluated on approximately **1,277 companies** listed on the **London Stock Exchange**, using:

* 10 years of **monthly historical stock prices** (April 2015 – March 2025)
* Most recent **financial metrics** (e.g., P/E ratio, revenue, total debt, etc.)

### Label Generation:

A weak supervision approach based on *The Intelligent Investor* principles was used to create soft valuation labels from financial indicators such as debt levels, earnings stability, and market capitalisation.

---

## 📈 Visualisations

Visual outputs include:

* Loss curves during LSTM training
* Forecast vs actual price charts
* CatBoost feature importances (gain, split count)
* Distribution of predicted valuations by industry

---

## 🧠 Model Performance

### CatBoost Classifier (Financial Metrics)

* **Inputs**: 36 numerical features + 1 categorical industry feature
* **Target**: 3-class ordinal label (undervalued / fairly valued / overvalued)
* **Metrics**:

  * Accuracy: \~\[Insert]
  * F1 Score: \~\[Insert]
  * Handles missing values and categorical features natively

### LSTM Forecasting Model

* **Inputs**: 108 months of historical price data
* **Target**: 12-month forward prediction
* **Metrics**:

  * MAE: \~\[Insert]
  * RMSE: \~\[Insert]
  * Generalised across companies using many-to-one supervised learning

---

## 🔮 Future Work

This repository is no longer actively maintained. However, future contributors may consider:

* Integrating CatBoost and LSTM outputs into a unified valuation system
* Enhancing label quality with third-party analyst ratings or stock screeners
* Applying the framework to other stock markets (e.g., NASDAQ, FTSE 100)
* Incorporating NLP from earnings reports or sentiment analysis from news sources
* Exploring alternative time-series models like Temporal Fusion Transformers (TFTs)

If you build upon this project, feel free to fork it and cite it in your own work.

---

## 📚 Acknowledgements

* [Yahoo Finance](https://finance.yahoo.com) – Data source
* [CatBoost](https://catboost.ai/) – Gradient boosting classifier
* [PyTorch](https://pytorch.org/) – Deep learning library used for LSTM
* *The Intelligent Investor* by Benjamin Graham – Value investing inspiration
* [Optuna](https://optuna.org/) – For hyperparameter tuning

---

## 📬 Contact

**Arslan Ishanov**
Data Science Undergraduate
[LinkedIn](https://www.linkedin.com/in/arslan-ishanov)
