# Final Year Project – Multimodal Machine Learning for Stock Recommendation

This project implements an end-to-end multimodal machine learning pipeline to evaluate and recommend stocks listed on the London Stock Exchange. By combining historical stock price trends with company financial metrics, the project aims to provide a robust assessment of a company's investment potential. Unlike traditional single-source approaches, this system leverages multiple data modalities (numerical financial indicators and time-series stock data), allowing it to capture both long-term market trends and company-specific fundamentals.

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
* **`readable-issuer_list_archive_2025-02-03.csv`**: list of all LSE-domiciled companies from the London Stock Exchange Group

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Arslan2003/Final_Year_Project.git
   cd Final_Year_Project
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:

   ```
   pip install -r requirements.txt
   ```

> **Note**: For GPU acceleration, install PyTorch with the appropriate CUDA version. See [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) for more information.


## ▶️ Usage

#### 1. Choose Your Datasets:
The project already comes with the required datasets:
- Financial Metrics, which is split into:
   - ```fin_metrics_train.csv``` - training dataset.
   - ```fin_metrics_val.csv``` - validation dataset.
   - ```fin_metrics_test.csv``` - testing dataset.
- ```monthly_avg_stock_prices.csv``` - Monthly-averaged prices for each company from April 2014 to March 2025.

<br>

The first dataset contains the latest financial metrics (as of March 2025), and the second - daily stock prices for each company at market close from April 2015 to March 2024. The provided datasets can be beneficial to save time since loading the latest data takes 2-4 hours for each dataset. However, pipelines are also offered to download the latest data.  

<br>

To download the latest financial metrics, run the ```fin_metrics_main.py``` in ```data/fin_metrics_pipiline/scrapping/```. To obtain the newest stock prices for the past decade, run the ```price_scrapper_main.py``` in ```data/stock_prices_pipeline```, followed by ```price_transformer.py``` to aggregate daily stock prices into month-averaged prices.

<br>

If you would also like to get the updated list of companies that are listed on the LSE, you would have to download their latest [issuer list](https://www.londonstockexchange.com/reports?tab=issuers). Make sure you delete any embedded logos from the file. After replacing ```readable-issuer_list_archive_2025-02-03.csv``` with your new issuer list, you can run ```ticker_finder.py``` to obtain the updated ``` found_tickers.csv```. This file can be used by both ```fin_metrics_main.py``` and ```price_scrapper_main.py``` to get the latest data.

> Tickers are used to find information about a company on Yahoo Finance through the yfinance library.

<br>

#### 2. Choose Your Models 
- **CatBoost**:  
```
python models/catboost/catboost_main.py
```

<br>

- **LSTM**:  
```
python models/lstm_main.py
```

<br>

> **Note**: By default, the files will load the pre-trained models, since the code to re-train, re-optimise, and re-evaluate them is commented out. The saved models are trained on the provided datasets. To build the models based on new data, uncomment the code before running it.

<br>

#### 3. Experiment!

<br>

## 📊 Results  

- **CatBoost Model** classified companies into valuation categories (e.g., undervalued vs. overvalued) using domain-driven financial metrics.  
- **LSTM Model** forecasted 12-month stock prices with strong accuracy across multiple companies, supporting forward-looking investment decisions.  
- **Multimodal Approach** combined fundamentals and market forecasts, delivering holistic stock recommendations that **outperformed the FTSE-100 benchmark**.  

#### Performance (01 April 2024 – 31 March 2025):  

- Multimodal Model Return: ![12.90%](https://img.shields.io/badge/Return-12.90%25-brightgreen)  
- FTSE-100 Benchmark: ![8.16%](https://img.shields.io/badge/Return-8.16%25-blue)  

>  The pipeline and models are designed to be extendable, allowing integration of additional data sources or advanced ensemble strategies in future work.



<!--- Add concrete results --->
---

## 🤝 Contributions

Contributions, suggestions, and improvements are welcome! Feel free to open an issue or submit a pull request. Here are a few starting ideas for improvements

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
* [The Intelligent Investor](https://irp-cdn.multiscreensite.com/cb9165b2/files/uploaded/The%20Intelligent%20Investor%20-%20BENJAMIN%20GRAHAM.pdf) by Benjamin Graham – Value investing inspiration
* [Optuna](https://optuna.org/) – For hyperparameter tuning

---

## 🧑‍💻 Author
[Arslonbek Ishanov](https://github.com/Arslan2003) - First-Class Graduate Data Scientist & AI/ML Enthusiast

<br>

## ⚖️ License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full terms.

<br>

## 🏷️ Tags
`Machine Learning` `Deep Learning` `Multimodal Learning` `Stock Recommendation` `Financial Modelling` `Time Series Forecasting` `CatBoost` `LSTM` `Investment Analysis` `Portfolio Optimisation` `FTSE-100` `Optuna` `Hyperparameter Tuning` `Value Investing` `Python` `PyTorch`



