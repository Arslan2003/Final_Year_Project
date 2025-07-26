import os
import time
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import optuna

# ----- SETTINGS -----
INPUT_WINDOW = 108  # Apr 2015 – Mar 2024
OUTPUT_WINDOW = 12  # Apr 2024 – Mar 2025
MODEL_DIR = "../../models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "stock_scaler.pkl")

# ----- LOAD AND PREPARE DATA -----
df = pd.read_csv("../../data/stock_prices_pipeline/scrapping/monthly_avg_stock_prices.csv", index_col=0)
df = df.loc[:, '2015-04':'2025-03']
df = df.ffill(axis=1).bfill(axis=1)

# Train/test split
train_size = int(len(df) * 0.8)
train_companies = df.index[:train_size]
test_companies = df.index[train_size:]

# Normalize the entire dataset
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# ----- DEVICE SETUP -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- MODEL -----
class BasicLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=OUTPUT_WINDOW):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ----- OPTUNA OBJECTIVE FUNCTION -----
def objective(trial):
    hidden_size = trial.suggest_int('hidden_size', 16, 128)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    epochs = trial.suggest_int('epochs', 50, 300)  # was 50, 300

    model = BasicLSTM(hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for idx, company in tqdm(enumerate(train_companies), total=len(train_companies),
                                  desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            series = df_scaled[train_companies.get_loc(company)].reshape(-1, 1)
            series_scaled = scaler.fit_transform(series)
            X = torch.tensor(series_scaled[:INPUT_WINDOW], dtype=torch.float32).unsqueeze(0).to(device)
            y = torch.tensor(series_scaled[INPUT_WINDOW:INPUT_WINDOW + OUTPUT_WINDOW],
                             dtype=torch.float32).squeeze().to(device)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output.squeeze(), y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f"Epoch [{epoch + 1}/{epochs}] - Avg Loss: {np.mean(epoch_losses):.6f}")

    # ----- EVALUATION -----
    model.eval()
    total_mse = 0
    with torch.no_grad():
        for company in tqdm(test_companies, desc="Evaluating Test Set", leave=False):
            series = df_scaled[test_companies.get_loc(company)].reshape(-1, 1)
            series_scaled = scaler.fit_transform(series)
            X = torch.tensor(series_scaled[:INPUT_WINDOW], dtype=torch.float32).unsqueeze(0).to(device)
            y_actual = df.loc[company].values[INPUT_WINDOW:INPUT_WINDOW + OUTPUT_WINDOW]

            pred_scaled = model(X).squeeze().cpu().numpy()
            pred_actual = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            mse = mean_squared_error(y_actual, pred_actual)
            total_mse += mse

    return total_mse / len(test_companies)

# ----- RUN OPTUNA -----
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1)
print("Best Hyperparameters:", study.best_params)
print("Best MSE:", study.best_value)

# ----- FINAL TRAINING WITH BEST PARAMS -----
best_params = study.best_params
model = BasicLSTM(hidden_size=best_params['hidden_size'], num_layers=best_params['num_layers']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
criterion = nn.MSELoss()

for epoch in range(best_params['epochs']):
    model.train()
    epoch_losses = []
    for company in tqdm(train_companies, desc=f"Final Training Epoch {epoch + 1}/{best_params['epochs']}", leave=False):
        series = df_scaled[train_companies.get_loc(company)].reshape(-1, 1)
        series_scaled = scaler.fit_transform(series)
        X = torch.tensor(series_scaled[:INPUT_WINDOW], dtype=torch.float32).unsqueeze(0).to(device)
        y = torch.tensor(series_scaled[INPUT_WINDOW:INPUT_WINDOW + OUTPUT_WINDOW],
                         dtype=torch.float32).squeeze().to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.squeeze(), y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    print(f"Epoch [{epoch + 1}/{best_params['epochs']}] - Avg Loss: {np.mean(epoch_losses):.6f}")

# ----- SAVE MODEL AND SCALER -----
torch.save(model.state_dict(), MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"Model saved to: {MODEL_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")

# ----- PREDICTIONS AND PLOTTING -----
model.eval()
pred_all_companies = {}
with torch.no_grad():
    for company in test_companies:
        series = df_scaled[test_companies.get_loc(company)].reshape(-1, 1)
        series_scaled = scaler.fit_transform(series)
        X = torch.tensor(series_scaled[:INPUT_WINDOW], dtype=torch.float32).unsqueeze(0).to(device)

        pred_scaled = model(X).squeeze().cpu().numpy()
        pred_actual = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        pred_all_companies[company] = pred_actual

# Plot results
# for company, pred_actual in pred_all_companies.items():
#     actual_values = df.loc[company].values[INPUT_WINDOW:INPUT_WINDOW + OUTPUT_WINDOW]
#     dates = df.columns[INPUT_WINDOW:INPUT_WINDOW + OUTPUT_WINDOW]
#
#     print(f"\n--- {company} ---")
#     for date, pred, actual in zip(dates, pred_actual, actual_values):
#         print(f"{date}: Predicted = {pred:.2f}, Actual = {actual:.2f}")
#
#     plt.plot(dates, pred_actual, label="Predicted")
#     plt.plot(dates, actual_values, label="Actual")
#     plt.xticks(rotation=45)
#     plt.title(f"{company} - Stock Price Prediction (Apr 2024 – Mar 2025)")
#     plt.ylabel("Price")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
