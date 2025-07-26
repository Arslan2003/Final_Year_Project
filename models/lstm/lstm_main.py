import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import optuna

# ----- SETTINGS -----
INPUT_WINDOW = 108  # Apr 2015 – Mar 2024
OUTPUT_WINDOW = 12  # Apr 2024 – Mar 2025

# ----- DEVICE CONFIGURATION -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- LOAD AND PREPARE DATA -----
df = pd.read_csv("../../data/stock_prices_pipeline/scrapping/monthly_avg_stock_prices.csv", index_col=0)
df = df.loc[:, '2015-04':'2025-03']
df = df.ffill(axis=1).bfill(axis=1)

# Choose one company
company = "3I INFRASTRUCTURE PLC"
series = df.loc[company].values.astype(np.float32).reshape(-1, 1)


# ----- Standardize -----
scaler = StandardScaler()
series_scaled = scaler.fit_transform(series)

# ----- Split raw data -----
train_series = series_scaled[:INPUT_WINDOW]
target_series = series_scaled[INPUT_WINDOW:INPUT_WINDOW + OUTPUT_WINDOW]

# ----- Convert to tensors and move to device -----
X = torch.tensor(train_series, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 108, 1)
y = torch.tensor(target_series, dtype=torch.float32).squeeze().to(device)    # (12,)


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


# ----- EARLY STOPPING CLASS -----
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_weights = None

    def step(self, loss, model):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_weights = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def restore_best_weights(self, model):
        model.load_state_dict(self.best_weights)


# ----- OBJECTIVE FUNCTION FOR OPTUNA -----
def objective(trial):
    hidden_size = trial.suggest_int('hidden_size', 16, 128)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    epochs = trial.suggest_int('epochs', 50, 300)

    model = BasicLSTM(hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Early stopping instance
    early_stopping = EarlyStopping(patience=10, min_delta=1e-5)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.squeeze(), y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        # Check early stopping
        if early_stopping.step(loss.item(), model):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Restore best model weights
    early_stopping.restore_best_weights(model)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X).squeeze().cpu().numpy()
        pred_actual = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    mse = mean_squared_error(df.loc[company].values[INPUT_WINDOW:INPUT_WINDOW + OUTPUT_WINDOW], pred_actual)
    return mse


# ----- OPTUNA OPTIMIZATION -----
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("Best Hyperparameters: ", study.best_params)
print("Best MSE: ", study.best_value)

# ----- PREDICTION WITH BEST PARAMETERS -----
best_params = study.best_params
model = BasicLSTM(hidden_size=best_params['hidden_size'], num_layers=best_params['num_layers']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
epochs = best_params['epochs']
criterion = nn.MSELoss()

# Early stopping instance
early_stopping = EarlyStopping(patience=10, min_delta=1e-5)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output.squeeze(), y)
    loss.backward()
    optimizer.step()

    # Check early stopping
    if early_stopping.step(loss.item(), model):
        print(f"Early stopping at epoch {epoch + 1}")
        break

# Restore best model weights
early_stopping.restore_best_weights(model)

# ----- PREDICTION -----
model.eval()
with torch.no_grad():
    pred_scaled = model(X).squeeze().cpu().numpy()
    pred_actual = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

# ----- PRINT RESULTS -----
dates = df.columns[INPUT_WINDOW:INPUT_WINDOW + OUTPUT_WINDOW]
actual_values = df.loc[company].values[INPUT_WINDOW:INPUT_WINDOW + OUTPUT_WINDOW]

print("\n--- Predicted vs Actual ---")
for date, pred, actual in zip(dates, pred_actual, actual_values):
    print(f"{date}: Predicted = {pred:.2f}, Actual = {actual:.2f}")

# ----- PLOT RESULTS -----
plt.plot(dates, pred_actual, label="Predicted")
plt.plot(dates, actual_values, label="Actual")
plt.xticks(rotation=45)
plt.title(f"{company} - Stock Price Prediction (Apr 2024 – Mar 2025)")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()
