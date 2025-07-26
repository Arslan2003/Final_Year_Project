import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# ----- SETTINGS -----
INPUT_WINDOW = 108  # Apr 2015 – Mar 2024
OUTPUT_WINDOW = 12  # Apr 2024 – Mar 2025
EPOCHS = 100
LR = 0.001
HIDDEN_SIZE = 32

# ----- LOAD AND PREPARE DATA -----
df = pd.read_csv("../../data/stock_prices_pipeline/scrapping/monthly_avg_stock_prices.csv", index_col=0)
print(df.head())
df = df.loc[:, '2015-04':'2025-03']
df = df.ffill(axis=1).bfill(axis=1)

# # Choose one company
# company = "4GLOBAL PLC"  # <- Change as needed
# series = df.loc[company].values.astype(np.float32).reshape(-1, 1)
#
# # Normalize
# scaler = StandardScaler()
# series_scaled = scaler.fit_transform(series)
#
# # Split
# X = torch.tensor(series_scaled[:INPUT_WINDOW], dtype=torch.float32).unsqueeze(0)
# y = torch.tensor(series_scaled[INPUT_WINDOW:INPUT_WINDOW + OUTPUT_WINDOW], dtype=torch.float32).squeeze()


# Choose one company
company = "4GLOBAL PLC"  # <- Change as needed
series = df.loc[company].values.astype(np.float32).reshape(-1, 1)

# ----- Split raw data -----
train_series = series[:INPUT_WINDOW]  # Apr 2015 – Mar 2024
target_series = series[INPUT_WINDOW:INPUT_WINDOW + OUTPUT_WINDOW]  # Apr 2024 – Mar 2025

# ----- Standardize only on training data -----
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_series)
target_scaled = scaler.transform(target_series)

# ----- Convert to tensors -----
X = torch.tensor(train_scaled, dtype=torch.float32).unsqueeze(0)  # shape: (1, 108, 1)
y = torch.tensor(target_scaled, dtype=torch.float32).squeeze()    # shape: (12,)


# ----- MODEL -----
class BasicLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, OUTPUT_WINDOW)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = BasicLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ----- TRAINING LOOP -----
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output.squeeze(), y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# ----- PREDICTION -----
model.eval()
with torch.no_grad():
    pred_scaled = model(X).squeeze().numpy()
    pred_actual = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()


# ----- EVALUATE -------
mse = mean_squared_error(df.loc[company].values[INPUT_WINDOW:INPUT_WINDOW+OUTPUT_WINDOW], pred_actual)
print(mse)
print(df.loc[company].values[INPUT_WINDOW:INPUT_WINDOW+OUTPUT_WINDOW])
print(pred_actual)
print(df.loc[company].values[INPUT_WINDOW:INPUT_WINDOW+OUTPUT_WINDOW] - pred_actual)


# ----- PLOT -----
dates = df.columns[INPUT_WINDOW:INPUT_WINDOW+OUTPUT_WINDOW]
plt.plot(dates, pred_actual, label="Predicted")
plt.plot(dates, df.loc[company].values[INPUT_WINDOW:INPUT_WINDOW+OUTPUT_WINDOW], label="Actual")
plt.xticks(rotation=45)
plt.title(f"{company} - Stock Price Prediction (Apr 2024 – Mar 2025)")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()
