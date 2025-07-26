import numpy as np
import pandas as pd

# Metric configurations
SCORING_CONFIG = {
    'P/E Ratio': {'weight': 2.0, 'limits': (5, 25)},
    'P/B Ratio': {'weight': 1.5, 'limits': (0.5, 3)},
    'Debt to Equity Ratio': {'weight': 1.5, 'limits': (0, 1)},
    'Dividend Yield': {'weight': 1.0, 'limits': (0.01, 0.05)}
}

def score_metric(value, limits, missing_penalty=-0.3):
    if pd.isna(value):
        return missing_penalty
    low, high = limits
    if value < low:
        return 2
    elif low <= value < (low + high) / 2:
        return 1
    elif (low + high) / 2 <= value <= high:
        return -1
    else:
        return -2

def evaluate_company(row):
    total = 0
    for metric, config in SCORING_CONFIG.items():
        score = score_metric(row.get(metric), config['limits'])
        total += score * config['weight']
    return np.clip(total, -10, 10)

def compute_quantile_labels(scores: pd.Series):
    quantiles = scores.quantile([0.08, 0.3, 0.7, 0.92])
    thresholds = {
        'low1': quantiles[0.08],
        'low2': quantiles[0.3],
        'high2': quantiles[0.7],
        'high1': quantiles[0.92]
    }

    def label(score):
        if score <= thresholds['low1']:
            return -2
        elif score <= thresholds['low2']:
            return -1
        elif score <= thresholds['high2']:
            return 0
        elif score <= thresholds['high1']:
            return 1
        else:
            return 2

    return scores.apply(label)

def apply_valuation(df):
    scores = df.apply(evaluate_company, axis=1)
    df['Valuation Label'] = compute_quantile_labels(scores)
    return df
