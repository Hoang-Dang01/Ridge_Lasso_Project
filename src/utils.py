# src/utils.py
import csv
import math
import random

def load_csv(filename):
    """Loads CSV data and returns headers, feature matrix X, and target vector y."""
    X = []
    y = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if not row:
                continue
            row_float = [float(val) for val in row]
            X.append(row_float[:-1])
            y.append(row_float[-1])
    return headers[:-1], X, y

def train_test_split(X, y, test_size=0.2, seed=42):
    """Splits features X and labels y into random training and testing sets."""
    random.seed(seed)
    data = list(zip(X, y))
    random.shuffle(data)
    
    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)
    
    return list(X_train), list(X_test), list(y_train), list(y_test)

def mean_std(X):
    """Computes the mean and standard deviation for each feature column in X."""
    n_samples = len(X)
    n_features = len(X[0])
    means = [0.0] * n_features
    stds = [0.0] * n_features
    
    for col in range(n_features):
        col_sum = sum(X[row][col] for row in range(n_samples))
        means[col] = col_sum / n_samples
        
        col_sq_diff = sum((X[row][col] - means[col])**2 for row in range(n_samples))
        stds[col] = math.sqrt(col_sq_diff / n_samples)
    return means, stds

def scale_features(X, means, stds):
    """Standardizes X by subtracting mean and dividing by standard deviation (Z-score)."""
    n_samples = len(X)
    n_features = len(X[0])
    X_scaled = []
    for row in range(n_samples):
        scaled_row = []
        for col in range(n_features):
            if stds[col] > 0.0:
                scaled_row.append((X[row][col] - means[col]) / stds[col])
            else:
                scaled_row.append(0.0)
        X_scaled.append(scaled_row)
    return X_scaled

def center_target(y):
    """Centers the target variable y by subtracting its mean."""
    mean_y = sum(y) / len(y)
    y_centered = [val - mean_y for val in y]
    return y_centered, mean_y
