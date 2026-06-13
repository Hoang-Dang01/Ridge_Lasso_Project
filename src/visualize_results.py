# src/visualize_results.py
import os
import matplotlib.pyplot as plt
from utils import load_csv, train_test_split, mean_std, scale_features, center_target
from model import RidgeLassoPure

def main():
    # Setup paths relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    presentation_dir = os.path.join(project_dir, 'presentation')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(presentation_dir, exist_ok=True)
    
    logistics_data_path = os.path.join(data_dir, 'logistics_data.csv')
    output_png = os.path.join(presentation_dir, 'logistics_results.png')
    
    # 1. Load logistics data
    headers, X, y = load_csv(logistics_data_path)
    
    # 2. Split train/test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    
    # 3. Scale features using training statistics
    means, stds = mean_std(X_train)
    X_train_scaled = scale_features(X_train, means, stds)
    X_test_scaled = scale_features(X_test, means, stds)
    
    # 4. Center target using training mean
    y_train_centered, mean_y_train = center_target(y_train)
    
    # 5. Fit Models
    model = RidgeLassoPure(X_train_scaled, y_train_centered)
    
    # - OLS (Ridge with alpha = 0.0)
    weights_ols = model.fit_ridge(0.0)
    # - Ridge (alpha = 1.0)
    weights_ridge = model.fit_ridge(1.0)
    # - Lasso (alpha = 1.0)
    weights_lasso = model.fit_lasso(1.0, iterations=1000)
    
    # 6. Evaluate Predictions on Test Set
    def predict(X_scaled, weights, intercept):
        preds = []
        for row in X_scaled:
            pred = sum(row[j] * weights[j] for j in range(len(weights))) + intercept
            preds.append(pred)
        return preds
        
    preds_ols = predict(X_test_scaled, weights_ols, mean_y_train)
    preds_ridge = predict(X_test_scaled, weights_ridge, mean_y_train)
    preds_lasso = predict(X_test_scaled, weights_lasso, mean_y_train)
    
    # Calculate MSE on Test Set
    def compute_mse(y_true, y_pred):
        return sum((t - p)**2 for t, p in zip(y_true, y_pred)) / len(y_true)
        
    mse_ols = compute_mse(y_test, preds_ols)
    mse_ridge = compute_mse(y_test, preds_ridge)
    mse_lasso = compute_mse(y_test, preds_lasso)
    
    print("Test MSE Comparison (Pure Python):")
    print(f"  OLS   MSE: {mse_ols:.4f}")
    print(f"  Ridge MSE: {mse_ridge:.4f}")
    print(f"  Lasso MSE: {mse_lasso:.4f}")
    
    print("\nCoefficients Comparison (Pure Python):")
    for h, w_o, w_r, w_l in zip(headers, weights_ols, weights_ridge, weights_lasso):
        print(f"  {h:15} | OLS: {w_o:8.4f} | Ridge: {w_r:8.4f} | Lasso: {w_l:8.4f}")

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    c_ols = '#94A3B8'
    c_ridge = '#3B82F6'
    c_lasso = '#10B981'
    
    # Plot 1: Coefficients Bar Chart
    x_indices = list(range(len(headers)))
    width = 0.25
    x_ols = [x - width for x in x_indices]
    x_ridge = x_indices
    x_lasso = [x + width for x in x_indices]
    
    ax1.bar(x_ols, weights_ols, width, label=f'OLS (MSE: {mse_ols:.2f})', color=c_ols, alpha=0.9, edgecolor='none')
    ax1.bar(x_ridge, weights_ridge, width, label=f'Ridge L2 (MSE: {mse_ridge:.2f})', color=c_ridge, alpha=0.9, edgecolor='none')
    ax1.bar(x_lasso, weights_lasso, width, label=f'Lasso L1 (MSE: {mse_lasso:.2f})', color=c_lasso, alpha=0.9, edgecolor='none')
    
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Standardized Coefficients')
    ax1.set_title('Feature Weights Comparison Across Models', fontweight='bold', pad=15)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(headers, rotation=15)
    ax1.legend(frameon=True)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    # Plot 2: Actual vs Predicted Scatter Plot on Test Set
    ax2.scatter(y_test, preds_ridge, color=c_ridge, alpha=0.7, edgecolors='none', label='Ridge L2', s=40)
    ax2.scatter(y_test, preds_lasso, color=c_lasso, alpha=0.7, edgecolors='none', label='Lasso L1', s=40)
    
    all_vals = y_test + preds_ridge + preds_lasso
    min_val = min(all_vals)
    max_val = max(all_vals)
    ax2.plot([min_val, max_val], [min_val, max_val], color='#EF4444', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')
    
    ax2.set_xlabel('Actual Inventory Level')
    ax2.set_ylabel('Predicted Inventory Level')
    ax2.set_title('Actual vs Predicted Values on Test Set', fontweight='bold', pad=15)
    ax2.legend(frameon=True)
    
    plt.suptitle('Regression Model Visualization: Ridge vs Lasso on Logistics Data', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print("\nSuccessfully generated and saved static plot in presentation/ folder!")

if __name__ == '__main__':
    main()
