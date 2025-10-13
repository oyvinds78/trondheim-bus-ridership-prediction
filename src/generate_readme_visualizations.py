import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Create visualizations directory
viz_dir = Path('visualizations')
viz_dir.mkdir(exist_ok=True)

# Load holdout test set
print("Loading test data...")
holdout = joblib.load('Data/processed/final_holdout_test_set_advanced.pkl')
print(f"Test set shape: {holdout.shape}")
print(f"Columns: {len(holdout.columns)}")

# Split features and target
target_col = 'boardings'
feature_cols = [col for col in holdout.columns if col != target_col]

X_test = holdout[feature_cols]
y_test = holdout[target_col]

print(f"Features: {X_test.shape}")
print(f"Target: {y_test.shape}")
print(f"Target mean: {y_test.mean():.2f}, median: {y_test.median():.2f}")

# Load trained model
print("\nLoading model...")
model = joblib.load('src/models/final_bus_ridership_model.joblib')
# Load trained model
print("\nLoading model...")
model_data = joblib.load('src/models/final_bus_ridership_model.joblib')
model = model_data['model']  # Extract the actual model
trained_features = model_data['features'] # Get exact features used during training
print(f"Model type: {type(model)}")
print(f"CV MAE from training: {model_data['cv_mae']:.2f}")
print(f"Number of features expected: {len(trained_features)}")

# Split features and target - USE ONLY THE TRAINED FEATURES
target_col = 'boardings'
X_test = holdout[trained_features]  # Use only features the model was trained on
y_test = holdout[target_col]
print(f"Features: {X_test.shape}")
print(f"Target: {y_test.shape}")
print(f"Target mean: {y_test.mean():.2f}, median: {y_test.median():.2f}")

# Generate predictions
print("Generating predictions...")
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Model Performance Metrics:")
print(f"{'='*50}")
print(f"MAE:  {mae:.2f} passengers/hour")
print(f"RMSE: {rmse:.2f} passengers/hour")
print(f"R²:   {r2:.4f} ({r2*100:.2f}%)")
print(f"{'='*50}\n")

# ============================================
# Visualization 1: Full Range Scatter Plot
# ============================================
print("Creating full-range scatter plot...")
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue', edgecolors='none')

# Perfect prediction line
max_val = max(y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual Ridership (passengers/hour)', fontsize=12)
plt.ylabel('Predicted Ridership (passengers/hour)', fontsize=12)
plt.title('Model Performance: Actual vs Predicted Ridership', fontsize=14, fontweight='bold')

# Add stats to plot
stats_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}'
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = viz_dir / 'actual_vs_predicted.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================
# Visualization 2: Zoomed to Typical Range
# ============================================
print("Creating zoomed scatter plot...")

# Use 95th percentile for zoom
zoom_threshold = np.percentile(y_test, 95)
print(f"Zooming to 95th percentile: {zoom_threshold:.1f} passengers/hour")

mask = (y_test <= zoom_threshold) & (y_pred <= zoom_threshold)
pct_shown = (mask.sum() / len(mask)) * 100

plt.figure(figsize=(10, 8))
plt.scatter(y_test[mask], y_pred[mask], alpha=0.4, s=15, color='steelblue', edgecolors='none')
plt.plot([0, zoom_threshold], [0, zoom_threshold], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual Ridership (passengers/hour)', fontsize=12)
plt.ylabel('Predicted Ridership (passengers/hour)', fontsize=12)
plt.title(f'Model Performance: Typical Volume Range (95th percentile)\n{pct_shown:.1f}% of predictions shown', 
          fontsize=14, fontweight='bold')

# Recalculate metrics for this range
mae_zoom = mean_absolute_error(y_test[mask], y_pred[mask])
r2_zoom = r2_score(y_test[mask], y_pred[mask])
stats_text_zoom = f'MAE: {mae_zoom:.2f}\nR²: {r2_zoom:.2f}\n(0-{zoom_threshold:.0f} range)'

plt.text(0.05, 0.95, stats_text_zoom, transform=plt.gca().transAxes, 
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path_zoom = viz_dir / 'actual_vs_predicted_zoomed.png'
plt.savefig(output_path_zoom, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path_zoom}")
plt.close()

# ============================================
# Bonus: Residual Distribution
# ============================================
print("Creating residual plot...")
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.xlabel('Prediction Error (Actual - Predicted)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')

# Add stats
median_error = np.median(residuals)
stats_text_resid = f'Median Error: {median_error:.2f}\nMAE: {mae:.2f}'
plt.text(0.72, 0.95, stats_text_resid, transform=plt.gca().transAxes, 
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

output_path_resid = viz_dir / 'residual_distribution.png'
plt.savefig(output_path_resid, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path_resid}")
plt.close()

print("\n" + "="*50)
print("✓ All visualizations complete!")
print("="*50)
print("\nAdd to README.md:")
print("```markdown")
print("![Model Performance](visualizations/actual_vs_predicted.png)")
print("```")