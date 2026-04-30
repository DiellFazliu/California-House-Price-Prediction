# ============================================
# PROJECT 1: California House Price Prediction
# Author: Diell Fazliu
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 50)
print("CALIFORNIA HOUSING PRICE PREDICTION")
print("=" * 50)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n📊 Loading data...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(f"✅ Dataset shape: {df.shape}")

# ============================================
# 2. DATA PREPARATION
# ============================================
print("\n📊 Preparing data...")
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Train: {X_train.shape[0]} samples")
print(f"✅ Test: {X_test.shape[0]} samples")

# ============================================
# 3. TRAIN MODEL
# ============================================
print("\n📊 Training model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model trained")

# ============================================
# 4. PREDICT
# ============================================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ============================================
# 5. EVALUATE
# ============================================
print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n📈 TRAINING SET:")
print(f"   RMSE: ${train_rmse * 100000:.2f}")
print(f"   MAE:  ${train_mae * 100000:.2f}")
print(f"   R²:   {train_r2:.4f}")

print(f"\n📈 TEST SET:")
print(f"   RMSE: ${test_rmse * 100000:.2f}")
print(f"   MAE:  ${test_mae * 100000:.2f}")
print(f"   R²:   {test_r2:.4f}")

print(f"\n⚠️ Overfitting Check:")
r2_diff = train_r2 - test_r2
print(f"   R² difference: {r2_diff:.4f}")

if r2_diff > 0.05:
    print("   ⚠️ Possible overfitting detected")
elif r2_diff > 0.02:
    print("   📊 Mild overfitting")
else:
    print("   ✅ Model generalizes well")

# ============================================
# 6. VISUALIZE
# ============================================
print("\n📊 Generating plots...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_test_pred, alpha=0.5, s=10)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual House Value ($100k)')
axes[0].set_ylabel('Predicted House Value ($100k)')
axes[0].set_title(f'Actual vs Predicted (R² = {test_r2:.3f})')

# Plot 2: Residuals
residuals = y_test - y_test_pred
axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1].axvline(0, color='red', linestyle='--', lw=2)
axes[1].set_xlabel('Residual Error ($100k)')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Residual Distribution (Std = {residuals.std():.3f})')

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Results saved as 'model_results.png'")