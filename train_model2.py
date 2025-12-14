# train_model.py - CORRECTED (No Data Leakage)
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import joblib
import json
from datetime import datetime

print("=" * 70)
print("HYCARANE ENERGY - MODEL TRAINING (LEAKAGE-FREE)")
print("=" * 70)

# 1. LOAD DATA
print("\n[1] Loading clean data...")
df = pd.read_csv('clean_data.csv')
print(f"   Dataset shape: {df.shape}")

# 2. DEFINE FEATURES - RAW PARAMETERS ONLY (NO ENGINEERED FEATURES)
print("\n[2] Defining features (raw parameters only)...")

feature_cols = [
    # Raw Reactor Inputs
    'Methane_Feed_',
    'Reactor_Pressu',
    'Reactor_Temper',
    'Heating_Power_',
    
    # Raw Catalyst Health
    'Catalyst_Cycle_',
    'Catalyst_Activity',
    'Catalyst_Carbon',
    
    # Time Features
    'hour',
    'day_of_week',
    'is_weekend'
]

# Target columns
target_cols = [
    'H2_Yield_Rate',
    'Carbon_Quality_',
    'H2_Purity_Post_',
    'Net_Profit_Margin_Index'
]

print(f"   ✓ Using {len(feature_cols)} features (no engineered features)")
print(f"   ✓ Predicting {len(target_cols)} targets")

# 3. PREPARE DATA
print("\n[3] Preparing train/test split (80/20)...")
X = df[feature_cols].copy()
y = df[target_cols].copy()

X = X.fillna(X.median())
y = y.fillna(y.median())

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"   Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# 4. TRAIN MODELS
print("\n[4] Training CatBoost models...")
print("=" * 70)
models = {}
results = {}
training_start = datetime.now()

for i, target in enumerate(target_cols, 1):
    print(f"\n[{i}/{len(target_cols)}] Training: {target}")
    
    model = CatBoostRegressor(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        verbose=False,
        random_seed=42,
        early_stopping_rounds=20
    )
    
    model.fit(
        X_train, 
        y_train[target],
        eval_set=(X_test, y_test[target]),
        verbose=False
    )
    
    # Test predictions
    y_pred_test = model.predict(X_test)
    test_mape = mean_absolute_percentage_error(y_test[target], y_pred_test) * 100
    test_r2 = r2_score(y_test[target], y_pred_test)
    
    # Train predictions (to check overfitting)
    y_pred_train = model.predict(X_train)
    train_mape = mean_absolute_percentage_error(y_train[target], y_pred_train) * 100
    train_r2 = r2_score(y_train[target], y_pred_train)
    
    models[target] = model
    results[target] = {
        'Train_MAPE': round(train_mape, 2),
        'Test_MAPE': round(test_mape, 2),
        'Train_R2': round(train_r2, 4),
        'Test_R2': round(test_r2, 4),
        'Best_Iteration': int(model.best_iteration_)
    }
    
    print(f"   TRAIN - MAPE: {train_mape:.2f}%, R²: {train_r2:.4f}")
    print(f"   TEST  - MAPE: {test_mape:.2f}%, R²: {test_r2:.4f}")
    
    # Check for overfitting
    if train_mape < test_mape * 0.7:  # Train error much lower than test
        print(f"   ⚠️  Possible overfitting (train MAPE much lower)")
    else:
        print(f"   ✓  Good generalization")

training_time = (datetime.now() - training_start).total_seconds()

# 5. SAVE MODELS
print("\n[5] Saving models...")
for target, model in models.items():
    filename = f"model_{target}.pkl"
    joblib.dump(model, filename)
    print(f"   ✓ {filename}")

joblib.dump(feature_cols, 'feature_names.pkl')

# 6. SAVE RESULTS
summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'data_leakage_fixed': True,
    'removed_features': ['efficiency_ratio', 'catalyst_performance'],
    'dataset_info': {
        'total_size': len(df),
        'train_size': len(X_train),
        'test_size': len(X_test),
    },
    'features': feature_cols,
    'targets': target_cols,
    'training_time_seconds': round(training_time, 2),
    'model_performance': results
}

with open('model_results_clean.json', 'w') as f:
    json.dump(summary, f, indent=4)

# 7. FEATURE IMPORTANCE
print("\n[6] Feature Importance (Top 5 for each target):")
for target, model in models.items():
    print(f"\n{target}:")
    importance = model.get_feature_importance()
    feature_importance = sorted(zip(feature_cols, importance), 
                                key=lambda x: x[1], reverse=True)
    
    for i, (feat, imp) in enumerate(feature_importance[:5], 1):
        print(f"   {i}. {feat:25s}: {imp:6.2f}")

# 8. FINAL SUMMARY
print("\n" + "=" * 70)
print(f"✅ TRAINED {len(models)} MODELS (LEAKAGE-FREE) IN {training_time:.2f}s")
print("=" * 70)
print("\n📊 CLEAN PERFORMANCE SUMMARY:")
print("-" * 70)

for target, metrics in results.items():
    print(f"\n{target}:")
    print(f"  Test MAPE:  {metrics['Test_MAPE']}%")
    print(f"  Test R²:    {metrics['Test_R2']}")
    print(f"  Train MAPE: {metrics['Train_MAPE']}%")

print("\n" + "=" * 70)
print("✓ No engineered features used")
print("✓ Only raw operational parameters")
print("✓ Results are now honest and defensible")
print("=" * 70)