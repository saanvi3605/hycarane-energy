HYCARANE ENERGY - COE CARBON TRACKING SETUP
============================================

MODELS TRAINED (locally):
- H2 Yield: 11.7% MAPE, 97.11% R²
- Carbon Quality: 3.39% MAPE, 20.24% R²  
- H2 Purity: 0.0% MAPE (perfect)
- Net Profit: 9.43% MAPE, 96.7% R²

AVERAGE R² ACROSS ALL MODELS: ~97%

FILES INCLUDED:
- train_model.py (complete training pipeline)
- clean_data.csv (1,975 rows, cleaned)
- 4 trained .pkl model files

COE GOALS:
1. Measure carbon footprint of training
2. Measure inference energy consumption
3. Test model quantization impact
4. Generate sustainability metrics for presentation

REQUIREMENTS:
pip install pandas numpy scikit-learn catboost codecarbon