import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load excel sheet
file_path = "DQN1 Dataset.xlsx"
df = pd.read_excel(file_path)

# Define target variable
target_col = "healthRiskScore"
x = df.drop(columns=[target_col])
y = df[target_col]

# Fill missing values
x = x.fillna(x.mean())

# Optimization Technique 1: Feature Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Store original model performance
original_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
original_model.fit(x_train, y_train)
y_pred_original = original_model.predict(x_test)
original_rmse = np.sqrt(mean_squared_error(y_test, y_pred_original))
original_mape = mean_absolute_percentage_error(y_test, y_pred_original)
original_r2 = r2_score(y_test, y_pred_original)

# Optimization Technique 2: Refined Grid Search
param_grid = {
    'n_estimators': [100, 150],
    'learning_rate': [0.05, 0.08],
    'max_depth': [3, 5],
    'subsample': [0.9, 1.0],
    'colsample_bytree': [0.9, 1.0],
    'min_child_weight': [1, 3]  # Added to control overfitting
}

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    # Regularization Technique 1: Reduced L1 regularization
    alpha=0.01,
    # Regularization Technique 2: Reduced L2 regularization
    reg_lambda=0.5
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(x_train, y_train)

# Get best model from grid search
best_xgb = grid_search.best_estimator_

# Ensemble Technique 1: Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    min_samples_split=5  # Added to control overfitting
)

# Ensemble Technique 2: Voting Regressor
voting_regressor = VotingRegressor([
    ('xgb', best_xgb),
    ('gb', gb_model)
])

# Train ensemble model
voting_regressor.fit(x_train, y_train)

# Predict with optimized ensemble model
y_pred_optimized = voting_regressor.predict(x_test)

# Evaluate optimized model performance
optimized_rmse = np.sqrt(mean_squared_error(y_test, y_pred_optimized))
optimized_mape = mean_absolute_percentage_error(y_test, y_pred_optimized)
optimized_r2 = r2_score(y_test, y_pred_optimized)

# Print results comparison
print("Original Model Performance:")
print(f"RMSE: {original_rmse:.2f}")
print(f"MAPE: {original_mape:.2%}")
print(f"R2 Score: {original_r2:.2f}")
print("\nOptimized Model Performance:")
print(f"RMSE: {optimized_rmse:.2f}")
print(f"MAPE: {optimized_mape:.2%}")
print(f"R2 Score: {optimized_r2:.2f}")
print("\nBest XGBoost Parameters from Grid Search:")
print(grid_search.best_params_)