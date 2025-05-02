import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

# Load excel sheet
file_path = "DQN1 Dataset.xlsx"

# Load and display first few rows
df = pd.read_excel(file_path)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# Define target variable
target_col = "healthRiskScore"

# Define feature columns
x = df.drop(columns=[target_col])
y = df[target_col]

# Fill missing values (if any)
x = x.fillna(x.mean())

# Confirm shapes
print("Feature Shape:", x.shape)
print("Target Shape:", y.shape)

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Training set:", x_train.shape, y_train.shape)
print("Testing set:", x_test.shape, y_test.shape)

# Initialize the XGBoost regressor
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

# Train the model
model.fit(x_train, y_train)

# Predict on test set
y_pred = model.predict(x_test)

# Evaluate performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2%}")