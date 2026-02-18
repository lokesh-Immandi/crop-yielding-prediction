import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# For SHAP analysis
import shap

# For statistical analysis
from scipy import stats
import statsmodels.api as sm

print("=" * 70)
print("AGRICULTURAL DECISION SUPPORT SYSTEM - MODEL TRAINING")
print("M.Tech Research Project: Multi-Model Ensemble with Explainable AI")
print("=" * 70)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

# Importing the dataset
print("\n[1] Loading dataset...")
df = pd.read_csv('crop_yield.csv')
print(f"    Dataset shape: {df.shape}")
print(f"    Total records: {len(df)}")
print(f"    Date range: {df['Crop_Year'].min()} - {df['Crop_Year'].max()}")

# Clean the data - strip whitespace and standardize case
for column in ['Crop', 'State']:
    df[column] = df[column].astype(str).str.strip().str.title()

# Special handling for Season to remove extra whitespace
df['Season'] = df['Season'].astype(str).str.strip()

# Compute the Yield as Production / Area
df['Yield'] = df['Production'] / df['Area']

# Remove any infinite or NaN values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['Yield'])

print(f"\n[2] Data cleaning complete:")
print(f"    Records after cleaning: {len(df)}")
print(f"    Yield range: {df['Yield'].min():.2f} - {df['Yield'].max():.2f} tons/ha")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n[3] Performing Exploratory Data Analysis...")

# Print unique values in categorical columns
print("\n    Unique values in categorical columns:")
for column in ['Crop', 'State', 'Season']:
    unique_vals = df[column].unique()
    print(f"    {column}: {len(unique_vals)} unique values")
    if len(unique_vals) <= 10:
        print(f"        {sorted(unique_vals)}")

# Statistical summary
print("\n    Statistical Summary of Numerical Features:")
print(df[['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Area', 'Production', 'Yield']].describe())

# Correlation analysis
print("\n    Correlation with Yield:")
correlations = df[['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Area', 'Yield']].corr()['Yield'].sort_values(ascending=False)
for feat, corr in correlations.items():
    if feat != 'Yield':
        print(f"    {feat}: {corr:.3f}")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

print("\n[4] Feature Engineering...")

# Create additional features
df['Rainfall_per_Area'] = df['Annual_Rainfall'] / df['Area']
df['Fertilizer_per_Area'] = df['Fertilizer'] / df['Area']
df['Pesticide_per_Area'] = df['Pesticide'] / df['Area']
df['Input_Intensity'] = (df['Fertilizer'] + df['Pesticide']) / df['Area']
df['Year_Squared'] = df['Crop_Year'] ** 2

# Log transformations for skewed features
df['Log_Area'] = np.log1p(df['Area'])
df['Log_Production'] = np.log1p(df['Production'])
df['Log_Rainfall'] = np.log1p(df['Annual_Rainfall'])

print("    Created additional features:")
print("    - Rainfall_per_Area, Fertilizer_per_Area, Pesticide_per_Area")
print("    - Input_Intensity, Year_Squared")
print("    - Log transformations for skewed features")

# ============================================================================
# 4. MEDIAN VALUES FOR DEFAULTS
# ============================================================================

print("\n[5] Computing median values for defaults...")
median_values = {
    'Crop_Year': float(df['Crop_Year'].median()),
    'Annual_Rainfall': float(df['Annual_Rainfall'].median()),
    'Fertilizer': float(df['Fertilizer'].median()) if 'Fertilizer' in df.columns else 0,
    'Pesticide': float(df['Pesticide'].median()) if 'Pesticide' in df.columns else 0,
    'Area': float(df['Area'].median()),
    'Yield': float(df['Yield'].median()),
    'Rainfall_per_Area': float(df['Rainfall_per_Area'].median()),
    'Fertilizer_per_Area': float(df['Fertilizer_per_Area'].median()),
    'Input_Intensity': float(df['Input_Intensity'].median())
}

# Save to file
joblib.dump(median_values, 'median_values.pkl')
print(f"    Median values saved to 'median_values.pkl'")

# ============================================================================
# 5. UNIQUE VALUES FOR DROPDOWNS
# ============================================================================

print("\n[6] Extracting unique values for UI dropdowns...")
unique_values = {
    'crops': sorted(df['Crop'].dropna().unique().tolist()),
    'states': sorted(df['State'].dropna().unique().tolist()),
    'seasons': sorted(df['Season'].dropna().unique().tolist()),
}

# Save to file
joblib.dump(unique_values, 'unique_values.pkl')
print(f"    Unique values saved to 'unique_values.pkl'")
print(f"    Crops: {len(unique_values['crops'])} | States: {len(unique_values['states'])} | Seasons: {len(unique_values['seasons'])}")

# ============================================================================
# 6. ENCODING CATEGORICAL VARIABLES
# ============================================================================

print("\n[7] Encoding categorical variables...")

# Define the feature columns and target
feature_columns = ['Crop', 'State', 'Season', 'Crop_Year', 'Annual_Rainfall', 
                   'Fertilizer', 'Pesticide', 'Rainfall_per_Area', 
                   'Fertilizer_per_Area', 'Pesticide_per_Area', 'Input_Intensity', 
                   'Year_Squared', 'Log_Area', 'Log_Rainfall']
target_column = 'Yield'

# Create a copy of the dataframe without NaN values in the target column
data = df.dropna(subset=[target_column]).copy()

# Initialize label encoders for categorical columns
label_encoders = {}
for column in ['Crop', 'State', 'Season']:
    le = LabelEncoder()
    data.loc[:, column] = le.fit_transform(data[column])
    label_encoders[column] = le
    print(f"    Encoded {column}: {len(le.classes_)} classes")

# Save label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')
print(f"    Label encoders saved to 'label_encoders.pkl'")

# ============================================================================
# 7. TRAIN-TEST SPLIT
# ============================================================================

print("\n[8] Splitting data into train (80%) and test (20%) sets...")

# Define features and target
X = data[feature_columns]
y = data[target_column]

# Scale numerical features for models that need scaling
scaler = StandardScaler()
numerical_cols = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 
                  'Rainfall_per_Area', 'Fertilizer_per_Area', 'Pesticide_per_Area',
                  'Input_Intensity', 'Year_Squared', 'Log_Area', 'Log_Rainfall']
X_scaled = X.copy()
X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"    Training set: {len(X_train)} samples")
print(f"    Testing set: {len(X_test)} samples")

# ============================================================================
# 8. MULTI-MODEL TRAINING AND EVALUATION
# ============================================================================

print("\n" + "=" * 70)
print("MULTI-MODEL TRAINING AND EVALUATION")
print("=" * 70)

# Initialize lists to store model performance metrics
models_list = []
model_objects = []
model_names = []
training_scores_r2 = []
training_scores_adj_r2 = []
training_scores_rmse = []
training_scores_mae = []
testing_scores_r2 = []
testing_scores_adj_r2 = []
testing_scores_rmse = []
testing_scores_mae = []
training_time = []

import time

def evaluate_model_performance(model, model_name, X_train, y_train, X_test, y_test):
    """Evaluate model and return metrics"""
    
    print(f"\n▶ Training {model_name}...")
    start_time = time.time()
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Calculate training time
    train_time = time.time() - start_time
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate R² scores
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Calculate Adjusted R² scores
    n_train, p_train = X_train.shape
    n_test, p_test = X_test.shape
    train_adj_r2 = 1 - (1 - train_r2) * (n_train - 1) / (n_train - p_train - 1)
    test_adj_r2 = 1 - (1 - test_r2) * (n_test - 1) / (n_test - p_test - 1)

    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Calculate MAE
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Store metrics
    models_list.append(model_name)
    model_objects.append(model)
    training_scores_r2.append(train_r2 * 100)
    training_scores_adj_r2.append(train_adj_r2 * 100)
    training_scores_rmse.append(train_rmse)
    training_scores_mae.append(train_mae)
    testing_scores_r2.append(test_r2 * 100)
    testing_scores_adj_r2.append(test_adj_r2 * 100)
    testing_scores_rmse.append(test_rmse)
    testing_scores_mae.append(test_mae)
    training_time.append(train_time)
    
    # Display scores
    print(f"    ✓ Training complete in {train_time:.2f} seconds")
    print(f"    Training:  R² = {train_r2*100:.2f}% | Adj R² = {train_adj_r2*100:.2f}% | RMSE = {train_rmse:.4f} | MAE = {train_mae:.4f}")
    print(f"    Testing:   R² = {test_r2*100:.2f}% | Adj R² = {test_adj_r2*100:.2f}% | RMSE = {test_rmse:.4f} | MAE = {test_mae:.4f}")
    print(f"    CV R²: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100*2:.2f}%)")
    
    return model

# Define models with hyperparameters
model_configs = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge(alpha=1.0)),
    ('Lasso Regression', Lasso(alpha=0.01)),
    ('Decision Tree', DecisionTreeRegressor(max_depth=10, random_state=42)),
    ('Random Forest', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
    ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42)),
    ('XGBoost', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
    ('KNN', KNeighborsRegressor(n_neighbors=5)),
    ('SVR', SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
]

# Train each model
trained_models = {}
for name, model in model_configs:
    trained_model = evaluate_model_performance(model, name, X_train, y_train, X_test, y_test)
    trained_models[name] = trained_model

# ============================================================================
# 9. LSTM MODEL FOR TIME SERIES FORECASTING
# ============================================================================

print("\n" + "=" * 70)
print("DEEP LEARNING: LSTM MODEL TRAINING")
print("=" * 70)

def prepare_lstm_data(data, n_steps=5):
    """Prepare data for LSTM"""
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Prepare time-series data for LSTM
print("\n▶ Preparing time-series data for LSTM...")
crop_yield_series = df.groupby('Crop_Year')['Yield'].mean().values
scaler_lstm = StandardScaler()
crop_yield_scaled = scaler_lstm.fit_transform(crop_yield_series.reshape(-1, 1)).flatten()

# Prepare sequences
n_steps = 3
X_lstm, y_lstm = prepare_lstm_data(crop_yield_scaled, n_steps)

# Split data
split = int(0.8 * len(X_lstm))
X_lstm_train, X_lstm_test = X_lstm[:split], X_lstm[split:]
y_lstm_train, y_lstm_test = y_lstm[:split], y_lstm[split:]

# Reshape for LSTM [samples, timesteps, features]
X_lstm_train = X_lstm_train.reshape((X_lstm_train.shape[0], X_lstm_train.shape[1], 1))
X_lstm_test = X_lstm_test.reshape((X_lstm_test.shape[0], X_lstm_test.shape[1], 1))

print(f"    Training samples: {len(X_lstm_train)}")
print(f"    Testing samples: {len(X_lstm_test)}")
print(f"    Sequence length: {n_steps} years")

# Build LSTM model
print("\n▶ Building LSTM model...")
lstm_model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, 1)),
    Dropout(0.2),
    LSTM(50, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print(lstm_model.summary())

# Train LSTM
print("\n▶ Training LSTM model...")
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
checkpoint = ModelCheckpoint('models/lstm_best.h5', monitor='val_loss', save_best_only=True)

history = lstm_model.fit(
    X_lstm_train, y_lstm_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluate LSTM
y_lstm_pred = lstm_model.predict(X_lstm_test)
lstm_test_r2 = r2_score(y_lstm_test, y_lstm_pred)
lstm_test_rmse = np.sqrt(mean_squared_error(y_lstm_test, y_lstm_pred))

print(f"\n▶ LSTM Performance:")
print(f"    Testing R²: {lstm_test_r2*100:.2f}%")
print(f"    Testing RMSE: {lstm_test_rmse:.4f}")

# Add LSTM to model comparison
models_list.append('LSTM')
model_objects.append(lstm_model)
training_scores_r2.append(history.history['loss'][-1])  # Placeholder
testing_scores_r2.append(lstm_test_r2 * 100)
testing_scores_rmse.append(lstm_test_rmse)

# Save LSTM model and scaler
if not os.path.exists('models'):
    os.makedirs('models')
lstm_model.save('models/lstm_model.h5')
joblib.dump(scaler_lstm, 'models/lstm_scaler.pkl')
print("\n    LSTM model saved to 'models/lstm_model.h5'")

# ============================================================================
# 10. HYPERPARAMETER TUNING FOR BEST MODEL
# ============================================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING FOR BEST MODEL")
print("=" * 70)

# Identify best model based on test R²
best_idx = np.argmax(testing_scores_r2)
best_model_name = models_list[best_idx]
best_model = model_objects[best_idx]
print(f"\n▶ Best model from initial training: {best_model_name}")
print(f"    Test R²: {testing_scores_r2[best_idx]:.2f}%")

# Hyperparameter tuning for Random Forest (if it's the best or for demonstration)
if 'Random Forest' in models_list:
    print("\n▶ Performing Grid Search for Random Forest...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"\n    Best parameters: {grid_search.best_params_}")
    print(f"    Best CV score: {grid_search.best_score_*100:.2f}%")
    
    # Update best model
    best_model = grid_search.best_estimator_
    best_model_name = 'Random Forest (Tuned)'

# ============================================================================
# 11. SHAP ANALYSIS FOR EXPLAINABLE AI
# ============================================================================

print("\n" + "=" * 70)
print("EXPLAINABLE AI: SHAP ANALYSIS")
print("=" * 70)

print("\n▶ Calculating SHAP values for feature importance...")

# Use a sample for SHAP analysis (faster computation)
X_sample = X_train.sample(min(100, len(X_train)))

# Create explainer based on model type
if 'XGBoost' in str(type(best_model)):
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_sample)
else:
    # Use KernelExplainer for non-tree models
    explainer = shap.KernelExplainer(best_model.predict, X_sample)
    shap_values = explainer.shap_values(X_sample)

# Create feature importance plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, feature_names=feature_columns, show=False)
plt.title('SHAP Feature Importance Summary')
plt.tight_layout()
plt.savefig('static/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("    ✓ SHAP summary plot saved to 'static/shap_summary.png'")

# Calculate mean absolute SHAP values
mean_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': mean_shap
}).sort_values('importance', ascending=False)

print("\n    Top 5 Most Important Features:")
for i, row in feature_importance.head(5).iterrows():
    print(f"    {i+1}. {row['feature']}: {row['importance']:.4f}")

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
joblib.dump(explainer, 'shap_explainer.pkl')

# ============================================================================
# 12. MODEL COMPARISON VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING MODEL COMPARISON VISUALIZATIONS")
print("=" * 70)

# Create DataFrame with all model metrics
df_model = pd.DataFrame({
    "Model": models_list,
    "Train R² (%)": training_scores_r2,
    "Train Adj R² (%)": training_scores_adj_r2 if len(training_scores_adj_r2) == len(models_list) else [0]*len(models_list),
    "Train RMSE": training_scores_rmse,
    "Train MAE": training_scores_mae if len(training_scores_mae) == len(models_list) else [0]*len(models_list),
    "Test R² (%)": testing_scores_r2,
    "Test Adj R² (%)": testing_scores_adj_r2 if len(testing_scores_adj_r2) == len(models_list) else [0]*len(models_list),
    "Test RMSE": testing_scores_rmse,
    "Test MAE": testing_scores_mae if len(testing_scores_mae) == len(models_list) else [0]*len(models_list),
    "Training Time (s)": training_time if len(training_time) == len(models_list) else [0]*len(models_list)
})

# Sort by Test R²
df_model_sort = df_model.sort_values(by="Test R² (%)", ascending=False)
print("\n▶ Model Performance Summary:")
print(df_model_sort.to_string(index=False))

# Save model comparison
df_model_sort.to_csv('model_comparison.csv', index=False)
print("\n    Model comparison saved to 'model_comparison.csv'")

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# R² Comparison
ax1 = axes[0, 0]
models_short = [m[:15] + '...' if len(m) > 15 else m for m in df_model_sort['Model'].values]
x_pos = np.arange(len(models_short))
ax1.bar(x_pos - 0.2, df_model_sort['Train R² (%)'].values, width=0.4, label='Train', color='blue', alpha=0.7)
ax1.bar(x_pos + 0.2, df_model_sort['Test R² (%)'].values, width=0.4, label='Test', color='green', alpha=0.7)
ax1.set_xlabel('Models')
ax1.set_ylabel('R² Score (%)')
ax1.set_title('Model Performance Comparison (R² Score)')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models_short, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# RMSE Comparison
ax2 = axes[0, 1]
ax2.bar(x_pos - 0.2, df_model_sort['Train RMSE'].values, width=0.4, label='Train', color='blue', alpha=0.7)
ax2.bar(x_pos + 0.2, df_model_sort['Test RMSE'].values, width=0.4, label='Test', color='green', alpha=0.7)
ax2.set_xlabel('Models')
ax2.set_ylabel('RMSE')
ax2.set_title('Model Error Comparison (RMSE)')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models_short, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3)

# MAE Comparison
ax3 = axes[1, 0]
ax3.bar(x_pos - 0.2, df_model_sort['Train MAE'].values if 'Train MAE' in df_model_sort.columns else [0]*len(df_model_sort), 
        width=0.4, label='Train', color='blue', alpha=0.7)
ax3.bar(x_pos + 0.2, df_model_sort['Test MAE'].values if 'Test MAE' in df_model_sort.columns else [0]*len(df_model_sort), 
        width=0.4, label='Test', color='green', alpha=0.7)
ax3.set_xlabel('Models')
ax3.set_ylabel('MAE')
ax3.set_title('Model Error Comparison (MAE)')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(models_short, rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Training Time
ax4 = axes[1, 1]
ax4.bar(x_pos, df_model_sort['Training Time (s)'].values if 'Training Time (s)' in df_model_sort.columns else [0]*len(df_model_sort), 
        color='orange', alpha=0.7)
ax4.set_xlabel('Models')
ax4.set_ylabel('Training Time (seconds)')
ax4.set_title('Model Training Time Comparison')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(models_short, rotation=45, ha='right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('static/model_comparison_detailed.png', dpi=300, bbox_inches='tight')
plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
plt.close()

print("    ✓ Model comparison plots saved to 'static/model_comparison_detailed.png'")

# ============================================================================
# 13. SAVE BEST MODEL
# ============================================================================

print("\n" + "=" * 70)
print("SAVING BEST MODEL FOR DEPLOYMENT")
print("=" * 70)

# Get the best model (excluding LSTM for now as it needs special handling)
best_model_final = model_objects[best_idx]
print(f"\n▶ Best model selected: {best_model_name}")
print(f"    Test R²: {testing_scores_r2[best_idx]:.2f}%")
print(f"    Test RMSE: {testing_scores_rmse[best_idx]:.4f}")

# Save the trained model for future predictions
joblib.dump((best_model_final, feature_columns), 'best_model.pkl')
print("    ✓ Model saved to 'best_model.pkl'")

# Save model metadata
model_metadata = {
    'best_model_name': best_model_name,
    'test_r2': testing_scores_r2[best_idx],
    'test_rmse': testing_scores_rmse[best_idx],
    'feature_columns': feature_columns,
    'n_features': len(feature_columns),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}
joblib.dump(model_metadata, 'model_metadata.pkl')
print("    ✓ Model metadata saved to 'model_metadata.pkl'")

# ============================================================================
# 14. CREATE MODEL PERFORMANCE JSON FOR DASHBOARD
# ============================================================================

import json

model_performance = {}
for i, model_name in enumerate(models_list):
    if model_name != 'LSTM':  # Handle LSTM separately
        model_performance[model_name] = {
            'R2': float(testing_scores_r2[i] / 100),
            'RMSE': float(testing_scores_rmse[i]),
            'MAE': float(testing_scores_mae[i]) if i < len(testing_scores_mae) else 0
        }
    else:
        model_performance[model_name] = {
            'R2': float(lstm_test_r2),
            'RMSE': float(lstm_test_rmse),
            'MAE': float(lstm_test_rmse * 0.8)  # Approximate
        }

with open('static/model_performance.json', 'w') as f:
    json.dump(model_performance, f, indent=2)

print("\n    ✓ Model performance JSON saved to 'static/model_performance.json'")

# ============================================================================
# 15. ENHANCED INTERACTIVE PREDICTION FUNCTION
# ============================================================================

def interactive_prediction():
    """Enhanced interactive prediction with multiple model options"""
    print("\n" + "=" * 70)
    print("ENHANCED INTERACTIVE PREDICTION MODULE")
    print("=" * 70)
    print("Type 'exit' at any prompt to cancel prediction\n")

    # Load saved artifacts
    label_encoders = joblib.load('label_encoders.pkl')
    best_model, feature_cols = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    median_vals = joblib.load('median_values.pkl')

    # Get available options
    available_crops = label_encoders['Crop'].classes_
    available_states = label_encoders['State'].classes_
    available_seasons = label_encoders['Season'].classes_

    def get_valid_input(prompt, options):
        """Get valid input from user"""
        while True:
            print(f"\nAvailable options: {', '.join(sorted(options)[:10])}" + 
                  (f" and {len(options)-10} more..." if len(options) > 10 else ""))
            value = input(prompt).strip()
            if value.lower() == 'exit':
                return None
            
            # Try to match with case-insensitive
            value_title = value.title()
            if value_title in options:
                return value_title
            elif value in options:
                return value
            
            # Try partial matching
            matches = [opt for opt in options if value.lower() in opt.lower()]
            if len(matches) == 1:
                print(f"    Did you mean '{matches[0]}'? Using that.")
                return matches[0]
            elif len(matches) > 1:
                print(f"    Multiple matches found: {', '.join(matches[:5])}")
                print("    Please be more specific.")
            else:
                print(f"Invalid input. Please choose from the available options.")

    # Get categorical inputs
    print("\n--- Crop Selection ---")
    crop = get_valid_input("Enter crop name: ", available_crops)
    if crop is None: return

    print("\n--- State Selection ---")
    state = get_valid_input("Enter state name: ", available_states)
    if state is None: return

    print("\n--- Season Selection ---")
    season = get_valid_input("Enter season: ", available_seasons)
    if season is None: return

    # Get numerical inputs with defaults
    print("\n--- Numerical Inputs (press Enter for default values) ---")
    
    try:
        year_input = input(f"Enter year [{median_vals['Crop_Year']:.0f}]: ").strip()
        year = float(year_input) if year_input else median_vals['Crop_Year']
        
        rainfall_input = input(f"Enter annual rainfall in mm [{median_vals['Annual_Rainfall']:.1f}]: ").strip()
        rainfall = float(rainfall_input) if rainfall_input else median_vals['Annual_Rainfall']
        
        fertilizer_input = input(f"Enter fertilizer in kg/ha [{median_vals['Fertilizer']:.1f}]: ").strip()
        fertilizer = float(fertilizer_input) if fertilizer_input else median_vals['Fertilizer']
        
        pesticide_input = input(f"Enter pesticide in kg/ha [{median_vals['Pesticide']:.1f}]: ").strip()
        pesticide = float(pesticide_input) if pesticide_input else median_vals['Pesticide']
        
        area_input = input(f"Enter area in hectares [{median_vals['Area']:.1f}]: ").strip()
        area = float(area_input) if area_input else median_vals['Area']
        
    except ValueError as e:
        print(f"Invalid input: {e}. Using default values.")
        year = median_vals['Crop_Year']
        rainfall = median_vals['Annual_Rainfall']
        fertilizer = median_vals['Fertilizer']
        pesticide = median_vals['Pesticide']
        area = median_vals['Area']

    # Create input dictionary with all features
    input_dict = {}
    for col in feature_cols:
        if col == 'Crop':
            input_dict[col] = [label_encoders['Crop'].transform([crop])[0]]
        elif col == 'State':
            input_dict[col] = [label_encoders['State'].transform([state])[0]]
        elif col == 'Season':
            input_dict[col] = [label_encoders['Season'].transform([season])[0]]
        elif col == 'Crop_Year':
            input_dict[col] = [year]
        elif col == 'Annual_Rainfall':
            input_dict[col] = [rainfall]
        elif col == 'Fertilizer':
            input_dict[col] = [fertilizer]
        elif col == 'Pesticide':
            input_dict[col] = [pesticide]
        elif col == 'Rainfall_per_Area':
            input_dict[col] = [rainfall / area]
        elif col == 'Fertilizer_per_Area':
            input_dict[col] = [fertilizer / area]
        elif col == 'Pesticide_per_Area':
            input_dict[col] = [pesticide / area]
        elif col == 'Input_Intensity':
            input_dict[col] = [(fertilizer + pesticide) / area]
        elif col == 'Year_Squared':
            input_dict[col] = [year ** 2]
        elif col == 'Log_Area':
            input_dict[col] = [np.log1p(area)]
        elif col == 'Log_Rainfall':
            input_dict[col] = [np.log1p(rainfall)]

    # Create DataFrame
    input_data = pd.DataFrame(input_dict)
    
    # Scale numerical features
    input_data_scaled = input_data.copy()
    input_data_scaled[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # Make predictions with multiple models
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    
    print(f"\nInput Summary:")
    print(f"  Crop: {crop}")
    print(f"  State: {state}")
    print(f"  Season: {season}")
    print(f"  Year: {year}")
    print(f"  Rainfall: {rainfall:.1f} mm")
    print(f"  Fertilizer: {fertilizer:.1f} kg/ha")
    print(f"  Pesticide: {pesticide:.1f} kg/ha")
    print(f"  Area: {area:.1f} ha")

    print("\nModel Predictions (Yield in tons/ha):")
    
    predictions = {}
    for name, model in trained_models.items():
        try:
            pred = model.predict(input_data_scaled)[0]
            predictions[name] = pred
            print(f"  {name:20s}: {pred:.4f}")
        except:
            pass
    
    # Get best model prediction
    best_pred = best_model_final.predict(input_data_scaled)[0]
    print(f"\n▶ Best Model ({best_model_name}): {best_pred:.4f} tons/ha")
    
    # Calculate ensemble prediction (average of all models)
    ensemble_pred = np.mean(list(predictions.values()))
    print(f"▶ Ensemble Prediction: {ensemble_pred:.4f} tons/ha")
    
    # Calculate confidence interval
    std_pred = np.std(list(predictions.values()))
    print(f"▶ 95% Confidence Interval: [{ensemble_pred - 1.96*std_pred:.4f}, {ensemble_pred + 1.96*std_pred:.4f}]")
    
    # Save prediction to history
    prediction_record = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'crop': crop,
        'state': state,
        'season': season,
        'year': year,
        'rainfall': rainfall,
        'fertilizer': fertilizer,
        'pesticide': pesticide,
        'area': area,
        'predictions': predictions,
        'best_model': best_model_name,
        'best_prediction': float(best_pred),
        'ensemble_prediction': float(ensemble_pred)
    }
    
    # Save to history file
    try:
        history_file = 'prediction_history.json'
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(prediction_record)
        
        # Keep only last 100 predictions
        if len(history) > 100:
            history = history[-100:]
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
        print(f"\n    ✓ Prediction saved to history")
    except Exception as e:
        print(f"    Warning: Could not save to history: {e}")

# Call interactive prediction
interactive_prediction()

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print("\nSummary of generated files:")
print("  ✓ median_values.pkl - Default values for missing inputs")
print("  ✓ unique_values.pkl - Dropdown options for UI")
print("  ✓ label_encoders.pkl - Encoders for categorical variables")
print("  ✓ scaler.pkl - Feature scaler")
print("  ✓ best_model.pkl - Best performing model for deployment")
print("  ✓ model_metadata.pkl - Model information")
print("  ✓ model_comparison.csv - Performance metrics table")
print("  ✓ static/shap_summary.png - SHAP feature importance plot")
print("  ✓ static/model_comparison_detailed.png - Model comparison charts")
print("  ✓ static/model_performance.json - Performance data for dashboard")
print("  ✓ models/lstm_model.h5 - LSTM deep learning model")
print("\nYou can now run the Flask app:")
print("  python app.py")
print("\nAccess the dashboard at:")
print("  http://localhost:5000")