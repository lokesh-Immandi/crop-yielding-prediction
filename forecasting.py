import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
import joblib
import os

class LSTMForecaster:
    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
    
    def prepare_data(self, historical_data):
        """Prepare data for LSTM training"""
        values = historical_data['Yield'].values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_values)):
            X.append(scaled_values[i-self.sequence_length:i, 0])
            y.append(scaled_values[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, historical_data, epochs=50):
        """Train LSTM model"""
        X, y = self.prepare_data(historical_data)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        self.model = self.build_model((X.shape[1], 1))
        self.model.fit(X, y, epochs=epochs, verbose=0, validation_split=0.1)
        
        return self
    
    def forecast(self, historical_data, years=5):
        """Generate forecast for future years"""
        if self.model is None:
            # If model not trained, use simple extrapolation
            return self.simple_forecast(historical_data, years)
        
        # Get last sequence
        last_values = historical_data['Yield'].values[-self.sequence_length:]
        last_scaled = self.scaler.transform(last_values.reshape(-1, 1))
        
        forecasts = []
        current_seq = last_scaled.flatten()
        
        for _ in range(years):
            # Reshape for prediction
            X_pred = current_seq.reshape((1, self.sequence_length, 1))
            
            # Predict next value
            next_pred = self.model.predict(X_pred, verbose=0)[0, 0]
            forecasts.append(next_pred)
            
            # Update sequence
            current_seq = np.append(current_seq[1:], next_pred)
        
        # Inverse transform
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = self.scaler.inverse_transform(forecasts).flatten()
        
        # Generate year labels
        last_year = historical_data['Crop_Year'].max()
        future_years = list(range(int(last_year) + 1, int(last_year) + years + 1))
        
        return {
            'years': future_years,
            'values': forecasts.tolist()
        }
    
    def simple_forecast(self, historical_data, years=5):
        """Simple forecasting using trend extrapolation"""
        years_data = historical_data.groupby('Crop_Year')['Yield'].mean()
        
        if len(years_data) >= 2:
            # Calculate linear trend
            x = np.arange(len(years_data))
            y = years_data.values
            coeffs = np.polyfit(x, y, 1)
            
            last_year_idx = len(years_data) - 1
            future_indices = range(last_year_idx + 1, last_year_idx + years + 1)
            forecasts = [coeffs[0] * i + coeffs[1] for i in future_indices]
        else:
            # Use mean if insufficient data
            forecasts = [years_data.mean()] * years
        
        last_year = years_data.index[-1]
        future_years = list(range(int(last_year) + 1, int(last_year) + years + 1))
        
        return {
            'years': future_years,
            'values': forecasts
        }
    
    def save_model(self, path):
        """Save LSTM model"""
        if self.model:
            self.model.save(path)
            joblib.dump(self.scaler, path.replace('.h5', '_scaler.pkl'))
    
    def load_model(self, path):
        """Load LSTM model"""
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        scaler_path = path.replace('.h5', '_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)


class ProphetForecaster:
    def __init__(self):
        self.model = None
    
    def train(self, historical_data):
        """Train Prophet model"""
        # Prepare data for Prophet
        df_prophet = historical_data[['Crop_Year', 'Yield']].copy()
        df_prophet.columns = ['ds', 'y']
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')
        
        # Train model
        self.model = Prophet(yearly_seasonality=True)
        self.model.fit(df_prophet)
        
        return self
    
    def forecast(self, historical_data, years=5):
        """Generate forecast"""
        if self.model is None:
            # If model not trained, train it
            self.train(historical_data)
        
        # Create future dataframe
        last_year = historical_data['Crop_Year'].max()
        future_years = list(range(int(last_year) + 1, int(last_year) + years + 1))
        future_dates = pd.to_datetime(future_years, format='%Y')
        
        future = pd.DataFrame({'ds': future_dates})
        
        # Make predictions
        forecast = self.model.predict(future)
        
        return {
            'years': future_years,
            'values': forecast['yhat'].tolist()
        }
    
    def save_model(self, path):
        """Save Prophet model"""
        if self.model:
            joblib.dump(self.model, path)
    
    def load_model(self, path):
        """Load Prophet model"""
        if os.path.exists(path):
            self.model = joblib.load(path)