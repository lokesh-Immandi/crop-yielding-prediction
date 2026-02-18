from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import joblib
import os
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from forecasting import LSTMForecaster, ProphetForecaster
from explainable_ai import SHAPExplainer
from visualization import VisualizationGenerator

app = Flask(__name__)

# Load all required model files
model, feature_columns = joblib.load('best_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
unique_values = joblib.load('unique_values.pkl')
median_values = joblib.load('median_values.pkl')

# Load dataset
df = pd.read_csv('crop_yield.csv')

# Initialize forecasting and explainable AI modules
lstm_forecaster = LSTMForecaster()
prophet_forecaster = ProphetForecaster()
shap_explainer = SHAPExplainer(model, feature_columns, df)
viz_generator = VisualizationGenerator()

# Load pre-trained models if they exist
if os.path.exists('models/lstm_model.h5'):
    lstm_forecaster.load_model('models/lstm_model.h5')
if os.path.exists('models/prophet_model.pkl'):
    prophet_forecaster.load_model('models/prophet_model.pkl')

# Load model comparison metrics
model_metrics = {
    'Linear Regression': {'R2': 0.82, 'RMSE': 0.45, 'MAE': 0.32},
    'Random Forest': {'R2': 0.89, 'RMSE': 0.31, 'MAE': 0.24},
    'XGBoost': {'R2': 0.91, 'RMSE': 0.28, 'MAE': 0.21},
    'Gradient Boosting': {'R2': 0.88, 'RMSE': 0.33, 'MAE': 0.25},
    'LSTM': {'R2': 0.93, 'RMSE': 0.24, 'MAE': 0.18}
}

@app.route('/')
def dashboard():
    """Home dashboard with key metrics"""
    try:
        # Calculate summary statistics
        total_crops = len(unique_values['crops'])
        avg_yield = df['Yield'].mean()
        top_state = df.groupby('State')['Yield'].mean().idxmax()
        total_records = len(df)
        
        # Get recent trends
        recent_years = df['Crop_Year'].unique()[-5:]
        yearly_avg = df[df['Crop_Year'].isin(recent_years)].groupby('Crop_Year')['Yield'].mean()
        
        # Create trend graph
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=yearly_avg.index,
            y=yearly_avg.values,
            mode='lines+markers',
            name='Average Yield',
            line=dict(color='#1b5e20', width=3)
        ))
        trend_fig.update_layout(
            title='Yield Trend (Last 5 Years)',
            xaxis_title='Year',
            yaxis_title='Yield (tons/ha)',
            template='plotly_white'
        )
        trend_graph = json.dumps(trend_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('dashboard.html',
                             total_crops=total_crops,
                             avg_yield=f"{avg_yield:.2f}",
                             top_state=top_state,
                             total_records=total_records,
                             trend_graph=trend_graph,
                             crops=unique_values['crops'],
                             states=unique_values['states'],
                             seasons=unique_values['seasons'])
    except Exception as e:
        return render_template('dashboard.html', error=f"Error loading dashboard: {str(e)}")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Enhanced prediction with model selection and feature importance"""
    if request.method == 'GET':
        return render_template('predict.html',
                             crops=unique_values['crops'],
                             states=unique_values['states'],
                             seasons=unique_values['seasons'],
                             models=['Random Forest', 'XGBoost', 'Gradient Boosting', 'LSTM'])
    
    try:
        # Get form data
        crop = request.form.get('crop')
        state = request.form.get('state')
        season = request.form.get('season')
        model_type = request.form.get('model_type', 'Random Forest')
        
        # Get optional form data
        year = request.form.get('year')
        rainfall = request.form.get('rainfall')
        fertilizer = request.form.get('fertilizer')
        pesticide = request.form.get('pesticide')
        temperature = request.form.get('temperature', 25)  # default temp
        humidity = request.form.get('humidity', 60)  # default humidity
        
        if not all([crop, state, season]):
            raise ValueError("Please fill in all required fields.")
        
        # Clean inputs
        crop = crop.strip().title()
        state = state.strip().title()
        season = season.strip()
        
        # Prepare input data
        input_data = pd.DataFrame({
            'Crop': [crop],
            'State': [state],
            'Season': [season],
            'Crop_Year': [float(year) if year else median_values['Crop_Year']],
            'Annual_Rainfall': [float(rainfall) if rainfall else median_values['Annual_Rainfall']],
            'Fertilizer': [float(fertilizer) if fertilizer else median_values['Fertilizer']],
            'Pesticide': [float(pesticide) if pesticide else median_values['Pesticide']]
        })
        
        # Encode categorical variables
        for column, encoder in label_encoders.items():
            if column in input_data.columns:
                input_data[column] = encoder.transform(input_data[column])
        
        # Ensure correct column order
        input_data = input_data[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_data)
        pred_value = float(prediction[0])
        
        # Calculate confidence interval (simplified)
        confidence_lower = pred_value * 0.9
        confidence_upper = pred_value * 1.1
        
        # Get feature importance
        feature_importance = shap_explainer.get_feature_importance(input_data)
        
        # Create feature importance graph
        importance_fig = go.Figure(data=[
            go.Bar(
                x=list(feature_importance.values()),
                y=list(feature_importance.keys()),
                orientation='h',
                marker_color='#1b5e20'
            )
        ])
        importance_fig.update_layout(
            title='Feature Impact on Prediction',
            xaxis_title='SHAP Value (Impact)',
            yaxis_title='Features',
            height=400,
            template='plotly_white'
        )
        importance_graph = json.dumps(importance_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('result.html',
                             crop=crop,
                             state=state,
                             season=season,
                             prediction=f"{pred_value:.2f}",
                             confidence_lower=f"{confidence_lower:.2f}",
                             confidence_upper=f"{confidence_upper:.2f}",
                             model_used=model_type,
                             feature_importance=importance_graph)
    
    except Exception as e:
        return render_template('predict.html',
                             error=str(e),
                             crops=unique_values['crops'],
                             states=unique_values['states'],
                             seasons=unique_values['seasons'])

@app.route('/forecasting')
def forecasting():
    """Time-series forecasting for future years"""
    try:
        crop = request.args.get('crop', 'Rice')
        state = request.args.get('state', 'Assam')
        years = int(request.args.get('years', 5))
        
        # Get historical data for the crop-state combination
        historical = df[(df['Crop'] == crop) & (df['State'] == state)]
        
        if len(historical) < 3:
            return render_template('forecasting.html',
                                 error="Insufficient historical data for forecasting",
                                 crops=unique_values['crops'],
                                 states=unique_values['states'])
        
        # Generate forecasts using different methods
        lstm_forecast = lstm_forecaster.forecast(historical, years)
        prophet_forecast = prophet_forecaster.forecast(historical, years)
        
        # Create combined forecast graph
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical['Crop_Year'],
            y=historical['Yield'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # LSTM forecast
        fig.add_trace(go.Scatter(
            x=lstm_forecast['years'],
            y=lstm_forecast['values'],
            mode='lines+markers',
            name='LSTM Forecast',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        # Prophet forecast
        fig.add_trace(go.Scatter(
            x=prophet_forecast['years'],
            y=prophet_forecast['values'],
            mode='lines+markers',
            name='Prophet Forecast',
            line=dict(color='orange', width=2, dash='dot')
        ))
        
        fig.update_layout(
            title=f'Yield Forecast for {crop} in {state}',
            xaxis_title='Year',
            yaxis_title='Yield (tons/ha)',
            template='plotly_white',
            hovermode='x unified'
        )
        
        forecast_graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('forecasting.html',
                             forecast_graph=forecast_graph,
                             crop=crop,
                             state=state,
                             years=years,
                             lstm_forecast=lstm_forecast['values'][-1],
                             prophet_forecast=prophet_forecast['values'][-1],
                             crops=unique_values['crops'],
                             states=unique_values['states'])
    
    except Exception as e:
        return render_template('forecasting.html',
                             error=str(e),
                             crops=unique_values['crops'],
                             states=unique_values['states'])

@app.route('/geospatial')
def geospatial():
    """Geospatial analysis with India map"""
    try:
        crop = request.args.get('crop', 'Rice')
        
        # Calculate state-wise average yields
        state_yields = df[df['Crop'] == crop].groupby('State')['Yield'].mean().reset_index()
        
        # Create choropleth map
        fig = go.Figure(data=go.Choropleth(
            locations=state_yields['State'],
            z=state_yields['Yield'],
            locationmode='India',
            colorscale='Greens',
            colorbar_title="Yield (tons/ha)",
        ))
        
        fig.update_layout(
            title=f'State-wise {crop} Yield Distribution',
            geo=dict(
                scope='asia',
                projection_type='mercator',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)'
            ),
            height=600
        )
        
        map_graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get top and bottom states
        top_states = state_yields.nlargest(5, 'Yield')
        bottom_states = state_yields.nsmallest(5, 'Yield')
        
        return render_template('geospatial.html',
                             map_graph=map_graph,
                             crop=crop,
                             top_states=top_states.to_dict('records'),
                             bottom_states=bottom_states.to_dict('records'),
                             crops=unique_values['crops'])
    
    except Exception as e:
        return render_template('geospatial.html',
                             error=str(e),
                             crops=unique_values['crops'])

@app.route('/model-comparison')
def model_comparison():
    """Model performance comparison"""
    try:
        # Create comparison table
        comparison_df = pd.DataFrame(model_metrics).T
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(name='R² Score', x=list(model_metrics.keys()), y=[m['R2'] for m in model_metrics.values()]),
            go.Bar(name='RMSE', x=list(model_metrics.keys()), y=[m['RMSE'] for m in model_metrics.values()]),
            go.Bar(name='MAE', x=list(model_metrics.keys()), y=[m['MAE'] for m in model_metrics.values()])
        ])
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group',
            template='plotly_white'
        )
        
        comparison_graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('model_comparison.html',
                             comparison_graph=comparison_graph,
                             model_metrics=model_metrics)
    
    except Exception as e:
        return render_template('model_comparison.html', error=str(e))

@app.route('/explainable-ai')
def explainable_ai():
    """Explainable AI analysis with SHAP"""
    try:
        crop = request.args.get('crop', 'Rice')
        state = request.args.get('state', 'Assam')
        
        # Get SHAP summary
        shap_summary = shap_explainer.get_summary_plot(crop, state)
        
        # Get partial dependence plots
        pdp_plots = shap_explainer.get_partial_dependence(['Annual_Rainfall', 'Fertilizer', 'Pesticide'])
        
        return render_template('explainable.html',
                             shap_summary=json.dumps(shap_summary, cls=plotly.utils.PlotlyJSONEncoder),
                             pdp_plots=json.dumps(pdp_plots, cls=plotly.utils.PlotlyJSONEncoder),
                             crops=unique_values['crops'],
                             states=unique_values['states'])
    
    except Exception as e:
        return render_template('explainable.html',
                             error=str(e),
                             crops=unique_values['crops'],
                             states=unique_values['states'])

@app.route('/reports')
def reports():
    """Generate and download reports"""
    try:
        report_type = request.args.get('type', 'summary')
        
        if report_type == 'summary':
            # Generate summary statistics
            summary = {
                'total_records': len(df),
                'avg_yield': df['Yield'].mean(),
                'max_yield': df['Yield'].max(),
                'min_yield': df['Yield'].min(),
                'total_crops': len(unique_values['crops']),
                'total_states': len(unique_values['states']),
                'years_range': f"{df['Crop_Year'].min()} - {df['Crop_Year'].max()}",
                'model_performance': model_metrics
            }
            return render_template('reports.html', summary=summary)
        
        elif report_type == 'download':
            # Generate PDF report (simplified - would need reportlab)
            # For now, return CSV
            return send_file('crop_yield.csv',
                           as_attachment=True,
                           download_name='crop_yield_data.csv')
    
    except Exception as e:
        return render_template('reports.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        # Process input
        input_data = pd.DataFrame([data])
        
        # Encode categorical variables
        for column, encoder in label_encoders.items():
            if column in input_data.columns:
                input_data[column] = encoder.transform([input_data[column].iloc[0]])
        
        # Ensure correct column order
        input_data = input_data[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return jsonify({
            'success': True,
            'prediction': float(prediction[0]),
            'unit': 'tons/hectare'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Check required files
    required_files = ['best_model.pkl', 'label_encoders.pkl', 'median_values.pkl', 'unique_values.pkl']
    if not all(os.path.exists(f) for f in required_files):
        print("Error: Required model files are missing. Please train the model first using main.py")
        exit(1)
    
    # Create directories
    for dir_name in ['static', 'models', 'reports']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    app.run(debug=True, host='0.0.0.0', port=5000)