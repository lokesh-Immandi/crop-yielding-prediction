import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import plotly

class VisualizationGenerator:
    def __init__(self):
        pass
    
    def create_yield_trend(self, df, crop=None, state=None):
        """Create yield trend visualization"""
        if crop and state:
            data = df[(df['Crop'] == crop) & (df['State'] == state)]
            title = f'Yield Trend for {crop} in {state}'
        elif crop:
            data = df[df['Crop'] == crop]
            title = f'Yield Trend for {crop} (All States)'
        elif state:
            data = df[df['State'] == state]
            title = f'Yield Trend in {state} (All Crops)'
        else:
            data = df
            title = 'Overall Yield Trend'
        
        yearly_avg = data.groupby('Crop_Year')['Yield'].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly_avg['Crop_Year'],
            y=yearly_avg['Yield'],
            mode='lines+markers',
            name='Average Yield',
            line=dict(color='#1b5e20', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Year',
            yaxis_title='Yield (tons/ha)',
            template='plotly_white',
            hovermode='x'
        )
        
        return fig
    
    def create_state_comparison(self, df, crop=None):
        """Create state-wise comparison bar chart"""
        if crop:
            data = df[df['Crop'] == crop]
            title = f'State-wise Yield Comparison for {crop}'
        else:
            data = df
            title = 'State-wise Average Yield (All Crops)'
        
        state_avg = data.groupby('State')['Yield'].mean().sort_values(ascending=False).head(15)
        
        fig = go.Figure(data=[
            go.Bar(
                x=state_avg.values,
                y=state_avg.index,
                orientation='h',
                marker_color='#1b5e20',
                text=state_avg.values.round(2),
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Average Yield (tons/ha)',
            yaxis_title='State',
            height=500,
            template='plotly_white',
            margin=dict(l=150)
        )
        
        return fig
    
    def create_crop_comparison(self, df, state=None):
        """Create crop-wise comparison"""
        if state:
            data = df[df['State'] == state]
            title = f'Crop-wise Yield Comparison in {state}'
        else:
            data = df
            title = 'Top Crops by Yield'
        
        crop_avg = data.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(15)
        
        fig = go.Figure(data=[
            go.Bar(
                x=crop_avg.values,
                y=crop_avg.index,
                orientation='h',
                marker_color='#2e7d32',
                text=crop_avg.values.round(2),
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Average Yield (tons/ha)',
            yaxis_title='Crop',
            height=500,
            template='plotly_white',
            margin=dict(l=150)
        )
        
        return fig
    
    def create_season_analysis(self, df):
        """Create seasonal analysis visualization"""
        season_avg = df.groupby('Season')['Yield'].mean().reset_index()
        
        colors = ['#1b5e20', '#2e7d32', '#388e3c', '#43a047', '#4caf50']
        
        fig = go.Figure(data=[
            go.Pie(
                labels=season_avg['Season'],
                values=season_avg['Yield'],
                hole=0.4,
                marker=dict(colors=colors),
                textinfo='label+percent',
                textposition='inside'
            )
        ])
        
        fig.update_layout(
            title='Yield Distribution by Season',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_correlation_heatmap(self, df):
        """Create correlation heatmap"""
        numeric_cols = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Area', 'Production', 'Yield']
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Greens',
            zmin=-1, zmax=1,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def create_rainfall_impact(self, df):
        """Create rainfall vs yield scatter plot"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Annual_Rainfall'],
            y=df['Yield'],
            mode='markers',
            name='Data Points',
            marker=dict(
                color='#1b5e20',
                size=8,
                opacity=0.6
            )
        ))
        
        # Add trend line
        z = np.polyfit(df['Annual_Rainfall'], df['Yield'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['Annual_Rainfall'].min(), df['Annual_Rainfall'].max(), 100)
        
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Impact of Rainfall on Yield',
            xaxis_title='Annual Rainfall (mm)',
            yaxis_title='Yield (tons/ha)',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_fertilizer_impact(self, df):
        """Create fertilizer vs yield scatter plot"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Fertilizer'],
            y=df['Yield'],
            mode='markers',
            name='Data Points',
            marker=dict(
                color='#2e7d32',
                size=8,
                opacity=0.6
            )
        ))
        
        fig.update_layout(
            title='Impact of Fertilizer on Yield',
            xaxis_title='Fertilizer (kg/ha)',
            yaxis_title='Yield (tons/ha)',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_yield_distribution(self, df):
        """Create yield distribution histogram"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df['Yield'],
            nbinsx=50,
            marker_color='#1b5e20',
            opacity=0.7,
            name='Yield Distribution'
        ))
        
        fig.update_layout(
            title='Distribution of Crop Yields',
            xaxis_title='Yield (tons/ha)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_time_series_forecast(self, historical, forecast, title="Yield Forecast"):
        """Create time series forecast visualization"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical['Crop_Year'],
            y=historical['Yield'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecast['years'],
            y=forecast['values'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='green', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Year',
            yaxis_title='Yield (tons/ha)',
            template='plotly_white',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def save_plot(self, fig, filename):
        """Save plot to file"""
        fig.write_html(f"static/{filename}.html")
        fig.write_image(f"static/{filename}.png", width=1200, height=600)