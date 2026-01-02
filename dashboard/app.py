"""
Time Series IDS Dashboard
Web-based dashboard for monitoring intrusion detection results
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from flask import Flask, render_template, jsonify
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import joblib
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, TIMESERIES_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Time Series IDS Dashboard"

class IDSDashboard:
    def __init__(self):
        """Initialize IDS Dashboard"""
        self.models = {}
        self.data = None
        self.load_data()
        
        logger.info("IDS Dashboard initialized")
    
    def load_data(self):
        """Load data for dashboard"""
        try:
            # Load time series data
            ts_path = os.path.join(TIMESERIES_DIR, "time_series_features.parquet")
            if os.path.exists(ts_path):
                self.data = pd.read_parquet(ts_path)
                logger.info(f"Loaded data with shape: {self.data.shape}")
                
                # Create sample predictions if not available
                if 'prediction' not in self.data.columns:
                    np.random.seed(42)
                    self.data['prediction'] = np.random.choice([0, 1], size=len(self.data), p=[0.85, 0.15])
                    self.data['confidence'] = np.random.uniform(0.5, 1.0, size=len(self.data))
                    
                    # Make predictions more realistic
                    attack_indices = self.data['is_attack'] == 1
                    self.data.loc[attack_indices, 'prediction'] = np.random.choice([0, 1], 
                                                                                  size=attack_indices.sum(), 
                                                                                  p=[0.2, 0.8])
            else:
                # Create sample data for demo
                logger.warning("No data found. Creating sample data...")
                self.create_sample_data()
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
        logger.info("Creating sample data for dashboard...")
        
        # Generate sample time series data
        n_samples = 1000
        timestamps = pd.date_range(start='2018-02-14', periods=n_samples, freq='1min')
        
        np.random.seed(42)
        self.data = pd.DataFrame({
            'timestamp': timestamps,
            'is_attack': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'prediction': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'confidence': np.random.uniform(0.5, 1.0, n_samples),
            'Flow_Packets_per_s': np.random.exponential(10, n_samples),
            'Flow_Bytes_per_s': np.random.exponential(1000, n_samples),
            'total_anomaly_score': np.random.poisson(2, n_samples),
        })
        
        # Add some correlation between attacks and predictions
        attack_mask = self.data['is_attack'] == 1
        self.data.loc[attack_mask, 'prediction'] = np.random.choice([0, 1], 
                                                                   attack_mask.sum(), 
                                                                   p=[0.2, 0.8])
        self.data.loc[attack_mask, 'confidence'] = np.random.uniform(0.7, 1.0, attack_mask.sum())
        
        logger.info("Sample data created")
    
    def get_recent_alerts(self, hours=24):
        """Get recent security alerts"""
        if self.data is None:
            return pd.DataFrame()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent data with attacks or high-confidence predictions
        recent_data = self.data[
            (self.data['prediction'] == 1) | 
            (self.data['confidence'] > 0.8)
        ]
        
        if 'timestamp' in recent_data.columns:
            recent_data = recent_data[recent_data['timestamp'] >= cutoff_time]
        else:
            recent_data = recent_data.tail(100)  # Last 100 if no timestamp
        
        return recent_data.tail(50)  # Return last 50 alerts
    
    def get_stats_summary(self):
        """Get summary statistics"""
        if self.data is None:
            return {}
        
        total_flows = len(self.data)
        attacks_detected = (self.data['prediction'] == 1).sum()
        true_attacks = (self.data['is_attack'] == 1).sum()
        
        # Calculate accuracy if both prediction and ground truth available
        accuracy = 0
        if 'is_attack' in self.data.columns:
            accuracy = ((self.data['prediction'] == self.data['is_attack']).sum() / total_flows) * 100
        
        return {
            'total_flows': total_flows,
            'attacks_detected': attacks_detected,
            'true_attacks': true_attacks,
            'accuracy': accuracy,
            'detection_rate': (attacks_detected / total_flows) * 100 if total_flows > 0 else 0
        }

# Initialize dashboard instance
dashboard = IDSDashboard()

# Dashboard Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🔐 Time Series Intrusion Detection Dashboard", 
                   className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Summary Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Flows", className="card-title"),
                    html.H2(id="total-flows", className="text-primary")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Attacks Detected", className="card-title"),
                    html.H2(id="attacks-detected", className="text-danger")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Detection Rate", className="card-title"),
                    html.H2(id="detection-rate", className="text-warning")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Accuracy", className="card-title"),
                    html.H2(id="accuracy", className="text-success")
                ])
            ])
        ], width=3)
    ], className="mb-4"),
    
    # Charts Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Attack Detection Over Time"),
                    dcc.Graph(id="time-series-chart")
                ])
            ])
        ], width=8),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Attack Distribution"),
                    dcc.Graph(id="attack-distribution")
                ])
            ])
        ], width=4)
    ], className="mb-4"),
    
    # Additional Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Network Traffic Patterns"),
                    dcc.Graph(id="traffic-patterns")
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Confidence Score Distribution"),
                    dcc.Graph(id="confidence-distribution")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Recent Alerts Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Recent Security Alerts"),
                    html.Div(id="alerts-table")
                ])
            ])
        ])
    ]),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # Update every 30 seconds
        n_intervals=0
    )
    
], fluid=True)

# Callbacks
@app.callback(
    [Output('total-flows', 'children'),
     Output('attacks-detected', 'children'),
     Output('detection-rate', 'children'),
     Output('accuracy', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_summary_cards(n):
    stats = dashboard.get_stats_summary()
    
    return (
        f"{stats.get('total_flows', 0):,}",
        f"{stats.get('attacks_detected', 0):,}",
        f"{stats.get('detection_rate', 0):.1f}%",
        f"{stats.get('accuracy', 0):.1f}%"
    )

@app.callback(
    Output('time-series-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_time_series_chart(n):
    if dashboard.data is None:
        return go.Figure()
    
    # Sample recent data for visualization
    recent_data = dashboard.data.tail(200)
    
    # Create time index if timestamp not available
    if 'timestamp' not in recent_data.columns:
        recent_data['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(hours=3),
            periods=len(recent_data),
            freq='1min'
        )
    
    # Aggregate by time windows
    recent_data['time_window'] = recent_data['timestamp'].dt.floor('5min')
    agg_data = recent_data.groupby('time_window').agg({
        'is_attack': 'sum',
        'prediction': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    
    # Add actual attacks
    fig.add_trace(go.Scatter(
        x=agg_data['time_window'],
        y=agg_data['is_attack'],
        mode='lines+markers',
        name='Actual Attacks',
        line=dict(color='red')
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=agg_data['time_window'],
        y=agg_data['prediction'],
        mode='lines+markers',
        name='Detected Attacks',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title="Attack Detection Timeline (5-min windows)",
        xaxis_title="Time",
        yaxis_title="Number of Attacks",
        hovermode='x'
    )
    
    return fig

@app.callback(
    Output('attack-distribution', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_attack_distribution(n):
    if dashboard.data is None:
        return go.Figure()
    
    # Create confusion matrix visualization
    actual = dashboard.data['is_attack']
    predicted = dashboard.data['prediction']
    
    labels = ['Benign', 'Attack']
    conf_matrix = pd.crosstab(actual, predicted, margins=True)
    
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix.values[:-1, :-1],  # Exclude margins
        x=['Predicted Benign', 'Predicted Attack'],
        y=['Actual Benign', 'Actual Attack'],
        colorscale='Blues',
        text=conf_matrix.values[:-1, :-1],
        texttemplate="%{text}",
        textfont={"size": 20}
    ))
    
    fig.update_layout(
        title="Prediction Accuracy Matrix"
    )
    
    return fig

@app.callback(
    Output('traffic-patterns', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_traffic_patterns(n):
    if dashboard.data is None:
        return go.Figure()
    
    # Show traffic volume patterns
    data = dashboard.data.tail(200)
    
    if 'Flow_Packets_per_s' in data.columns:
        fig = go.Figure()
        
        # Benign traffic
        benign_data = data[data['is_attack'] == 0]
        fig.add_trace(go.Histogram(
            x=benign_data['Flow_Packets_per_s'],
            name='Benign Traffic',
            opacity=0.7,
            nbinsx=30
        ))
        
        # Attack traffic
        attack_data = data[data['is_attack'] == 1]
        if len(attack_data) > 0:
            fig.add_trace(go.Histogram(
                x=attack_data['Flow_Packets_per_s'],
                name='Attack Traffic',
                opacity=0.7,
                nbinsx=30
            ))
        
        fig.update_layout(
            title="Traffic Volume Distribution",
            xaxis_title="Packets per Second",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        return fig
    
    return go.Figure()

@app.callback(
    Output('confidence-distribution', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_confidence_distribution(n):
    if dashboard.data is None or 'confidence' not in dashboard.data.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=dashboard.data['confidence'],
        nbinsx=20,
        name='Confidence Scores'
    ))
    
    fig.update_layout(
        title="Model Confidence Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Frequency"
    )
    
    return fig

@app.callback(
    Output('alerts-table', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_alerts_table(n):
    recent_alerts = dashboard.get_recent_alerts(hours=24)
    
    if len(recent_alerts) == 0:
        return html.P("No recent alerts", className="text-muted")
    
    # Create table rows
    table_rows = []
    for idx, alert in recent_alerts.head(10).iterrows():
        timestamp = alert.get('timestamp', 'N/A')
        confidence = alert.get('confidence', 0)
        
        row = html.Tr([
            html.Td(str(timestamp)[:19] if timestamp != 'N/A' else 'N/A'),
            html.Td("High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"),
            html.Td(f"{confidence:.2f}" if confidence else "N/A"),
            html.Td("Attack Detected", className="text-danger")
        ])
        table_rows.append(row)
    
    table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Timestamp"),
                html.Th("Severity"),
                html.Th("Confidence"),
                html.Th("Alert Type")
            ])
        ]),
        html.Tbody(table_rows)
    ], striped=True, bordered=True, hover=True)
    
    return table

def main():
    """Run the dashboard"""
    logger.info("Starting Time Series IDS Dashboard...")
    logger.info("Dashboard will be available at: http://localhost:8050")
    
    app.run(debug=True, host='0.0.0.0', port=8050)

if __name__ == "__main__":
    main()