from flask import Flask, render_template, request, redirect
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objs as go
import plotly.io as pio
import io
import base64

app = Flask(__name__)

# Train models
def train_models():
    data = pd.DataFrame({
        'Open': [100.0, 102.0, 106.0, 107.0, 111.0],
        'High': [105.0, 108.0, 110.0, 112.0, 115.0],
        'Low': [95.0, 101.0, 104.0, 106.0, 110.0],
        'Close': [102.0, 106.0, 107.0, 111.0, 113.0],
        'Volume': [1000000, 1200000, 1100000, 1150000, 1300000]
    })
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor()
    }
    
    for name, model in models.items():
        model.fit(X, y)
    
    return models

models = train_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect('/')
    
    if file:
        if file.filename.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        elif file.filename.endswith('.xls'):
            df = pd.read_excel(file, engine='xlrd')
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            return "Unsupported file format"
        features = df[['Open', 'High', 'Low', 'Volume']]
        predictions = {}
        for name, model in models.items():
            preds = model.predict(features)
            predictions[name] = preds
        fig = go.Figure()

        for name, preds in predictions.items():
            fig.add_trace(go.Scatter(
                x=df.index,
                y=preds,
                mode='lines+markers',
                name=name
            ))

        fig.update_layout(
            title='Model Predictions',
            xaxis_title='Index',
            yaxis_title='Predicted Close Price',
            template='plotly_dark'  
        )
        graph_html = pio.to_html(fig, full_html=False)

        return render_template('result.html', plot=graph_html)

if __name__ == '__main__':
    app.run(debug=True)
