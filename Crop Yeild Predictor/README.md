# Crop Yield Prediction

A simple web application that predicts crop production using Linear Regression.

## Features
- Linear regression model for crop yield prediction
- Interactive Streamlit interface
- Real-time predictions based on user inputs
- Model performance metrics display

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Usage
1. Select state, district, crop year, season, and crop type
2. Enter the area in hectares
3. Click "Predict Production" to get the estimated yield
4. View yield per hectare calculation

## Dataset
Uses crop production data with features:
- State Name
- District Name  
- Crop Year
- Season
- Crop Type
- Area (hectares)
- Production (target variable)