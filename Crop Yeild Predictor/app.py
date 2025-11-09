import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('DataSet/crop_production.csv')
    df = df.dropna()
    return df

def preprocess_data(df):
    le_state = LabelEncoder()
    le_district = LabelEncoder()
    le_season = LabelEncoder()
    le_crop = LabelEncoder()
    
    df_encoded = df.copy()
    # Calculate yield per acre (1 hectare = 2.47105 acres)
    df_encoded['Yield_Per_Acre'] = df_encoded['Production'] / (df_encoded['Area'] * 2.47105)
    # Remove rows with invalid yields
    df_encoded = df_encoded[(df_encoded['Yield_Per_Acre'] > 0) & (df_encoded['Yield_Per_Acre'] < 20)]
    
    df_encoded['State_Name'] = le_state.fit_transform(df_encoded['State_Name'])
    df_encoded['District_Name'] = le_district.fit_transform(df_encoded['District_Name'])
    df_encoded['Season'] = le_season.fit_transform(df_encoded['Season'])
    df_encoded['Crop'] = le_crop.fit_transform(df_encoded['Crop'])
    
    return df_encoded, le_state, le_district, le_season, le_crop

def train_model(df_encoded):
    X = df_encoded[['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop']]
    y = df_encoded['Yield_Per_Acre']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

# Streamlit App
st.title("🌾 Crop Yield Prediction")
st.write("Predict crop production using Linear Regression")

# Load data
df = load_data()
df_encoded, le_state, le_district, le_season, le_crop = preprocess_data(df)

# Train model
model, mse, r2 = train_model(df_encoded)

# Display model performance
st.subheader("Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("R² Score", f"{r2:.3f}")
with col2:
    st.metric("MSE", f"{mse:.2e}")

# Prediction interface
st.subheader("Make Prediction")

col1, col2 = st.columns(2)
with col1:
    state = st.selectbox("State", df['State_Name'].unique())
    district = st.selectbox("District", df[df['State_Name']==state]['District_Name'].unique())
    crop_year = st.number_input("Crop Year", min_value=1997, max_value=2025, value=2020)

with col2:
    season = st.selectbox("Season", df['Season'].unique())
    crop = st.selectbox("Crop", df['Crop'].unique())
    area = st.number_input("Area (acres)", min_value=0.1, value=1.0)

if st.button("Predict Yield"):
    # Encode inputs
    state_encoded = le_state.transform([state])[0]
    district_encoded = le_district.transform([district])[0]
    season_encoded = le_season.transform([season])[0]
    crop_encoded = le_crop.transform([crop])[0]
    
    # Make prediction
    input_data = np.array([[state_encoded, district_encoded, crop_year, season_encoded, crop_encoded]])
    yield_per_acre = model.predict(input_data)[0]
    
    st.success(f"Predicted Yield: {yield_per_acre:.2f} tonnes/acre")
    
    # Calculate total production for given area
    total_production = yield_per_acre * area
    st.info(f"Total Production for {area} acres: {total_production:.2f} tonnes")

