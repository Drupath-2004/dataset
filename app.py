import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

def calculate_score(engine_temp, brake_thickness, tire_pressure):
    score = 0

    if 95 < engine_temp <= 100:
        score += 1
    elif engine_temp > 100:
        score += 2

    if 3 < brake_thickness <= 5:
        score += 1
    elif brake_thickness <= 3:
        score += 2

    if 28 <= tire_pressure < 30 or 35 < tire_pressure <= 36:
        score += 1
    elif tire_pressure < 28 or tire_pressure > 36:
        score += 2

    return score

def determine_maintenance_type(score):
    if score <= 2:
        return "Routine Maintenance"
    elif score <= 4:
        return "Repair"
    else:
        return "Component Replacement"

def calculate_anomaly_probability(row):
    score = 0
    if row['Engine Temperature (°C)'] > 100:
        score += 0.4
    if row['Brake Pad Thickness (mm)'] < 3:
        score += 0.4
    elif row['Brake Pad Thickness (mm)'] < 5:
        score += 0.2
    if row['Tire Pressure (PSI)'] < 30 or row['Tire Pressure (PSI)'] > 35:
        score += 0.2
    return score

def determine_anomaly_status(row, model_prediction, anomaly_score):
    is_anomaly = False
    if (row['Engine Temperature (°C)'] > 100 or
        row['Brake Pad Thickness (mm)'] < 3 or
        row['Tire Pressure (PSI)'] < 28 or
        row['Tire Pressure (PSI)'] > 36):
        is_anomaly = True
    if anomaly_score > 0.3:
        is_anomaly = True
    return is_anomaly

def calculate_rul(row, model_prediction):
    base_rul = float(model_prediction)
    maintenance_type_multipliers = {
        "Routine Maintenance": 1.0,
        "Repair": 0.8,
        "Component Replacement": 0.9
    }
    base_rul *= maintenance_type_multipliers.get(row['Maintenance Type'], 1.0)
    return max(0, min(365, base_rul))

def calculate_health_score(data):
    score = 100
    if data['Engine Temperature (°C)'].values[0] > 100:
        score -= 25
    if data['Brake Pad Thickness (mm)'].values[0] < 3:
        score -= 30
    elif data['Brake Pad Thickness (mm)'].values[0] < 5:
        score -= 15
    if (data['Tire Pressure (PSI)'].values[0] < 30) or (data['Tire Pressure (PSI)'].values[0] > 35):
        score -= 20
    return max(0, score)

try:
    lstm_model_rul = tf.keras.models.load_model("lstm_model_rul.h5")
    random_forest_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    X_test = joblib.load("X_test.pkl")
    y_test_rul = joblib.load("y_test_rul.pkl")
    y_test_anomaly = joblib.load("y_test_anomaly.pkl")
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
except Exception as e:
    st.error("Error loading models or data: " + str(e))
    st.stop()

features = {
    'Engine Temperature (°C)': {'min': 70, 'max': 120, 'critical': 100, 'optimal': 90},
    'Brake Pad Thickness (mm)': {'min': 1, 'max': 10, 'critical': 3, 'optimal': 7},
    'Tire Pressure (PSI)': {'min': 20, 'max': 50, 'critical': 35, 'optimal': 32}
}

feature_descriptions = {
    "Engine_Temperature": "Critical above 100°C. Optimal range: 85-95°C.",
    "Brake_Pad_Thickness": "Critical below 3mm. Replacement recommended below 5mm.",
    "Tire_Pressure": "Optimal range: 28-35 PSI. Check if outside this range."
}

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.markdown("""
    <style>
    .centered {
        text-align: center;
        text-decoration: underline;
    }
    .selectbox-container select {
        width: auto;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("<h1 class='centered'>Predictive Maintenance Analytics Dashboard</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    engine_temp = st.slider("Engine Temperature (°C)", 70.0, 120.0, 90.0, step=0.1)
    if engine_temp > 100:
        st.markdown("<p class='critical'>Critical: Temperature too high!</p>", unsafe_allow_html=True)
        st.markdown("<p class='critical'>Recommendation: Check cooling system immediately.</p>", unsafe_allow_html=True)
    elif engine_temp > 95:
        st.markdown("<p class='warning'>Warning: Temperature elevated</p>", unsafe_allow_html=True)
        st.markdown("<p class='warning'>Recommendation: Monitor and reduce load if possible.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='good'>Normal temperature range</p>", unsafe_allow_html=True)
        st.markdown("<p class='good'>Recommendation: No action required.</p>", unsafe_allow_html=True)

with col2:
    brake_thickness = st.slider("Brake Pad Thickness (mm)", 1.0, 10.0, 7.0, step=0.1)
    if brake_thickness < 3:
        st.markdown("<p class='critical'>Critical: Replace brake pads immediately!</p>", unsafe_allow_html=True)
        st.markdown("<p class='critical'>Recommendation: Schedule urgent brake replacement.</p>", unsafe_allow_html=True)
    elif brake_thickness < 5:
        st.markdown("<p class='warning'>Warning: Plan for replacement soon</p>", unsafe_allow_html=True)
        st.markdown("<p class='warning'>Recommendation: Inspect and replace within 30 days.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='good'>Brake pads in good condition</p>", unsafe_allow_html=True)
        st.markdown("<p class='good'>Recommendation: No action required.</p>", unsafe_allow_html=True)

with col3:
    tire_pressure = st.slider("Tire Pressure (PSI)", 20.0, 50.0, 32.0, step=0.1)
    if tire_pressure < 28 or tire_pressure > 35:
        st.markdown("<p class='warning'>Warning: Pressure needs adjustment</p>", unsafe_allow_html=True)
        st.markdown("<p class='warning'>Recommendation: Adjust pressure to 30-35 PSI.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='good'>Optimal pressure range</p>", unsafe_allow_html=True)
        st.markdown("<p class='good'>Recommendation: No action required.</p>", unsafe_allow_html=True)

# Create a user input DataFrame with the necessary features
user_input = pd.DataFrame({
    "Engine Temperature (°C)": [engine_temp],
    "Brake Pad Thickness (mm)": [brake_thickness],
    "Tire Pressure (PSI)": [tire_pressure]
})

# Scale the input data
input_data_scaled = scaler.transform(user_input)
input_data_lstm = input_data_scaled.reshape(1, 1, input_data_scaled.shape[1])

# Calculate the score
score = calculate_score(engine_temp, brake_thickness, tire_pressure)

# Determine the maintenance type
maintenance_type = determine_maintenance_type(score)

# Add Maintenance Type to user_input for further calculations
user_input["Maintenance Type"] = maintenance_type

# Predict RUL and other metrics using the scaled input data
rul_prediction = lstm_model_rul.predict(input_data_lstm)[0][0]
anomaly_prediction = random_forest_model.predict(input_data_scaled)[0]
anomaly_score = calculate_anomaly_probability(user_input.iloc[0])
health_score = calculate_health_score(user_input)

rul_days = calculate_rul(user_input.iloc[0], rul_prediction)
is_anomaly = determine_anomaly_status(user_input.iloc[0], anomaly_prediction, anomaly_score)

def map_status_to_color(status):
    return {
        'good': 'normal',
        'warning': 'inverse',
        'critical': 'inverse'
    }.get(status, 'off')

st.markdown("<h2 class='centered'>System Health Analysis</h2>", unsafe_allow_html=True)

rul_days_color = map_status_to_color('good' if rul_days > 90 else 'warning' if rul_days > 30 else 'critical')
anomaly_status = "Anomaly Detected" if is_anomaly else "Normal"
anomaly_status_color = map_status_to_color('critical' if is_anomaly else 'good')
health_score_color = map_status_to_color('good' if health_score > 75 else 'warning' if health_score > 50 else 'critical')

st.markdown(f"""
<style>
    .good {{
        color: green;
    }}
    .warning {{
        color: orange;
    }}
    .critical {{
        color: red;
    }}
</style>
""", unsafe_allow_html=True)

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.subheader("Remaining Useful Life")
    st.metric("RUL (days)", f"{rul_days:.2f}", delta_color=rul_days_color)

with metric_col2:
    st.subheader("Anomaly Status")
    st.metric("Status", anomaly_status, delta_color=anomaly_status_color)

with metric_col3:
    st.subheader("Health Score")
    st.metric("Score", f"{health_score}/100", delta_color=health_score_color)

st.markdown("<h2 class='centered'>Maintenance Predictions</h2>", unsafe_allow_html=True)
st.markdown(f"<p class='good'>Predicted Maintenance Type: {maintenance_type}</p>", unsafe_allow_html=True)

st.markdown("<h2 class='centered'>Maintenance Recommendations</h2>", unsafe_allow_html=True)
recommendations = []

if engine_temp > 100:
    recommendations.append("•  Check cooling system immediately.")
elif engine_temp > 95:
    recommendations.append("•  Monitor engine temperature closely and reduce load if possible.")
if brake_thickness < 3:
    recommendations.append("•  Replace brake pads urgently.")
elif brake_thickness < 5:
    recommendations.append("•  Schedule brake pad replacement soon.")
if tire_pressure < 30 or tire_pressure > 35:
    recommendations.append("•  Adjust tire pressure to optimal range.")
elif tire_pressure < 28 or tire_pressure > 36:
    recommendations.append("•  Immediate tire pressure adjustment required.")
if rul_days < 45:
    recommendations.append("•  Schedule comprehensive maintenance within 30 days.")
elif rul_days < 90:
    recommendations.append("•  Plan for maintenance within 90 days.")
if anomaly_score > 0.7:
    recommendations.append("•  Schedule comprehensive inspection due to high anomaly probability.")
if anomaly_score > 0.5:
    recommendations.append("•  Investigate potential issues based on anomaly indicators.")
if health_score < 50:
    recommendations.append("•  Perform an overall system check.")
if health_score < 75:
    recommendations.append("•  Schedule maintenance to improve system health.")
if recommendations:
    for rec in recommendations:
        st.markdown(f"<p class='warning'>{rec}</p>", unsafe_allow_html=True)
else:
    st.markdown("<p class='good'>All systems operating within normal parameters</p>", unsafe_allow_html=True)

rul_test_predictions = lstm_model_rul.predict(X_test_lstm)
