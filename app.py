import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import r2_score, accuracy_score

# Function to calculate anomaly probability based on domain rules
def calculate_anomaly_probability(row):
    """Calculate probability of anomaly based on domain-specific rules."""
    score = 0
    # Engine Temperature rules
    if row['Engine Temperature (°C)'] > 100:
        score += 0.4

    # Brake Pad Thickness rules
    if row['Brake Pad Thickness (mm)'] < 3:
        score += 0.4
    elif row['Brake Pad Thickness (mm)'] < 5:
        score += 0.2

    # Tire Pressure rules
    if row['Tire Pressure (PSI)'] < 30 or row['Tire Pressure (PSI)'] > 35:
        score += 0.2

    return score

# Function to determine anomaly status based on predictions and rules
def determine_anomaly_status(row, model_prediction, anomaly_score):
    """Determine final anomaly status using model prediction and domain rules."""
    is_anomaly = False

    # Check domain rule thresholds
    if (row['Engine Temperature (°C)'] > 100 or
        row['Brake Pad Thickness (mm)'] < 3 or
        row['Tire Pressure (PSI)'] < 28 or
        row['Tire Pressure (PSI)'] > 36):
        is_anomaly = True

    # Check anomaly score threshold
    if anomaly_score > 0.3:
        is_anomaly = True

    return is_anomaly

# Function to calculate Remaining Useful Life (RUL)
def calculate_rul(row, model_prediction):
    """Adjust RUL prediction based on maintenance type."""
    base_rul = float(model_prediction)

    maintenance_type_multipliers = {
        0: 1.0,  # Routine Maintenance
        1: 0.8,  # Repair (reduce RUL by 20%)
        2: 0.9   # Component Replacement (reduce RUL by 10%)
    }

    base_rul *= maintenance_type_multipliers.get(row['Maintenance Type Label'], 1.0)
    return max(0, min(365, base_rul))  # Ensure RUL is between 0 and 365 days

# Function to calculate health score
def calculate_health_score(data):
    """Calculate a health score based on system parameters."""
    score = 100
    if data['Engine Temperature (°C)'].values[0] > 100:
        score -= 25
    if data['Brake Pad Thickness (mm)'].values[0] < 3:
        score -= 30
    elif data['Brake Pad Thickness (mm)'].values[0] < 5:
        score -= 15
    if data['Tire Pressure (PSI)'].values[0] < 30 or data['Tire Pressure (PSI)'].values[0] > 35:
        score -= 20
    return max(0, score)

# Load models and scaler
try:
    lstm_model = tf.keras.models.load_model("lstm_model.h5")
    random_forest_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    X_test = joblib.load("X_test.pkl")
    y_test_rul = joblib.load("y_test_rul.pkl")
    y_test_anomaly = joblib.load("y_test_anomaly.pkl")
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
except Exception as e:
    st.error("Error loading models or data: " + str(e))
    st.stop()

# Feature configurations
features = {
    'Engine Temperature (°C)': {'min': 70, 'max': 150, 'critical': 100, 'optimal': 90},
    'Brake Pad Thickness (mm)': {'min': 1, 'max': 10, 'critical': 3, 'optimal': 7},
    'Tire Pressure (PSI)': {'min': 20, 'max': 50, 'critical': 35, 'optimal': 32},
    'Maintenance Type Label': {}
}

feature_descriptions = {
    "Engine_Temperature": "Critical above 100°C. Optimal range: 85-95°C.",
    "Brake_Pad_Thickness": "Critical below 3mm. Replacement recommended below 5mm.",
    "Tire_Pressure": "Optimal range: 28-35 PSI. Check if outside this range.",
    "Maintenance_Type": "Impacts RUL: Repair (-20%), Component Replacement (-10%)"
}

# Streamlit UI
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.sidebar.title("System Parameters Guide")

for feature, description in feature_descriptions.items():
    st.sidebar.markdown(f"**{feature}**")
    st.sidebar.write(description)
    st.sidebar.write("---")

st.title("Predictive Maintenance Analytics Dashboard")

# Input sliders and dropdown
col1, col2, col3 = st.columns(3)

with col1:
    engine_temp = st.slider(
        "Engine Temperature (°C)",
        float(features['Engine Temperature (°C)']['min']),
        float(features['Engine Temperature (°C)']['max']),
        float(features['Engine Temperature (°C)']['optimal']),
        step=0.1
    )
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
    brake_thickness = st.slider(
        "Brake Pad Thickness (mm)",
        float(features['Brake Pad Thickness (mm)']['min']),
        float(features['Brake Pad Thickness (mm)']['max']),
        float(features['Brake Pad Thickness (mm)']['optimal']),
        step=0.1
    )
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
    tire_pressure = st.slider(
        "Tire Pressure (PSI)",
        float(features['Tire Pressure (PSI)']['min']),
        float(features['Tire Pressure (PSI)']['max']),
        float(features['Tire Pressure (PSI)']['optimal']),
        step=0.1
    )
    if tire_pressure < 28 or tire_pressure > 35:
        st.markdown("<p class='warning'>Warning: Pressure needs adjustment</p>", unsafe_allow_html=True)
        st.markdown("<p class='warning'>Recommendation: Adjust pressure to 30-35 PSI.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='good'>Optimal pressure range</p>", unsafe_allow_html=True)
        st.markdown("<p class='good'>Recommendation: No action required.</p>", unsafe_allow_html=True)

maintenance_type = st.selectbox(
    "Maintenance Type",
    ["Routine Maintenance", "Component Replacement", "Repair"]
)

# Input processing
maintenance_type_mapping = {"Routine Maintenance": 0, "Component Replacement": 1, "Repair": 2}
user_input = pd.DataFrame({
    "Engine Temperature (°C)": [engine_temp],
    "Brake Pad Thickness (mm)": [brake_thickness],
    "Tire Pressure (PSI)": [tire_pressure],
    "Maintenance Type Label": [maintenance_type_mapping[maintenance_type]]
})

input_data_scaled = scaler.transform(user_input)
input_data_lstm = input_data_scaled.reshape(1, 1, user_input.shape[1])

# Predictions
rul_prediction = lstm_model.predict(input_data_lstm)[0][0]
anomaly_prediction = random_forest_model.predict(input_data_scaled)[0]
anomaly_score = calculate_anomaly_probability(user_input.iloc[0])
health_score = calculate_health_score(user_input)

rul_days = int(calculate_rul(user_input.iloc[0], rul_prediction))
is_anomaly = determine_anomaly_status(user_input.iloc[0], anomaly_prediction, anomaly_score)

# Results Display
st.header("System Health Analysis")
metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.subheader("Remaining Useful Life")
    st.metric("RUL (days)", rul_days)

with metric_col2:
    st.subheader("Anomaly Status")
    anomaly_status = "Anomaly Detected" if is_anomaly else "Normal"
    st.metric("Status", anomaly_status)

with metric_col3:
    st.subheader("Health Score")
    st.metric("Score", f"{health_score}/100")

st.header("Maintenance Recommendations")
recommendations = []

if engine_temp > 100:
    recommendations.append("Check cooling system immediately.")
if brake_thickness < 3:
    recommendations.append("Replace brake pads urgently.")
elif brake_thickness < 5:
    recommendations.append("Schedule brake pad replacement soon.")
if tire_pressure < 30 or tire_pressure > 35:
    recommendations.append("Adjust tire pressure to optimal range.")
if rul_days < 30:
    recommendations.append("Schedule comprehensive maintenance within 30 days.")
if anomaly_score > 0.7:
    recommendations.append("Schedule comprehensive inspection due to high anomaly probability")
elif rul_days < 90:
    recommendations.append("Plan for maintenance within 90 days")

if recommendations:
    for rec in recommendations:
        st.markdown(rec)
else:
    st.markdown("All systems operating within normal parameters")

# Footer with model performance metrics
st.markdown("---")
st.markdown("### Model Performance Metrics")
col1, col2 = st.columns(2)

# Calculate RUL predictions for test data
rul_test_predictions = lstm_model.predict(X_test_lstm)
r2_value = r2_score(y_test_rul, rul_test_predictions)

# Calculate basic anomaly detection accuracy
y_pred_anomaly = random_forest_model.predict(X_test)
anomaly_accuracy = accuracy_score(y_test_anomaly, y_pred_anomaly)

with col1:
    st.metric("RUL Prediction R²", f"{r2_value:.3f}")

with col2:
    st.metric("Anomaly Detection Accuracy", f"{anomaly_accuracy:.1%}")