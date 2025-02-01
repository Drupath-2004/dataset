import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

data = pd.read_csv('cars_hyundai_one_hot_encoded_improved.csv')

features = ['Engine Temperature (°C)', 'Brake Pad Thickness (mm)', 'Tire Pressure (PSI)']
X = data[features]
y_rul = data['Adjusted_RUL']
y_anomaly = data['Anomaly Indication']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train_rul, y_test_rul = train_test_split(X_scaled, y_rul, test_size=0.2, random_state=42)
_, _, y_train_anomaly, y_test_anomaly = train_test_split(X_scaled, y_anomaly, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=8, random_state=42)
rf_model.fit(X_train, y_train_anomaly)

X_train_lstm = np.expand_dims(X_train, axis=1)
X_test_lstm = np.expand_dims(X_test, axis=1)

lstm_model_rul = Sequential([
    LSTM(128, activation='relu', input_shape=(1, X_train.shape[1]), return_sequences=True),
    Dropout(0.3),
    BatchNormalization(),
    LSTM(64, activation='relu', return_sequences=False),
    Dropout(0.3),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

lstm_model_rul.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)

history_rul = lstm_model_rul.fit(X_train_lstm, y_train_rul, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=1)

lstm_model_rul.save("lstm_model_rul.h5")
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test_rul, "y_test_rul.pkl")
joblib.dump(y_test_anomaly, "y_test_anomaly.pkl")

print("Models and data saved successfully.")
