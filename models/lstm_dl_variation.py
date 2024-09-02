import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate

# Data Preparation
def prepare_data(n_samples=1000, n_features=10, n_timesteps=100):
    # Generate sample data
    np.random.seed(0)
    time_series = np.cumsum(np.random.randn(n_samples, n_timesteps, 1), axis=1)
    additional_features = np.random.randn(n_samples, n_features)
    target = np.sum(time_series[:, -10:, 0], axis=1) + np.sum(additional_features, axis=1)
    
    # Normalize data
    scaler_ts = MinMaxScaler()
    scaler_af = MinMaxScaler()
    scaler_target = MinMaxScaler()
    
    time_series_scaled = scaler_ts.fit_transform(time_series.reshape(-1, 1)).reshape(n_samples, n_timesteps, 1)
    additional_features_scaled = scaler_af.fit_transform(additional_features)
    target_scaled = scaler_target.fit_transform(target.reshape(-1, 1)).flatten()
    
    # Split data
    X_lstm_train, X_lstm_test, X_dnn_train, X_dnn_test, y_train, y_test = train_test_split(
        time_series_scaled, additional_features_scaled, target_scaled, test_size=0.2, random_state=42)
    
    return X_lstm_train, X_lstm_test, X_dnn_train, X_dnn_test, y_train, y_test, scaler_target

# Model Creation Functions
def create_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm = LSTM(64, return_sequences=True)(inputs)
    lstm = LSTM(32)(lstm)
    outputs = Dense(1)(lstm)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def create_dnn_model(input_shape):
    inputs = Input(shape=input_shape)
    dense = Dense(64, activation='relu')(inputs)
    dense = Dense(32, activation='relu')(dense)
    outputs = Dense(1)(dense)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def create_combined_model(lstm_input_shape, dnn_input_shape):
    lstm_input = Input(shape=lstm_input_shape)
    lstm = LSTM(64, return_sequences=True)(lstm_input)
    lstm = LSTM(32)(lstm)

    dnn_input = Input(shape=dnn_input_shape)
    dense = Dense(64, activation='relu')(dnn_input)
    dense = Dense(32, activation='relu')(dense)

    combined = concatenate([lstm, dense])
    outputs = Dense(1)(combined)

    model = Model(inputs=[lstm_input, dnn_input], outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Training and Evaluation
def train_and_evaluate_models(X_lstm_train, X_lstm_test, X_dnn_train, X_dnn_test, y_train, y_test, epochs=100):
    lstm_model = create_lstm_model(X_lstm_train.shape[1:])
    dnn_model = create_dnn_model(X_dnn_train.shape[1:])
    combined_model = create_combined_model(X_lstm_train.shape[1:], X_dnn_train.shape[1:])

    lstm_history = lstm_model.fit(X_lstm_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)
    dnn_history = dnn_model.fit(X_dnn_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)
    combined_history = combined_model.fit([X_lstm_train, X_dnn_train], y_train, epochs=epochs, validation_split=0.2, verbose=0)

    lstm_score = lstm_model.evaluate(X_lstm_test, y_test, verbose=0)
    dnn_score = dnn_model.evaluate(X_dnn_test, y_test, verbose=0)
    combined_score = combined_model.evaluate([X_lstm_test, X_dnn_test], y_test, verbose=0)

    return lstm_history, dnn_history, combined_history, lstm_score, dnn_score, combined_score

# Visualization
def plot_training_history(lstm_history, dnn_history, combined_history):
    plt.figure(figsize=(12, 4))
    plt.plot(lstm_history.history['val_loss'], label='LSTM')
    plt.plot(dnn_history.history['val_loss'], label='DNN')
    plt.plot(combined_history.history['val_loss'], label='Combined')
    plt.title('Model Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Prepare data
    X_lstm_train, X_lstm_test, X_dnn_train, X_dnn_test, y_train, y_test, scaler_target = prepare_data()

    # Train and evaluate models
    lstm_history, dnn_history, combined_history, lstm_score, dnn_score, combined_score = train_and_evaluate_models(
        X_lstm_train, X_lstm_test, X_dnn_train, X_dnn_test, y_train, y_test)

    # Print results
    print(f"LSTM Model Score: {lstm_score}")
    print(f"DNN Model Score: {dnn_score}")
    print(f"Combined Model Score: {combined_score}")

    # Visualize training history
    plot_training_history(lstm_history, dnn_history, combined_history)