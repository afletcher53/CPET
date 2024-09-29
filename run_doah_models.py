import copy
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
    r2_score
)
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from models.models import (
    CNNModel,
    CombinedCNNDnnModel,
    CombinedDNNModel,
    DNNModel,
    DNNTimeseriesModel,
    LSTMModel,
    SuperLearner,
    create_timeseries_model_wrapper,
    init_weights,
    load_and_preprocess_data,
    set_random_seeds,
)
from plotting_functions import (
    plot_roc_curves,
    # plot_pr_curves
)

class IntegerRegressionLSTMDnnModel(nn.Module):
    def __init__(self, bxb_input_size, cat_input_size, lstm_hidden_size, dnn_hidden_sizes, output_size, dropout_rate=0.3):
        super(IntegerRegressionLSTMDnnModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size=bxb_input_size, hidden_size=lstm_hidden_size, batch_first=True, num_layers=2, dropout=dropout_rate)
        self.bn_lstm = nn.BatchNorm1d(lstm_hidden_size)
        
        dnn_layers = []
        input_size = cat_input_size
        for hidden_size in dnn_hidden_sizes:
            dnn_layers.append(nn.Linear(input_size, hidden_size))
            dnn_layers.append(nn.BatchNorm1d(hidden_size))
            dnn_layers.append(nn.LeakyReLU())
            dnn_layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        self.dnn = nn.Sequential(*dnn_layers)
        
        self.output_layer = nn.Linear(lstm_hidden_size + dnn_hidden_sizes[-1], output_size)
        
    def forward(self, x_bxb, x_cat):
        lstm_out, _ = self.lstm(x_bxb)
        lstm_last = lstm_out[:, -1, :]
        lstm_last = self.bn_lstm(lstm_last)
        
        dnn_out = self.dnn(x_cat)
        
        combined = torch.cat((lstm_last, dnn_out), dim=1)
        output = self.output_layer(combined)
        
        # Round the output to the nearest integer
        return torch.round(output)

class IntegerMSELoss(nn.Module):
    def __init__(self):
        super(IntegerMSELoss, self).__init__()

    def forward(self, pred, target):
        rounded_pred = torch.round(pred)
        return F.mse_loss(rounded_pred, target)

def train_integer_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, patience=15):
    best_val_loss = float('inf')
    counter = 0
    best_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            X_bxb_batch, X_cat_batch, y_batch = [t.to(device) for t in batch]
            optimizer.zero_grad()
            outputs = model(X_bxb_batch, X_cat_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                X_bxb_val, X_cat_val, y_val = [t.to(device) for t in batch]
                val_outputs = model(X_bxb_val, X_cat_val)
                val_loss += criterion(val_outputs, y_val).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            best_model = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                model.load_state_dict(best_model)
                break

    return model

def evaluate_integer_regression_model(model, data_loader, device, scaler_y):
    model.eval()
    true_values = []
    predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            X_bxb_batch, X_cat_batch, y_batch = [t.to(device) for t in batch]
            outputs = model(X_bxb_batch, X_cat_batch)
            true_values.extend(y_batch.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

    true_values = np.array(true_values)
    predictions = np.array(predictions)
    
    # Inverse transform the predictions and true values
    true_values_original = scaler_y.inverse_transform(true_values)
    predictions_original = scaler_y.inverse_transform(predictions)
    
    # Round predictions to nearest integer
    predictions_original = np.round(predictions_original)
    
    # Print out a selection of predictions in original scale
    print("Original scale:")
    print(f"True: {true_values_original[:5].flatten().astype(int)}")
    print(f"Pred: {predictions_original[:5].flatten().astype(int)}")
    
    # Calculate metrics using the original scale values
    mae = mean_absolute_error(true_values_original, predictions_original)
    mse = mean_squared_error(true_values_original, predictions_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values_original, predictions_original)
    
    # Calculate MAPE
    mape = np.mean(np.abs((true_values_original - predictions_original) / true_values_original)) * 100
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }

class OptimizedCombinedLSTMDnnModel(nn.Module):
    def __init__(self, bxb_input_size, cat_input_size, lstm_hidden_size, dnn_hidden_sizes, output_size, dropout_rate=0.3):
        super(OptimizedCombinedLSTMDnnModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size=bxb_input_size, hidden_size=lstm_hidden_size, batch_first=True, num_layers=2, dropout=dropout_rate)
        self.bn_lstm = nn.BatchNorm1d(lstm_hidden_size)
        
        dnn_layers = []
        input_size = cat_input_size
        for hidden_size in dnn_hidden_sizes:
            dnn_layers.append(nn.Linear(input_size, hidden_size))
            dnn_layers.append(nn.BatchNorm1d(hidden_size))
            dnn_layers.append(nn.LeakyReLU())
            dnn_layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        self.dnn = nn.Sequential(*dnn_layers)
        
        self.output_layer = nn.Linear(lstm_hidden_size + dnn_hidden_sizes[-1], output_size)
        
    def forward(self, x_bxb, x_cat):
        lstm_out, _ = self.lstm(x_bxb)
        lstm_last = lstm_out[:, -1, :]
        lstm_last = self.bn_lstm(lstm_last)
        
        dnn_out = self.dnn(x_cat)
        
        combined = torch.cat((lstm_last, dnn_out), dim=1)
        output = self.output_layer(combined)
        
        return output


def preprocess_data(X_bxb, X_cat, y):
    scaler_bxb = StandardScaler()
    scaler_cat = StandardScaler()
    scaler_y = StandardScaler()
    
    X_bxb_scaled = scaler_bxb.fit_transform(X_bxb.reshape(-1, X_bxb.shape[-1])).reshape(X_bxb.shape)
    X_cat_scaled = scaler_cat.fit_transform(X_cat)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_bxb_scaled, X_cat_scaled, y_scaled, (scaler_bxb, scaler_cat, scaler_y)

def create_dataloaders(X_bxb, X_cat, y, batch_size):
    dataset = TensorDataset(
        torch.FloatTensor(X_bxb),
        torch.FloatTensor(X_cat),
        torch.FloatTensor(y).unsqueeze(1)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class CombinedLSTMDnnModel(nn.Module):
    def __init__(self, bxb_input_size, cat_input_size, lstm_hidden_size, dnn_hidden_sizes, output_size):
        super(CombinedLSTMDnnModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size=bxb_input_size, hidden_size=lstm_hidden_size, batch_first=True)
        self.bn_lstm = nn.BatchNorm1d(lstm_hidden_size)
        
        dnn_layers = []
        input_size = cat_input_size
        for hidden_size in dnn_hidden_sizes:
            dnn_layers.append(nn.Linear(input_size, hidden_size))
            dnn_layers.append(nn.BatchNorm1d(hidden_size))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        self.dnn = nn.Sequential(*dnn_layers)
        
        self.output_layer = nn.Linear(lstm_hidden_size + dnn_hidden_sizes[-1], output_size)
        
    def forward(self, x_bxb, x_cat):
        lstm_out, _ = self.lstm(x_bxb)
        lstm_last = lstm_out[:, -1, :]
        lstm_last = self.bn_lstm(lstm_last)
        
        dnn_out = self.dnn(x_cat)
        
        combined = torch.cat((lstm_last, dnn_out), dim=1)
        output = self.output_layer(combined)
        
        return output

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, patience=15):
    best_val_loss = float('inf')
    counter = 0
    best_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            X_bxb_batch, X_cat_batch, y_batch = [t.to(device) for t in batch]
            optimizer.zero_grad()
            outputs = model(X_bxb_batch, X_cat_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                X_bxb_val, X_cat_val, y_val = [t.to(device) for t in batch]
                val_outputs = model(X_bxb_val, X_cat_val)
                val_loss += criterion(val_outputs, y_val).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            best_model = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                model.load_state_dict(best_model)
                break

    return model


def  optimize_hyperparameters(X_bxb_train, X_cat_train, y_train):
    # Use RandomForestRegressor for quick hyperparameter optimization
    rf_model = RandomForestRegressor(n_jobs=-1, random_state=42)
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=20, cv=3, random_state=42)
    
    # Combine X_bxb and X_cat for RandomForestRegressor
    X_combined = np.concatenate((X_bxb_train.reshape(X_bxb_train.shape[0], -1), X_cat_train), axis=1)
    
    random_search.fit(X_combined, y_train)
    
    best_params = random_search.best_params_
    print("Best hyperparameters:", best_params)
    
    # Convert RandomForest hyperparameters to neural network hyperparameters
    nn_params = {
        'lstm_hidden_size': best_params['max_depth'] * 4 if best_params['max_depth'] else 64,
        'dnn_hidden_sizes': [best_params['n_estimators'], best_params['n_estimators'] // 2],
        'batch_size': 2 ** (best_params['min_samples_split'] + 4),  # Convert to power of 2
        'learning_rate': 0.001 * (best_params['min_samples_leaf'] / 2)  # Adjust learning rate based on leaf size
    }
    return nn_params


def create_model_and_loaders(
    model_class, X_train, X_cat_train, y_train, X_val, X_cat_val, y_val, best_params
):
    if model_class in [CombinedLSTMDnnModel, CombinedCNNDnnModel]:
        model = model_class(
            X_train.shape[2],
            X_cat_train.shape[1],
            best_params["lstm_hidden_size"],
            best_params["dnn_hidden_sizes"],
            1,
        )
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(X_cat_train),
            torch.FloatTensor(y_train).unsqueeze(1),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(X_cat_val),
            torch.FloatTensor(y_val).unsqueeze(1),
        )
    elif model_class == LSTMModel:
        model = model_class(X_train.shape[2], best_params["lstm_hidden_size"], 1)
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1)
        )
    elif model_class == DNNModel:
        model = model_class(X_cat_train.shape[1], best_params["dnn_hidden_sizes"], 1)
        train_dataset = TensorDataset(
            torch.FloatTensor(X_cat_train), torch.FloatTensor(y_train).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_cat_val), torch.FloatTensor(y_val).unsqueeze(1)
        )
    elif model_class == CNNModel:
        model = model_class(X_train.shape[2], X_train.shape[1], 1)
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1)
        )
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

    model.apply(init_weights)
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_params["batch_size"],
        shuffle=True,
        num_workers=0,
        worker_init_fn=lambda _: set_random_seeds(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=best_params["batch_size"], shuffle=False
    )

    return model, train_loader, val_loader


def evaluate_regression_model(model, data_loader, device, scaler_y):
    model.eval()
    true_values = []
    predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            X_bxb_batch, X_cat_batch, y_batch = [t.to(device) for t in batch]
            outputs = model(X_bxb_batch, X_cat_batch)
            true_values.extend(y_batch.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

    true_values = np.array(true_values)
    predictions = np.array(predictions)
    
    # Inverse transform the predictions and true values
    true_values_original = scaler_y.inverse_transform(true_values)
    predictions_original = scaler_y.inverse_transform(predictions)
    
    # Print out a selection of predictions in original scale
    print("Original scale:")
    print(f"True: {true_values_original[:5].flatten()}")
    print(f"Pred: {predictions_original[:5].flatten()}")
    
    # Calculate metrics using the original scale values
    mae = mean_absolute_error(true_values_original, predictions_original)
    mse = mean_squared_error(true_values_original, predictions_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values_original, predictions_original)
    
    # Calculate MAPE
    mape = np.mean(np.abs((true_values_original - predictions_original) / true_values_original)) * 100
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }

def main():
    # Load and preprocess data (assuming this part remains the same)
    X_bxb_train, X_cat_train, y_train, X_bxb_val, X_cat_val, y_val, X_bxb_test, X_cat_test, y_test = load_and_preprocess_data(directory="./data/ml_inputs/daoh_30")
    X_bxb_train, X_cat_train, y_train, scalers = preprocess_data(X_bxb_train, X_cat_train, y_train)
    X_bxb_val, X_cat_val, y_val, _ = preprocess_data(X_bxb_val, X_cat_val, y_val)
    X_bxb_test, X_cat_test, y_test, _ = preprocess_data(X_bxb_test, X_cat_test, y_test)

    # Create dataloaders (assuming this part remains the same)
    best_params = {'lstm_hidden_size': 64, 'dnn_hidden_sizes': [200, 100], 'batch_size': 512, 'learning_rate': 0.001}
    train_loader = create_dataloaders(X_bxb_train, X_cat_train, y_train, best_params['batch_size'])
    val_loader = create_dataloaders(X_bxb_val, X_cat_val, y_val, best_params['batch_size'])
    test_loader = create_dataloaders(X_bxb_test, X_cat_test, y_test, best_params['batch_size'])

    # Initialize model
    model = IntegerRegressionLSTMDnnModel(
        X_bxb_train.shape[2],
        X_cat_train.shape[1],
        best_params['lstm_hidden_size'],
        best_params['dnn_hidden_sizes'],
        1
    )

    # Define loss function and optimizer
    criterion = IntegerMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train_integer_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

    # Evaluate model
    test_metrics = evaluate_integer_regression_model(trained_model, test_loader, device, scalers[2])
    print("Test metrics:", test_metrics)

    # Save the model
    torch.save(trained_model.state_dict(), "integer_regression_lstm_dnn_model.pth")

if __name__ == "__main__":
    main()
