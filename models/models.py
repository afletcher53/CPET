import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import wandb
from sklearn.model_selection import ParameterGrid
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_auc_score, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

def set_random_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def handle_nan_values(data, strategy='mean'):
    if strategy == 'mean':
        return np.nan_to_num(data, nan=np.nanmean(data, axis=0))
    elif strategy == 'median':
        return np.nan_to_num(data, nan=np.nanmedian(data, axis=0))
    elif strategy == 'zero':
        return np.nan_to_num(data, nan=0)
    else:
        raise ValueError("Invalid strategy. Choose 'mean', 'median', or 'zero'.")
def load_and_preprocess_data(directory=''):
    def load_file(filename):
        return np.load(os.path.join(directory, filename))

    X_bxb_train = load_file('X_bxb_train.npy')
    X_bxb_val = load_file('X_bxb_val.npy')
    X_bxb_test = load_file('X_bxb_test.npy')
    X_cat_train = load_file('X_cat_train.npy')
    X_cat_val = load_file('X_cat_val.npy')
    X_cat_test = load_file('X_cat_test.npy')
    y_train = load_file('y_train.npy')
    y_val = load_file('y_val.npy')
    y_test = load_file('y_test.npy')

    X_cat_train = handle_nan_values(X_cat_train, strategy='mean')
    X_cat_val = handle_nan_values(X_cat_val, strategy='mean')
    X_cat_test = handle_nan_values(X_cat_test, strategy='mean')

    scaler = StandardScaler()
    X_cat_train = scaler.fit_transform(X_cat_train)
    X_cat_val = scaler.transform(X_cat_val)
    X_cat_test = scaler.transform(X_cat_test)

    return (X_bxb_train, X_cat_train, y_train,
            X_bxb_val, X_cat_val, y_val,
            X_bxb_test, X_cat_test, y_test)

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

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_last = lstm_out[:, -1, :]
        lstm_last = self.bn(lstm_last)
        output = self.fc(lstm_last)
        return output

class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DNNModel, self).__init__()
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.dnn = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.dnn(x)

class CNNModel(nn.Module):
    def __init__(self, input_channels, seq_length, output_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        self.fc_input_dim = self.calculate_conv_output_size(input_channels, seq_length)
        
        self.fc1 = nn.Linear(self.fc_input_dim, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.2)

    def calculate_conv_output_size(self, channels, length):
        x = torch.randn(1, channels, length)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class CombinedCNNDnnModel(nn.Module):
    def __init__(self, cnn_input_channels, cnn_seq_length, cat_input_size, dnn_hidden_sizes, output_size):
        super(CombinedCNNDnnModel, self).__init__()
        
        self.cnn = CNNModel(cnn_input_channels, cnn_seq_length, 32)  # Output 32 features
        
        dnn_layers = []
        input_size = cat_input_size
        for hidden_size in dnn_hidden_sizes:
            dnn_layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = hidden_size
        self.dnn = nn.Sequential(*dnn_layers)
        
        self.output_layer = nn.Linear(32 + dnn_hidden_sizes[-1], output_size)
        
    def forward(self, x_bxb, x_cat):
        cnn_out = self.cnn(x_bxb)
        dnn_out = self.dnn(x_cat)
        combined = torch.cat((cnn_out, dnn_out), dim=1)
        output = self.output_layer(combined)
        return output

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_uniform_(param.data, nonlinearity='sigmoid')
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

class DNNTimeseriesModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DNNTimeseriesModel, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        return self.network(x)
class CombinedDNNModel(nn.Module):
    def __init__(self, static_input_size, timeseries_input_size, hidden_sizes, output_size):
        super(CombinedDNNModel, self).__init__()
        
        self.static_dnn = DNNModel(static_input_size, hidden_sizes, hidden_sizes[-1])
        self.timeseries_dnn = DNNTimeseriesModel(timeseries_input_size, hidden_sizes, hidden_sizes[-1])
        
        self.output_layer = nn.Linear(hidden_sizes[-1] * 2, output_size)
        
    def forward(self, x_timeseries, x_static):
        # Flatten the timeseries input
        x_timeseries = x_timeseries.view(x_timeseries.size(0), -1)
        
        static_out = self.static_dnn(x_static)
        timeseries_out = self.timeseries_dnn(x_timeseries)
        
        combined = torch.cat((static_out, timeseries_out), dim=1)
        output = self.output_layer(combined)
        
        return output


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

class SuperLearner:
    def __init__(self, base_models, meta_model=None, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else LogisticRegression()
        self.n_folds = n_folds

    def fit(self, X, y):
        n_models = len(self.base_models)
        n_samples = X.shape[0]
        
        # Initialize out-of-fold predictions
        S_train = np.zeros((n_samples, n_models))
        
        # K-fold cross-validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Train base models and generate out-of-fold predictions
        for i, model in enumerate(self.base_models):
            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]
                
                model.fit(X_train, y_train)
                S_train[val_index, i] = model.predict_proba(X_val)[:, 1]
        
        # Fit the meta-model on the out-of-fold predictions
        self.meta_model.fit(S_train, y)
        
        # Retrain base models on full dataset
        for model in self.base_models:
            model.fit(X, y)

    def predict_proba(self, X):
        n_models = len(self.base_models)
        n_samples = X.shape[0]
        
        # Generate predictions from base models
        S_test = np.zeros((n_samples, n_models))
        for i, model in enumerate(self.base_models):
            S_test[:, i] = model.predict_proba(X)[:, 1]
        
        # Use meta-model to make final predictions
        return self.meta_model.predict_proba(S_test)

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class SuperLearner:
    def __init__(self, base_models, meta_model=None, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else LogisticRegression()
        self.n_folds = n_folds

    def fit(self, X, y):
        n_models = len(self.base_models)
        n_samples = X.shape[0]
        
        # Initialize out-of-fold predictions
        S_train = np.zeros((n_samples, n_models))
        
        # K-fold cross-validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Train base models and generate out-of-fold predictions
        for i, model in enumerate(self.base_models):
            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]
                
                model.fit(X_train, y_train)
                S_train[val_index, i] = model.predict_proba(X_val)[:, 1]
        
        # Fit the meta-model on the out-of-fold predictions
        self.meta_model.fit(S_train, y)
        
        # Retrain base models on full dataset
        for model in self.base_models:
            model.fit(X, y)

    def predict_proba(self, X):
        n_models = len(self.base_models)
        n_samples = X.shape[0]
        
        # Generate predictions from base models
        S_test = np.zeros((n_samples, n_models))
        for i, model in enumerate(self.base_models):
            S_test[:, i] = model.predict_proba(X)[:, 1]
        
        # Use meta-model to make final predictions
        return self.meta_model.predict_proba(S_test)

def create_pytorch_model_wrapper(model_class, **kwargs):
    class PyTorchModelWrapper:
        def __init__(self):
            self.model = model_class(**kwargs)
            self.model.eval()
            self.is_combined_model = isinstance(self.model, (CombinedLSTMDnnModel, CombinedCNNDnnModel))

        def fit(self, X, y):
            self.model.train()
            if self.is_combined_model:
                X_bxb, X_cat = X[:, :, :kwargs['bxb_input_size']], X[:, 0, kwargs['bxb_input_size']:]
                X_bxb_tensor = torch.FloatTensor(X_bxb)
                X_cat_tensor = torch.FloatTensor(X_cat)
                y_tensor = torch.FloatTensor(y).unsqueeze(1)
                dataset = TensorDataset(X_bxb_tensor, X_cat_tensor, y_tensor)
            else:
                X_tensor = torch.FloatTensor(X)
                y_tensor = torch.FloatTensor(y).unsqueeze(1)
                dataset = TensorDataset(X_tensor, y_tensor)
            
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(self.model.parameters())
            
            for epoch in range(10):  # You may want to adjust the number of epochs
                for batch in dataloader:
                    optimizer.zero_grad()
                    if self.is_combined_model:
                        X_bxb_batch, X_cat_batch, y_batch = batch
                        outputs = self.model(X_bxb_batch, X_cat_batch)
                    else:
                        X_batch, y_batch = batch
                        outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
            
            self.model.eval()

        def predict_proba(self, X):
            with torch.no_grad():
                if self.is_combined_model:
                    X_bxb, X_cat = X[:, :, :kwargs['bxb_input_size']], X[:, 0, kwargs['bxb_input_size']:]
                    X_bxb_tensor = torch.FloatTensor(X_bxb)
                    X_cat_tensor = torch.FloatTensor(X_cat)
                    outputs = self.model(X_bxb_tensor, X_cat_tensor)
                else:
                    X_tensor = torch.FloatTensor(X)
                    outputs = self.model(X_tensor)
                probs = torch.sigmoid(outputs)
            return np.column_stack((1 - probs.numpy(), probs.numpy()))

    return PyTorchModelWrapper()

def create_timeseries_model_wrapper(model_class, **kwargs):
    class TimeSeriesModelWrapper:
        def __init__(self):
            # For CNNModel, input_channels and seq_length are expected, not input_size and hidden_size
            if model_class == CNNModel:
                self.model = model_class(input_channels=kwargs['input_channels'], seq_length=kwargs['seq_length'], output_size=kwargs['output_size'])
            else:
                self.model = model_class(input_size=kwargs['input_size'], hidden_size=kwargs['hidden_size'], output_size=kwargs['output_size'])
            self.model.eval()

        def fit(self, X, y):
            self.model.train()
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(self.model.parameters())

            for epoch in range(10):  # Adjust the number of epochs as needed
                for batch in dataloader:
                    optimizer.zero_grad()
                    X_batch, y_batch = batch
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

            self.model.eval()

        def predict_proba(self, X):
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                outputs = self.model(X_tensor)
                probs = torch.sigmoid(outputs)
            return np.column_stack((1 - probs.numpy(), probs.numpy()))  # Return probabilities

    return TimeSeriesModelWrapper()
