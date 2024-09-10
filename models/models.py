import numpy as np
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



