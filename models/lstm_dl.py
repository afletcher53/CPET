import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math
import wandb
from sklearn.model_selection import ParameterGrid

# Initialize wandb
wandb.init(project="combined-model", name="hyperparameter-tuning")

# Load and check data
def load_and_check_data(file_name):
    data = np.load(file_name)
    print(f"{file_name} shape: {data.shape}")
    print(f"Contains NaN: {np.isnan(data).any()}")
    print(f"Contains Inf: {np.isinf(data).any()}")
    print(f"Min: {np.min(data)}, Max: {np.max(data)}")
    return data

def handle_nan_values(data, strategy='mean'):
    if strategy == 'mean':
        return np.nan_to_num(data, nan=np.nanmean(data, axis=0))
    elif strategy == 'median':
        return np.nan_to_num(data, nan=np.nanmedian(data, axis=0))
    elif strategy == 'zero':
        return np.nan_to_num(data, nan=0)
    else:
        raise ValueError("Invalid strategy. Choose 'mean', 'median', or 'zero'.")

# Load and preprocess data
X_bxb_train = load_and_check_data('X_bxb_train.npy')
X_bxb_val = load_and_check_data('X_bxb_val.npy')
X_bxb_test = load_and_check_data('X_bxb_test.npy')
X_cat_train = load_and_check_data('X_cat_train.npy')
X_cat_val = load_and_check_data('X_cat_val.npy')
X_cat_test = load_and_check_data('X_cat_test.npy')

X_cat_train = handle_nan_values(X_cat_train, strategy='mean')
X_cat_val = handle_nan_values(X_cat_val, strategy='mean')
X_cat_test = handle_nan_values(X_cat_test, strategy='mean')

y_train = load_and_check_data('y_train.npy')
y_val = load_and_check_data('y_val.npy')
y_test = load_and_check_data('y_test.npy')

print("Class distribution (train):", np.unique(y_train, return_counts=True))
print("Class distribution (val):", np.unique(y_val, return_counts=True))

class CombinedModel(nn.Module):
    def __init__(self, bxb_input_size, cat_input_size, lstm_hidden_size, dnn_hidden_sizes, output_size):
        super(CombinedModel, self).__init__()
        
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

# Convert numpy arrays to PyTorch tensors
X_bxb_train_tensor = torch.FloatTensor(X_bxb_train)
X_cat_train_tensor = torch.FloatTensor(X_cat_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)

X_bxb_val_tensor = torch.FloatTensor(X_bxb_val)
X_cat_val_tensor = torch.FloatTensor(X_cat_val)
y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

X_bxb_test_tensor = torch.FloatTensor(X_bxb_test)
X_cat_test_tensor = torch.FloatTensor(X_cat_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

# Hyperparameter grid
param_grid = {
    'lstm_hidden_size': [16, 32, 64],
    'dnn_hidden_sizes': [[16, 8], [32, 16], [64, 32]],
    'batch_size': [128, 256, 512],
    'learning_rate': [0.001, 0.0001, 0.00001]
}

# Function to train and evaluate model
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_bxb_batch, X_cat_batch, y_batch in train_loader:
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
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_bxb_val, X_cat_val, y_val in val_loader:
                val_outputs = model(X_bxb_val, X_cat_val)
                val_loss += criterion(val_outputs, y_val).item()
                val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
                val_correct += (val_preds == y_val).float().sum().item()
                val_total += y_val.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy
        })
    
    return best_val_loss, val_accuracy

# Grid search
best_params = None
best_val_loss = float('inf')

for i, params in enumerate(ParameterGrid(param_grid)):
    run = wandb.init(project="combined-model", name=f"hparam-tuning-{i}", config=params, reinit=True)
    
    model = CombinedModel(X_bxb_train.shape[2], X_cat_train.shape[1], params['lstm_hidden_size'], params['dnn_hidden_sizes'], 1)
    model.apply(init_weights)
    
    train_dataset = TensorDataset(X_bxb_train_tensor, X_cat_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    
    val_dataset = TensorDataset(X_bxb_val_tensor, X_cat_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    class_weights = torch.tensor([1.0, (2330/160)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    
    val_loss, val_accuracy = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler)
    
    print(f"Params: {params}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = params
    
    wandb.finish()

print("Best parameters:", best_params)
print("Best validation loss:", best_val_loss)

# Train final model with best parameters
wandb.init(project="combined-model", name="final-model", config=best_params)

final_model = CombinedModel(X_bxb_train.shape[2], X_cat_train.shape[1], best_params['lstm_hidden_size'], best_params['dnn_hidden_sizes'], 1)
final_model.apply(init_weights)

train_dataset = TensorDataset(X_bxb_train_tensor, X_cat_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)

val_dataset = TensorDataset(X_bxb_val_tensor, X_cat_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)

criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
optimizer = optim.AdamW(final_model.parameters(), lr=best_params['learning_rate'], weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

train_and_evaluate(final_model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100)

# Final evaluation
final_model.eval()
with torch.no_grad():
    test_outputs = final_model(X_bxb_test_tensor, X_cat_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    test_preds = (torch.sigmoid(test_outputs) > 0.5).float()
    accuracy = (test_preds == y_test_tensor).float().mean()
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    wandb.log({
        "test_loss": test_loss.item(),
        "test_accuracy": accuracy.item()
    })

wandb.finish()