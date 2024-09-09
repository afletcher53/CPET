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

def set_random_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

def load_and_preprocess_data():
    X_bxb_train = np.load('X_bxb_train.npy')
    X_bxb_val = np.load('X_bxb_val.npy')
    X_bxb_test = np.load('X_bxb_test.npy')
    X_cat_train = np.load('X_cat_train.npy')
    X_cat_val = np.load('X_cat_val.npy')
    X_cat_test = np.load('X_cat_test.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')

    X_cat_train = handle_nan_values(X_cat_train, strategy='mean')
    X_cat_val = handle_nan_values(X_cat_val, strategy='mean')
    X_cat_test = handle_nan_values(X_cat_test, strategy='mean')

    X_bxb_train_reshaped = X_bxb_train.reshape(X_bxb_train.shape[0], -1)
    X_combined_train = np.concatenate((X_bxb_train_reshaped, X_cat_train), axis=1)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_combined_train, y_train)

    X_bxb_resampled = X_resampled[:, :X_bxb_train_reshaped.shape[1]].reshape(-1, X_bxb_train.shape[1], X_bxb_train.shape[2])
    X_cat_resampled = X_resampled[:, X_bxb_train_reshaped.shape[1]:]

    scaler = StandardScaler()
    X_cat_resampled = scaler.fit_transform(X_cat_resampled)
    X_cat_val = scaler.transform(X_cat_val)
    X_cat_test = scaler.transform(X_cat_test)

    return (X_bxb_resampled, X_cat_resampled, y_resampled,
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

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, model_name=""):
    best_val_loss = float('inf')
    step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if len(batch) == 3:  # Combined model
                X_bxb_batch, X_cat_batch, y_batch = batch
                outputs = model(X_bxb_batch, X_cat_batch)
            else:  # LSTM, DNN, or CNN model
                X_batch, y_batch = batch
                outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
            wandb.log({f"{model_name}/train_loss": loss.item()}, step=step)
            step += 1
        
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:  # Combined model
                    X_bxb_val, X_cat_val, y_val = batch
                    val_outputs = model(X_bxb_val, X_cat_val)
                else:  # LSTM, DNN, or CNN model
                    X_val, y_val = batch
                    val_outputs = model(X_val)
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
            f"{model_name}/epoch": epoch + 1,
            f"{model_name}/avg_train_loss": avg_train_loss,
            f"{model_name}/val_loss": avg_val_loss,
            f"{model_name}/val_accuracy": val_accuracy
        }, step=step)
        
        print(f"{model_name} - Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    return best_val_loss, val_accuracy, step

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:  # Combined model
                X_bxb_test, X_cat_test, y_test = batch
                outputs = model(X_bxb_test, X_cat_test)
            else:  # LSTM, DNN, or CNN model
                X_test, y_test = batch
                outputs = model(X_test)
            
            test_loss += criterion(outputs, y_test).item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_test.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    test_loss /= len(test_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    specificity = tn / (tn + fp)
    balanced_accuracy = (recall + specificity) / 2
    
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0
    
    metrics = {
        "test_loss": test_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mcc": mcc
    }
    
    return metrics, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    wandb.log({f"{model_name}_confusion_matrix": wandb.Image(f'{model_name}_confusion_matrix.png')})

def main():
    set_random_seeds()
    
    X_bxb_train, X_cat_train, y_train, X_bxb_val, X_cat_val, y_val, X_bxb_test, X_cat_test, y_test = load_and_preprocess_data()
    
    # Convert to tensors
    X_bxb_train_tensor = torch.FloatTensor(X_bxb_train)
    X_cat_train_tensor = torch.FloatTensor(X_cat_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_bxb_val_tensor = torch.FloatTensor(X_bxb_val)
    X_cat_val_tensor = torch.FloatTensor(X_cat_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    X_bxb_test_tensor = torch.FloatTensor(X_bxb_test)
    X_cat_test_tensor = torch.FloatTensor(X_cat_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Hyperparameters
    best_params = {
        'batch_size': 512,
        'dnn_hidden_sizes': [64, 32],
        'learning_rate': 1e-03,
        'lstm_hidden_size': 16,
        'num_epochs': 20
    }
    
    # Initialize the main run
    wandb.init(project="CPET", name="model_comparison", config=best_params)
    
    # Combined LSTM+DNN Model
    with wandb.init(project="CPET", name="Combined LSTM+DNN", config=best_params, group="model_comparison") as run:
        combined_model = CombinedLSTMDnnModel(X_bxb_train.shape[2], X_cat_train.shape[1], best_params['lstm_hidden_size'], best_params['dnn_hidden_sizes'], 1)
        combined_model.apply(init_weights)
        
        combined_train_dataset = TensorDataset(X_bxb_train_tensor, X_cat_train_tensor, y_train_tensor)
        combined_train_loader = DataLoader(combined_train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        combined_val_dataset = TensorDataset(X_bxb_val_tensor, X_cat_val_tensor, y_val_tensor)
        combined_val_loader = DataLoader(combined_val_dataset, batch_size=best_params['batch_size'], shuffle=False)
        combined_test_dataset = TensorDataset(X_bxb_test_tensor, X_cat_test_tensor, y_test_tensor)
        combined_test_loader = DataLoader(combined_test_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(combined_model.parameters(), lr=best_params['learning_rate'], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
        
        combined_best_val_loss, combined_val_accuracy, combined_steps = train_and_evaluate(
            combined_model, combined_train_loader, combined_val_loader, criterion, optimizer, scheduler, 
            num_epochs=best_params['num_epochs'], model_name="Combined LSTM+DNN"
        )
        
        combined_metrics, combined_preds, combined_labels = evaluate_model(combined_model, combined_test_loader, criterion)
        plot_confusion_matrix(combined_labels, combined_preds, "Combined LSTM+DNN")
        wandb.log({"Combined LSTM+DNN/test_metrics": combined_metrics})
    
    # LSTM-only Model
    with wandb.init(project="CPET", name="LSTM", config=best_params, group="model_comparison") as run:
        lstm_model = LSTMModel(X_bxb_train.shape[2], best_params['lstm_hidden_size'], 1)
        lstm_model.apply(init_weights)
        
        lstm_train_dataset = TensorDataset(X_bxb_train_tensor, y_train_tensor)
        lstm_train_loader = DataLoader(lstm_train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        lstm_val_dataset = TensorDataset(X_bxb_val_tensor, y_val_tensor)
        lstm_val_loader = DataLoader(lstm_val_dataset, batch_size=best_params['batch_size'], shuffle=False)
        lstm_test_dataset = TensorDataset(X_bxb_test_tensor, y_test_tensor)
        lstm_test_loader = DataLoader(lstm_test_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        lstm_optimizer = optim.AdamW(lstm_model.parameters(), lr=best_params['learning_rate'], weight_decay=1e-5)
        lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(lstm_optimizer, 'min', patience=3, factor=0.1)
        
        lstm_best_val_loss, lstm_val_accuracy, lstm_steps = train_and_evaluate(
            lstm_model, lstm_train_loader, lstm_val_loader, criterion, lstm_optimizer, lstm_scheduler, 
            num_epochs=best_params['num_epochs'], model_name="LSTM"
        )
        
        lstm_metrics, lstm_preds, lstm_labels = evaluate_model(lstm_model, lstm_test_loader, criterion)
        plot_confusion_matrix(lstm_labels, lstm_preds, "LSTM")
        wandb.log({"LSTM/test_metrics": lstm_metrics})
    
    # DNN-only Model
    with wandb.init(project="CPET", name="DNN", config=best_params, group="model_comparison") as run:
        dnn_model = DNNModel(X_cat_train.shape[1], best_params['dnn_hidden_sizes'], 1)
        dnn_model.apply(init_weights)
        
        dnn_train_dataset = TensorDataset(X_cat_train_tensor, y_train_tensor)
        dnn_train_loader = DataLoader(dnn_train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        dnn_val_dataset = TensorDataset(X_cat_val_tensor, y_val_tensor)
        dnn_val_loader = DataLoader(dnn_val_dataset, batch_size=best_params['batch_size'], shuffle=False)
        dnn_test_dataset = TensorDataset(X_cat_test_tensor, y_test_tensor)
        dnn_test_loader = DataLoader(dnn_test_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        dnn_optimizer = optim.AdamW(dnn_model.parameters(), lr=best_params['learning_rate'], weight_decay=1e-5)
        dnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(dnn_optimizer, 'min', patience=3, factor=0.1)
        
        dnn_best_val_loss, dnn_val_accuracy, dnn_steps = train_and_evaluate(
            dnn_model, dnn_train_loader, dnn_val_loader, criterion, dnn_optimizer, dnn_scheduler, 
            num_epochs=best_params['num_epochs'], model_name="DNN"
        )
        
        dnn_metrics, dnn_preds, dnn_labels = evaluate_model(dnn_model, dnn_test_loader, criterion)
        plot_confusion_matrix(dnn_labels, dnn_preds, "DNN")
        wandb.log({"DNN/test_metrics": dnn_metrics})
    
    # CNN Model
    with wandb.init(project="CPET", name="CNN", config=best_params, group="model_comparison") as run:
        cnn_model = CNNModel(X_bxb_train.shape[2], X_bxb_train.shape[1], 1)
        cnn_model.apply(init_weights)
        
        cnn_train_dataset = TensorDataset(X_bxb_train_tensor, y_train_tensor)
        cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        cnn_val_dataset = TensorDataset(X_bxb_val_tensor, y_val_tensor)
        cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=best_params['batch_size'], shuffle=False)
        cnn_test_dataset = TensorDataset(X_bxb_test_tensor, y_test_tensor)
        cnn_test_loader = DataLoader(cnn_test_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        cnn_optimizer = optim.AdamW(cnn_model.parameters(), lr=best_params['learning_rate'], weight_decay=1e-5)
        cnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, 'min', patience=3, factor=0.1)
        
        cnn_best_val_loss, cnn_val_accuracy, cnn_steps = train_and_evaluate(
            cnn_model, cnn_train_loader, cnn_val_loader, criterion, cnn_optimizer, cnn_scheduler, 
            num_epochs=best_params['num_epochs'], model_name="CNN"
        )
        
        cnn_metrics, cnn_preds, cnn_labels = evaluate_model(cnn_model, cnn_test_loader, criterion)
        plot_confusion_matrix(cnn_labels, cnn_preds, "CNN")
        wandb.log({"CNN/test_metrics": cnn_metrics})

    # Combined CNN+DNN Model
    with wandb.init(project="CPET", name="Combined CNN+DNN", config=best_params, group="model_comparison") as run:
        combined_cnn_model = CombinedCNNDnnModel(X_bxb_train.shape[2], X_bxb_train.shape[1], X_cat_train.shape[1], best_params['dnn_hidden_sizes'], 1)
        combined_cnn_model.apply(init_weights)
        
        combined_cnn_train_dataset = TensorDataset(X_bxb_train_tensor, X_cat_train_tensor, y_train_tensor)
        combined_cnn_train_loader = DataLoader(combined_cnn_train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        combined_cnn_val_dataset = TensorDataset(X_bxb_val_tensor, X_cat_val_tensor, y_val_tensor)
        combined_cnn_val_loader = DataLoader(combined_cnn_val_dataset, batch_size=best_params['batch_size'], shuffle=False)
        combined_cnn_test_dataset = TensorDataset(X_bxb_test_tensor, X_cat_test_tensor, y_test_tensor)
        combined_cnn_test_loader = DataLoader(combined_cnn_test_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        combined_cnn_optimizer = optim.AdamW(combined_cnn_model.parameters(), lr=best_params['learning_rate'], weight_decay=1e-5)
        combined_cnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(combined_cnn_optimizer, 'min', patience=3, factor=0.1)
        
        combined_cnn_best_val_loss, combined_cnn_val_accuracy, combined_cnn_steps = train_and_evaluate(
            combined_cnn_model, combined_cnn_train_loader, combined_cnn_val_loader, criterion, combined_cnn_optimizer, combined_cnn_scheduler, 
            num_epochs=best_params['num_epochs'], model_name="Combined CNN+DNN"
        )
        
        combined_cnn_metrics, combined_cnn_preds, combined_cnn_labels = evaluate_model(combined_cnn_model, combined_cnn_test_loader, criterion)
        plot_confusion_matrix(combined_cnn_labels, combined_cnn_preds, "Combined CNN+DNN")
        wandb.log({"Combined CNN+DNN/test_metrics": combined_cnn_metrics})
    
 
        metrics_dict = {
            "Combined LSTM+DNN": combined_metrics,
            "Combined CNN+DNN": combined_cnn_metrics,
            "LSTM": lstm_metrics,
            "DNN": dnn_metrics,
            "CNN": cnn_metrics
        }

        print(f"Model Comparison:{metrics_dict}")

        # Log the compiled metrics to wandb
        wandb.log({"model_comparison": metrics_dict})



if __name__ == "__main__":
    main()



