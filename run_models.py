import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import confusion_matrix,  precision_recall_fscore_support, roc_auc_score, average_precision_score

from models.models import CNNModel, CombinedCNNDnnModel, CombinedLSTMDnnModel, DNNModel, LSTMModel, init_weights, load_and_preprocess_data, set_random_seeds
from plotting_functions import plot_confusion_matrix, plot_pr_curve, plot_roc_curve

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

    return metrics, all_preds, all_labels, all_probs


def main():
    set_random_seeds()
    directory = 'data/ml_inputs/mortality_90'
    X_bxb_train, X_cat_train, y_train, X_bxb_val, X_cat_val, y_val, X_bxb_test, X_cat_test, y_test = load_and_preprocess_data(directory=directory)
    
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
    

    combined_model = CombinedLSTMDnnModel(X_bxb_train.shape[2], X_cat_train.shape[1], best_params['lstm_hidden_size'], best_params['dnn_hidden_sizes'], 1)
    combined_model.apply(init_weights)
        
    combined_train_dataset = TensorDataset(X_bxb_train_tensor, X_cat_train_tensor, y_train_tensor)
    combined_train_loader = DataLoader(combined_train_dataset, batch_size=best_params['batch_size'], shuffle=True, worker_init_fn=lambda _: set_random_seeds())
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
      

    lstm_model = LSTMModel(X_bxb_train.shape[2], best_params['lstm_hidden_size'], 1)
    lstm_model.apply(init_weights)
        
    lstm_train_dataset = TensorDataset(X_bxb_train_tensor, y_train_tensor)
    lstm_train_loader = DataLoader(lstm_train_dataset, batch_size=best_params['batch_size'], shuffle=True, worker_init_fn=lambda _: set_random_seeds())
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
        
    
    # DNN-only Model

    dnn_model = DNNModel(X_cat_train.shape[1], best_params['dnn_hidden_sizes'], 1)
    dnn_model.apply(init_weights)
        
    dnn_train_dataset = TensorDataset(X_cat_train_tensor, y_train_tensor)
    dnn_train_loader = DataLoader(dnn_train_dataset, batch_size=best_params['batch_size'], shuffle=True, worker_init_fn=lambda _: set_random_seeds())
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
        
    
    # CNN Model

    cnn_model = CNNModel(X_bxb_train.shape[2], X_bxb_train.shape[1], 1)
    cnn_model.apply(init_weights)
        
    cnn_train_dataset = TensorDataset(X_bxb_train_tensor, y_train_tensor)
    cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=best_params['batch_size'], shuffle=True, worker_init_fn=lambda _: set_random_seeds())
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
        
    

    combined_cnn_model = CombinedCNNDnnModel(X_bxb_train.shape[2], X_bxb_train.shape[1], X_cat_train.shape[1], best_params['dnn_hidden_sizes'], 1)
    combined_cnn_model.apply(init_weights)
        
    combined_cnn_train_dataset = TensorDataset(X_bxb_train_tensor, X_cat_train_tensor, y_train_tensor)
    combined_cnn_train_loader = DataLoader(combined_cnn_train_dataset, batch_size=best_params['batch_size'], shuffle=True, worker_init_fn=lambda _: set_random_seeds())
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
        
    
    # Model Comparison
    all_labels = []
    all_probs = []
    model_names = ["Combined LSTM+DNN", "LSTM", "DNN", "CNN", "Combined CNN+DNN"]

    for model_name in model_names:
        if model_name == "Combined LSTM+DNN":
            metrics, preds, labels, probs = evaluate_model(combined_model, combined_test_loader, criterion)
        elif model_name == "LSTM":
            metrics, preds, labels, probs = evaluate_model(lstm_model, lstm_test_loader, criterion)
        elif model_name == "DNN":
            metrics, preds, labels, probs = evaluate_model(dnn_model, dnn_test_loader, criterion)
        elif model_name == "CNN":
            metrics, preds, labels, probs = evaluate_model(cnn_model, cnn_test_loader, criterion)
        elif model_name == "Combined CNN+DNN":
            metrics, preds, labels, probs = evaluate_model(combined_cnn_model, combined_cnn_test_loader, criterion)
        
        plot_confusion_matrix(labels, preds, model_name)

        all_labels.append(labels)
        all_probs.append(probs)

    plot_roc_curve(all_labels, all_probs, model_names)
    plot_pr_curve(all_labels, all_probs, model_names)





if __name__ == "__main__":
    main()