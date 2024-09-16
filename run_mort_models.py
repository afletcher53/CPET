import os
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.models import (
    CNNModel,
    CombinedCNNDnnModel,
    CombinedDNNModel,
    CombinedLSTMDnnModel,
    DNNModel,
    DNNTimeseriesModel,
    LSTMModel,
    SuperLearner,
    create_timeseries_model_wrapper,
    init_weights,
    load_and_preprocess_data,
    set_random_seeds,
)
import pandas as pd

from plotting_functions import (
    plot_roc_curves,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                X_bxb_test, X_cat_test, y_test = batch
                outputs = model(X_bxb_test, X_cat_test)
            else:
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
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )
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
        "mcc": mcc,
    }

    return metrics, all_preds, all_labels, all_probs


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=50,
    model_name="",
):
    epoch_results = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if len(batch) == 3:
                X_bxb_batch, X_cat_batch, y_batch = batch
                outputs = model(X_bxb_batch, X_cat_batch)
            else:
                X_batch, y_batch = batch
                outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == y_batch).float().sum().item()
            total += y_batch.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    X_bxb_val, X_cat_val, y_val = batch
                    val_outputs = model(X_bxb_val, X_cat_val)
                else:
                    X_val, y_val = batch
                    val_outputs = model(X_val)
                val_loss += criterion(val_outputs, y_val).item()
                val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
                val_correct += (val_preds == y_val).float().sum().item()
                val_total += y_val.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        scheduler.step(avg_val_loss)

        epoch_results.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        print(
            f"{model_name} - Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

    return epoch_results


class CombinedSuperLearnerCNNModel:
    def __init__(self, superlearner, cnn_model):
        self.superlearner = superlearner
        self.cnn_model = cnn_model
        self.meta_model = LogisticRegression()

    def fit(self, X_cat, X_bxb, y):
        # Get predictions from SuperLearner
        superlearner_probs = self.superlearner.predict_proba(X_cat)[:, 1]

        # Get predictions from CNN
        self.cnn_model.eval()
        with torch.no_grad():
            cnn_probs = (
                torch.sigmoid(self.cnn_model(torch.FloatTensor(X_bxb)))
                .numpy()
                .flatten()
            )

        # Combine predictions
        X_combined = np.column_stack((superlearner_probs, cnn_probs))

        # Train meta-model
        self.meta_model.fit(X_combined, y)

    def predict_proba(self, X_cat, X_bxb):
        # Get predictions from SuperLearner
        superlearner_probs = self.superlearner.predict_proba(X_cat)[:, 1]

        # Get predictions from CNN
        self.cnn_model.eval()
        with torch.no_grad():
            cnn_probs = (
                torch.sigmoid(self.cnn_model(torch.FloatTensor(X_bxb)))
                .numpy()
                .flatten()
            )

        # Combine predictions
        X_combined = np.column_stack((superlearner_probs, cnn_probs))

        # Get final predictions
        return self.meta_model.predict_proba(X_combined)[:, 1]

    def evaluate(self, X_cat_test, X_bxb_test, y_test):
        probs = self.predict_proba(X_cat_test, X_bxb_test)
        preds = (probs > 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary"
        )

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }


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
        worker_init_fn=lambda _: set_random_seeds(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=best_params["batch_size"], shuffle=False
    )

    return model, train_loader, val_loader


def main():
    set_random_seeds()
    days = [365, 180, 90, 30]

    for day in days:
        directory = f"data/ml_inputs/mortality_{day}"
        if not os.path.exists(directory):
            raise FileNotFoundError(
                f"Directory {directory} does not exist. Please run the data preparation script first."
            )
        if not os.path.exists(f"{directory}/X_bxb_train.npy"):
            raise FileNotFoundError(
                f"Data files for {directory} do not exist. Please run the data preparation script first."
            )

        if not os.path.exists(f"saved_models/mortality/{day}"):
            os.makedirs(f"saved_models/mortality/{day}")
    for day in days:
        directory = f"data/ml_inputs/mortality_{day}"
        (
            X_bxb_train,
            X_cat_train,
            y_train,
            X_bxb_val,
            X_cat_val,
            y_val,
            X_bxb_test,
            X_cat_test,
            y_test,
        ) = load_and_preprocess_data(directory=directory)

        X_bxb_train_tensor = torch.FloatTensor(X_bxb_train)
        X_cat_train_tensor = torch.FloatTensor(X_cat_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_bxb_val_tensor = torch.FloatTensor(X_bxb_val)
        X_cat_val_tensor = torch.FloatTensor(X_cat_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        X_bxb_test_tensor = torch.FloatTensor(X_bxb_test)
        X_cat_test_tensor = torch.FloatTensor(X_cat_test)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

        best_params = {
            "batch_size": 512,
            "dnn_hidden_sizes": [64, 32],
            "learning_rate": 1e-03,
            "lstm_hidden_size": 16,
            "num_epochs": 20,
        }

        ####################
        # DATA PREPARATION #
        ####################

        combined_train_dataset = TensorDataset(
            X_bxb_train_tensor, X_cat_train_tensor, y_train_tensor
        )
        combined_train_loader = DataLoader(
            combined_train_dataset,
            batch_size=best_params["batch_size"],
            shuffle=True,
            worker_init_fn=lambda _: set_random_seeds(),
        )
        combined_val_dataset = TensorDataset(
            X_bxb_val_tensor, X_cat_val_tensor, y_val_tensor
        )
        combined_val_loader = DataLoader(
            combined_val_dataset, batch_size=best_params["batch_size"], shuffle=False
        )

        combined_test_dataset = TensorDataset(
            X_bxb_test_tensor, X_cat_test_tensor, y_test_tensor
        )
        combined_test_loader = DataLoader(
            combined_test_dataset, batch_size=best_params["batch_size"], shuffle=False
        )

        timeseries_train_dataset = TensorDataset(X_bxb_train_tensor, y_train_tensor)
        timeseries_train_loader = DataLoader(
            timeseries_train_dataset,
            batch_size=best_params["batch_size"],
            shuffle=True,
            worker_init_fn=lambda _: set_random_seeds(),
        )
        timeseries_val_dataset = TensorDataset(X_bxb_val_tensor, y_val_tensor)
        timeseries_val_loader = DataLoader(
            timeseries_val_dataset, batch_size=best_params["batch_size"], shuffle=False
        )

        timeseries_test_dataset = TensorDataset(X_bxb_test_tensor, y_test_tensor)
        timeseries_test_loader = DataLoader(
            timeseries_test_dataset, batch_size=best_params["batch_size"], shuffle=False
        )

        static_train_dataset = TensorDataset(X_cat_train_tensor, y_train_tensor)
        static_train_loader = DataLoader(
            static_train_dataset,
            batch_size=best_params["batch_size"],
            shuffle=True,
            worker_init_fn=lambda _: set_random_seeds(),
        )
        static_val_dataset = TensorDataset(X_cat_val_tensor, y_val_tensor)
        static_val_loader = DataLoader(
            static_val_dataset, batch_size=best_params["batch_size"], shuffle=False
        )
        static_test_dataset = TensorDataset(X_cat_test_tensor, y_test_tensor)
        static_test_loader = DataLoader(
            static_test_dataset, batch_size=best_params["batch_size"], shuffle=False
        )

        total_samples = len(y_train_tensor)
        positive_samples = y_train_tensor.sum().item()
        negative_samples = total_samples - positive_samples

        positive_ratio = positive_samples / total_samples
        negative_ratio = negative_samples / total_samples

        print(f"Positive samples: {positive_samples} ({positive_ratio:.2%})")
        print(f"Negative samples: {negative_samples} ({negative_ratio:.2%})")

        # Set alpha for Focal Loss
        alpha = 1 - positive_ratio
        print(f"Suggested alpha value for focal loss: {alpha:.4f}")

        ####################
        # MODEL TRAINING   #
        ####################

        combined_lstm_dnn_model = CombinedLSTMDnnModel(
            X_bxb_train.shape[2],
            X_cat_train.shape[1],
            best_params["lstm_hidden_size"],
            best_params["dnn_hidden_sizes"],
            1,
        )
        combined_lstm_dnn_model.apply(init_weights)

        # criterion = nn.BCEWithLogitsLoss()
        criterion = FocalLoss(alpha=alpha, gamma=2)
        optimizer = optim.AdamW(
            combined_lstm_dnn_model.parameters(),
            lr=best_params["learning_rate"],
            weight_decay=1e-5,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=3, factor=0.1
        )

        combined_lstm_dnn_results = train_model(
            combined_lstm_dnn_model,
            combined_train_loader,
            combined_val_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs=best_params["num_epochs"],
            model_name="Combined LSTM+DNN",
        )

        lstm_model = LSTMModel(X_bxb_train.shape[2], best_params["lstm_hidden_size"], 1)
        lstm_model.apply(init_weights)

        lstm_optimizer = optim.AdamW(
            lstm_model.parameters(), lr=best_params["learning_rate"], weight_decay=1e-5
        )
        lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            lstm_optimizer, "min", patience=3, factor=0.1
        )

        lstm_results = train_model(
            lstm_model,
            timeseries_train_loader,
            timeseries_val_loader,
            criterion,
            lstm_optimizer,
            lstm_scheduler,
            num_epochs=best_params["num_epochs"],
            model_name="LSTM",
        )

        dnn_model = DNNModel(X_cat_train.shape[1], best_params["dnn_hidden_sizes"], 1)
        dnn_model.apply(init_weights)

        dnn_optimizer = optim.AdamW(
            dnn_model.parameters(), lr=best_params["learning_rate"], weight_decay=1e-5
        )
        dnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            dnn_optimizer, "min", patience=3, factor=0.1
        )

        dnn_results = train_model(
            dnn_model,
            static_train_loader,
            static_val_loader,
            criterion,
            dnn_optimizer,
            dnn_scheduler,
            num_epochs=best_params["num_epochs"],
            model_name="DNN",
        )

        cnn_model = CNNModel(X_bxb_train.shape[2], X_bxb_train.shape[1], 1)
        cnn_model.apply(init_weights)

        cnn_optimizer = optim.AdamW(
            cnn_model.parameters(), lr=best_params["learning_rate"], weight_decay=1e-5
        )
        cnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            cnn_optimizer, "min", patience=3, factor=0.1
        )

        cnn_results = train_model(
            cnn_model,
            timeseries_train_loader,
            timeseries_val_loader,
            criterion,
            cnn_optimizer,
            cnn_scheduler,
            num_epochs=best_params["num_epochs"],
            model_name="CNN",
        )

        combined_cnn_model = CombinedCNNDnnModel(
            X_bxb_train.shape[2],
            X_bxb_train.shape[1],
            X_cat_train.shape[1],
            best_params["dnn_hidden_sizes"],
            1,
        )
        combined_cnn_model.apply(init_weights)

        combined_cnn_optimizer = optim.AdamW(
            combined_cnn_model.parameters(),
            lr=best_params["learning_rate"],
            weight_decay=1e-5,
        )
        combined_cnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            combined_cnn_optimizer, "min", patience=3, factor=0.1
        )

        combined_cnn_dnn_results = train_model(
            combined_cnn_model,
            combined_train_loader,
            combined_val_loader,
            criterion,
            combined_cnn_optimizer,
            combined_cnn_scheduler,
            num_epochs=best_params["num_epochs"],
            model_name="Combined CNN+DNN",
        )
        dnn_timeseries_model = DNNTimeseriesModel(
            X_bxb_train.shape[1] * X_bxb_train.shape[2],
            best_params["dnn_hidden_sizes"],
            1,
        )
        dnn_timeseries_model.apply(init_weights)

        dnn_timeseries_optimizer = optim.AdamW(
            dnn_timeseries_model.parameters(),
            lr=best_params["learning_rate"],
            weight_decay=1e-5,
        )
        dnn_timeseries_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            dnn_timeseries_optimizer, "min", patience=3, factor=0.1
        )

        dnn_timeseries_results = train_model(
            dnn_timeseries_model,
            timeseries_train_loader,
            timeseries_val_loader,
            criterion,
            dnn_timeseries_optimizer,
            dnn_timeseries_scheduler,
            num_epochs=best_params["num_epochs"],
            model_name="DNN Timeseries",
        )

        combined_dnn_model = CombinedDNNModel(
            static_input_size=X_cat_train.shape[1],
            timeseries_input_size=X_bxb_train.shape[1] * X_bxb_train.shape[2],
            hidden_sizes=best_params["dnn_hidden_sizes"],
            output_size=1
        )
        combined_dnn_model.apply(init_weights)

        combined_dnn_optimizer = optim.AdamW(
            combined_dnn_model.parameters(),
            lr=best_params["learning_rate"],
            weight_decay=1e-5,
        )
        combined_dnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            combined_dnn_optimizer, "min", patience=3, factor=0.1
        )

        combined_dnn_results = train_model(
            combined_dnn_model,
            combined_train_loader,
            combined_val_loader,
            criterion,
            combined_dnn_optimizer,
            combined_dnn_scheduler,
            num_epochs=best_params["num_epochs"],
            model_name="Combined DNN",
        )

        #################
        # Superlearners #
        #################

        # Create base models
        base_models = [
            make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
            make_pipeline(
                StandardScaler(), GradientBoostingClassifier(random_state=42)
            ),
            make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
            make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)),
        ]

        # Create SuperLearner
        superlearner = SuperLearner(base_models, meta_model=LogisticRegression())
        superlearner.fit(X_cat_train, y_train)

        # Wrapping your existing LSTM and CNN models
        lstm_wrapper = create_timeseries_model_wrapper(
            LSTMModel, input_size=X_bxb_train.shape[2], hidden_size=16, output_size=1
        )
        # For CNNModel, pass input_channels and seq_length instead of input_size and hidden_size
        cnn_wrapper = create_timeseries_model_wrapper(
            CNNModel,
            input_channels=X_bxb_train.shape[
                2
            ],  # Channels typically correspond to features in timeseries data
            seq_length=X_bxb_train.shape[
                1
            ],  # Sequence length corresponds to the time dimension
            output_size=1,  # Binary classification
        )

        # Define the SuperLearner with timeseries models only
        base_models = [lstm_wrapper, cnn_wrapper]

        # Create SuperLearner using Logistic Regression as the meta-model
        supertimelearner = SuperLearner(
            base_models=base_models, meta_model=LogisticRegression()
        )

        # Fit the SuperLearner on your timeseries data
        supertimelearner.fit(X_bxb_train, y_train)

        ####################
        # MODEL EVALUATION #
        ####################
        df = pd.DataFrame(combined_lstm_dnn_results)
        df["model"] = "Combined LSTM+DNN"
        df2 = pd.DataFrame(lstm_results)
        df2["model"] = "LSTM"
        df3 = pd.DataFrame(dnn_results)
        df3["model"] = "DNN"
        df4 = pd.DataFrame(cnn_results)
        df4["model"] = "CNN"
        df5 = pd.DataFrame(combined_cnn_dnn_results)
        df5["model"] = "Combined CNN+DNN"
        df6 = pd.DataFrame(dnn_timeseries_results)
        df6["model"] = "DNN Timeseries"
        df7 = pd.DataFrame(combined_dnn_results)
        df7["model"] = "Combined DNN"
        df = pd.concat([df, df2, df3, df4, df5, df6, df7])


        df.to_csv(f"saved_models/mortality/{day}/train_log.csv", index=False)
        print("Model results saved to model_results.csv")

        torch.save(
            combined_lstm_dnn_model.state_dict(),
            f"saved_models/mortality/{day}/combined_lstm_dnn_model.pth",
        )
        torch.save(
            lstm_model.state_dict(), f"saved_models/mortality/{day}/lstm_model.pth"
        )
        torch.save(
            dnn_model.state_dict(), f"saved_models/mortality/{day}/dnn_model.pth"
        )
        torch.save(
            cnn_model.state_dict(), f"saved_models/mortality/{day}/cnn_model.pth"
        )
        torch.save(
            combined_cnn_model.state_dict(),
            f"saved_models/mortality/{day}/combined_cnn_model.pth",
        )

        torch.save(
            dnn_timeseries_model.state_dict(),
            f"saved_models/mortality/{day}/dnn_timeseries_model.pth",
        )

        torch.save(
            combined_dnn_model.state_dict(), f"saved_models/mortality/{day}/combined_dnn_model.pth"
        )

        print("Models saved")

        all_labels = []
        all_probs = []
        model_names = [
            "Combined LSTM+DNN",
            "LSTM",
            "DNN",
            "CNN",
            "Combined CNN+DNN",
            "SuperLearner",
            "SuperTimeLearner",
            "DNN Timeseries",
            "Combined DNN",
        ]

        for model_name in model_names:
            if model_name == "Combined LSTM+DNN":
                metrics, preds, labels, probs = evaluate_model(
                    combined_lstm_dnn_model, combined_test_loader, criterion
                )
            elif model_name == "LSTM":
                metrics, preds, labels, probs = evaluate_model(
                    lstm_model, timeseries_test_loader, criterion
                )
            elif model_name == "DNN":
                metrics, preds, labels, probs = evaluate_model(
                    dnn_model, static_test_loader, criterion
                )
            elif model_name == "CNN":
                metrics, preds, labels, probs = evaluate_model(
                    cnn_model, timeseries_test_loader, criterion
                )
            elif model_name == "Combined CNN+DNN":
                metrics, preds, labels, probs = evaluate_model(
                    combined_cnn_model, combined_test_loader, criterion
                )
            elif model_name == "SuperLearner":
                probs = superlearner.predict_proba(X_cat_test)[:, 1]
                preds = (probs > 0.5).astype(int)
                labels = y_test
            elif model_name == "SuperTimeLearner":
                probs = supertimelearner.predict_proba(X_bxb_test)[:, 1]
                preds = (probs > 0.5).astype(int)
                labels = y_test
            elif model_name == "DNN Timeseries":
                metrics, preds, labels, probs = evaluate_model(
                    dnn_timeseries_model, timeseries_test_loader, criterion
                )

            elif model_name == "Combined DNN":
                metrics, preds, labels, probs = evaluate_model(
                    combined_dnn_model, combined_test_loader, criterion
                )

            all_labels.append(labels)
            all_probs.append(probs)

        # lets create a model that uses the pretrained CNN + Superlearner models to make predictions on the test set

        # Usage in main function:
        combined_model = CombinedSuperLearnerCNNModel(superlearner, cnn_model)
        combined_model.fit(X_cat_train, X_bxb_train, y_train)
       
        all_probs.append(combined_model.predict_proba(X_cat_test, X_bxb_test))
        all_labels.append(y_test)
        model_names.append("Combined SuperLearner+CNN")


        plot_roc_curves(
            all_labels, all_probs, model_names, f"saved_models/mortality/{day}"
        )


if __name__ == "__main__":
    main()
