from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, average_precision_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred, model_name, dir_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{dir_path}/{model_name}_confusion_matrix.png')
    plt.close()
    return f'{model_name}_confusion_matrix.png'

def plot_roc_curve(all_labels, all_probs, model_names, dir_path):
    plt.figure(figsize=(10, 8))
    for i, model_name in enumerate(model_names):
        fpr, tpr, _ = roc_curve(all_labels[i], all_probs[i])
        roc_auc = roc_auc_score(all_labels[i], all_probs[i])
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{dir_path}/roc_curve_comparison.png')
    plt.close()
    return 'roc_curve_comparison.png'
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score



# def plot_pr_curve(all_labels, all_probs, model_names, dir_path):
#     plt.figure(figsize=(10, 8))
#     for i, model_name in enumerate(model_names):
#         precision, recall, _ = precision_recall_curve(all_labels[i], all_probs[i])
#         pr_auc = average_precision_score(all_labels[i], all_probs[i])
#         plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.2f})')
    
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend(loc="lower right")
#     plt.savefig(f'{dir_path}/pr_curve_comparison.png')
#     plt.close()
#     return 'pr_curve_comparison.png'

# def plot_pr_curves(all_labels, all_probs, model_names, dir_path):
#     # Categorize models
#     static_models = ['DNN', 'SuperLearner']
#     timeseries_models = ['LSTM', 'CNN', 'SuperTimeLearner', 'DNN Timeseries']
#     combined_models = ['Combined LSTM+DNN', 'Combined CNN+DNN', 'Combined SuperLearner+CNN', 'Combined DNN']

#     categories = [
#         ('Static', static_models),
#         ('Timeseries', timeseries_models),
#         ('Combined', combined_models)
#     ]

#     best_models = {}

#     # Plot PR curve for each category
#     for category, models in categories:
#         plt.figure(figsize=(10, 8))
#         best_auc = 0
#         best_model = ''

#         for model in models:
#             if model in model_names:
#                 i = model_names.index(model)
#                 precision, recall, _ = precision_recall_curve(all_labels[i], all_probs[i])
#                 pr_auc = average_precision_score(all_labels[i], all_probs[i])
#                 plt.plot(recall, precision, label=f'{model} (AUC = {pr_auc:.2f})')

#                 if pr_auc > best_auc:
#                     best_auc = pr_auc
#                     best_model = model

#         best_models[category] = (best_model, best_auc)

#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel('Recall')
#         plt.ylabel('Precision')
#         plt.title(f'Precision-Recall Curve - {category} Models')
#         plt.legend(loc="lower right")
#         plt.savefig(f'{dir_path}/pr_curve_{category.lower()}.png')
#         plt.close()

#     # Plot best models from each category
#     plt.figure(figsize=(10, 8))
#     for category, (model, auc) in best_models.items():
#         i = model_names.index(model)
#         precision, recall, _ = precision_recall_curve(all_labels[i], all_probs[i])
#         plt.plot(recall, precision, label=f'{category}: {model} (AUC = {auc:.2f})')

#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve - Best Models Comparison')
#     plt.legend(loc="lower right")
#     plt.savefig(f'{dir_path}/pr_curve_best_models.png')
#     plt.close()

#     return ['pr_curve_static.png', 'pr_curve_timeseries.png', 'pr_curve_combined.png', 'pr_curve_best_models.png']


def plot_roc_curves(all_labels, all_probs, model_names, dir_path):
    # Categorize models
    static_models = ['DNN', 'SuperLearner']
    timeseries_models = ['LSTM', 'CNN', 'SuperTimeLearner', 'DNN Timeseries']
    combined_models = ['Combined LSTM+DNN', 'Combined CNN+DNN', 'Combined SuperLearner+CNN', 'Combined DNN']
    
    categories = [
        ('Static', static_models),
        ('Timeseries', timeseries_models),
        ('Combined', combined_models)
    ]
    
    best_models = {}
    
    # Plot ROC curve for each category
    for category, models in categories:
        plt.figure(figsize=(10, 8))
        best_auc = 0
        best_model = ''
        
        for model in models:
            if model in model_names:
                i = model_names.index(model)
                fpr, tpr, _ = roc_curve(all_labels[i], all_probs[i])
                roc_auc = roc_auc_score(all_labels[i], all_probs[i])
                plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')
                
                if roc_auc > best_auc:
                    best_auc = roc_auc
                    best_model = model
        
        best_models[category] = (best_model, best_auc)
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {category} Models')
        plt.legend(loc="lower right")
        plt.savefig(f'{dir_path}/roc_curve_{category.lower()}.png')
        plt.close()
    
    # Plot best models from each category
    plt.figure(figsize=(10, 8))
    for category, (model, auc) in best_models.items():
        i = model_names.index(model)
        fpr, tpr, _ = roc_curve(all_labels[i], all_probs[i])
        plt.plot(fpr, tpr, label=f'{category}: {model} (AUC = {auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Best Models Comparison')
    plt.legend(loc="lower right")
    plt.savefig(f'{dir_path}/roc_curve_best_models.png')
    plt.close()
    
    return ['roc_curve_static.png', 'roc_curve_timeseries.png', 'roc_curve_combined.png', 'roc_curve_best_models.png']
