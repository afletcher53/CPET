from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, average_precision_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'outputs/{model_name}_confusion_matrix.png')
    plt.close()
    return f'{model_name}_confusion_matrix.png'

def plot_roc_curve(all_labels, all_probs, model_names):
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
    plt.savefig('outputs/roc_curve_comparison.png')
    plt.close()
    return 'roc_curve_comparison.png'

def plot_pr_curve(all_labels, all_probs, model_names):
    plt.figure(figsize=(10, 8))
    for i, model_name in enumerate(model_names):
        precision, recall, _ = precision_recall_curve(all_labels[i], all_probs[i])
        pr_auc = average_precision_score(all_labels[i], all_probs[i])
        plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('outputs/pr_curve_comparison.png')
    plt.close()
    return 'pr_curve_comparison.png'