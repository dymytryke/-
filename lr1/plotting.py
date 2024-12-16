import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score


# Функції для побудови кривих
def plot_roc(model_name, pred_values_prob, true_values):
    roc_auc = roc_auc_score(true_values, pred_values_prob)
    false_positive_rates, true_positive_rates, thresholds = roc_curve(true_values, pred_values_prob)
    plt.plot(false_positive_rates, true_positive_rates, marker='.', label=f'ROC AUC = {roc_auc:.2f}')
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.show()


def plot_precision_recall(model_name, predicted_probability, true_values):
    precision, recall, thresholds = precision_recall_curve(true_values, predicted_probability)
    plt.plot(thresholds, precision[1:], color='blue', label='Precision')
    plt.plot(thresholds, recall[1:], color='orange', label='Recall')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.legend()
    plt.show()
