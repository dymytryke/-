from metrics_calc import calc_accuracy, calc_precision, calc_recall, calc_f1_score, calc_log_loss
from plotting import plot_roc, plot_precision_recall


# Допоміжна функція для виводу метрик в термінал
def print_metrics_to_terminal(pred_values, pred_values_prob, true_values):
    print("    Model metrics")
    print(f"Accuracy: {calc_accuracy(pred_values, true_values)}")
    print(f"Precision: {calc_precision(pred_values, true_values)}")
    print(f"Recall: {calc_recall(pred_values, true_values)}")
    print(f"F1 score: {calc_f1_score(pred_values, true_values)}")
    print(f"Log loss: {calc_log_loss(pred_values_prob, true_values)}")


# Допоміжна функція для навчання, оцінки та візуалізації роботи моделі.
def train_eval_visualize_model(model, model_name, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)
    y_pred_class_of_interest_probabilities = y_pred_prob[:, 1]

    print(f"-----{model_name}-----")
    print_metrics_to_terminal(y_pred, y_pred_class_of_interest_probabilities, y_test.values)
    plot_precision_recall(model_name, y_pred_class_of_interest_probabilities, y_test.values)
    plot_roc(model_name, y_pred_class_of_interest_probabilities, y_test.values)
    print("\n")
