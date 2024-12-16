import math


# Функції для обчислення метрик
def calc_accuracy(pred_values, true_values):
    number_of_predictions = len(pred_values)
    correct_predictions_count = 0
    for i in range(len(pred_values)):
        if pred_values[i] == true_values[i]:  # Перевіряємо чи прогноз відповідаємо істинному значенню
            correct_predictions_count += 1
    return correct_predictions_count / number_of_predictions  # Розрахуємо частку правильних відповідей


def calc_precision(pred_values, true_values):
    true_positives_count = 0
    false_positives_count = 0
    for i in range(len(pred_values)):
        if pred_values[i] == 1 and true_values[i] == 1:  # True positive
            true_positives_count += 1
        elif pred_values[i] == 1 and true_values[i] == 0:  # False positive
            false_positives_count += 1
    return true_positives_count / (true_positives_count + false_positives_count)  # Обчислення точності моделі


def calc_recall(pred_values, true_values):
    true_positives_count = 0
    false_negatives_count = 0
    for i in range(len(pred_values)):
        if pred_values[i] == 1 and true_values[i] == 1:  # True positive
            true_positives_count += 1
        elif pred_values[i] == 0 and true_values[i] == 1:  # False negative
            false_negatives_count += 1
    return true_positives_count / (true_positives_count + false_negatives_count)  # Обчислення повноти моделі


def calc_f1_score(pred_values, true_values):
    precision = calc_precision(pred_values, true_values)
    recall = calc_recall(pred_values, true_values)
    return 2 * precision * recall / (precision + recall)  # Обчислення F1-score моделі


def calc_log_loss(pred_values_prob, true_values):
    all_values_count = len(pred_values_prob)
    log_sum = 0

    for i in range(len(pred_values_prob)):
        predicted_probability = pred_values_prob[i]
        real = true_values[i]
        # Додано невелику костанту для уникнення обчилення логарифму від 0
        addition = real * math.log(predicted_probability + 1e-9) + \
                   (1 - real) * math.log(1 - predicted_probability + 1e-9)
        log_sum += addition

    return -1 / all_values_count * log_sum  # Обчислення log-loss моделі
