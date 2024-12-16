import torch
from sklearn.metrics import accuracy_score

# Обчислення точності моделі
def calculate_model_accuracy(model, test_loader):
    """
    Оцінка точності моделі за тестовим набором даних.
    """
    model.eval()  # Встановлення моделі в режим оцінки
    predictions = []  # Список для прогнозів
    targets = []  # Список для реальних міток

    with torch.no_grad():  # Вимкнення обчислення градієнтів
        for inputs, labels in test_loader:
            outputs = model(inputs)  # Передбачення
            predicted = outputs.squeeze().round()  # Перетворення ймовірностей у мітки {0, 1}
            predictions.extend(predicted.numpy())  # Збереження передбачень
            targets.extend(labels.numpy())  # Збереження міток

    return accuracy_score(targets, predictions)  # Повертає точність моделі
