import torch.nn as nn
import torch

# Клас для реалізації моделі логістичної регресії
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # Один лінійний шар для прогнозування

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Застосування сигмоїдної функції для отримання ймовірностей


# Обчислення логістичних втрат (logistic loss)
def compute_logistic_loss(output, target):
    output = torch.clamp(output, min=1e-9, max=1 - 1e-9)  # Уникнення log(0) через обмеження значень
    loss = - (target * torch.log(output) + (1 - target) * torch.log(1 - output))  # Формула логістичних втрат
    return loss.mean()  # Середнє значення втрат


# Функція для обчислення втрат AdaBoost
def compute_adaboost_loss(output, target):
    target = 2 * target - 1  # Конвертація міток у {-1, 1} для AdaBoost
    loss = torch.exp(-target * output)  # Обчислення втрат згідно з методом AdaBoost
    return loss.mean()


# Обчислення втрат за допомогою вбудованої PyTorch функції BCE
def compute_binary_crossentropy_loss(output, target):
    criterion = nn.BCELoss()  # BCE функція втрат
    return criterion(output, target)
