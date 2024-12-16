import torch.optim as optim
from prep_data import prepare_data_loaders
from logistic_regression_model import LogisticRegressionModel, compute_logistic_loss, compute_adaboost_loss, compute_binary_crossentropy_loss
from model_training import train_and_evaluate_models
from plotting import plot_loss_curves


# Ініціалізація моделей, оптимізаторів та функцій втрат
def initialize_models(input_dim):
    """
    Повертає словники з моделями, оптимізаторами та функціями втрат.
    """
    models = {
        'logistic_loss': LogisticRegressionModel(input_dim),  # Модель для логістичних втрат
        'adaboost_loss': LogisticRegressionModel(input_dim),  # Модель для втрат AdaBoost
        'binary_crossentropy_loss': LogisticRegressionModel(input_dim),  # Модель для BCE втрат
    }

    optimizers = {
        name: optim.SGD(model.parameters(), lr=0.01) for name, model in models.items()  # Використання SGD
    }

    criteria = {
        'logistic_loss': compute_logistic_loss,
        'adaboost_loss': compute_adaboost_loss,
        'binary_crossentropy_loss': compute_binary_crossentropy_loss,
    }

    return models, optimizers, criteria


# Основна функція для запуску тренування і оцінки моделей
def main():
    # Завантаження даних
    train_loader, test_loader, input_dim = prepare_data_loaders()

    # Ініціалізація моделей
    models, optimizers, criteria = initialize_models(input_dim)

    # Тренування моделей і оцінка їх продуктивності
    loss_histories, accuracies = train_and_evaluate_models(models, optimizers, criteria, train_loader, test_loader)

    # Побудова графіків втрат
    plot_loss_curves(loss_histories)


if __name__ == '__main__':
    main()
