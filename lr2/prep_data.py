import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch

# Завантаження і обробка даних
def load_dataset():
    """
    Завантаження датасету та розділення на ознаки і мітки.
    """
    column_names = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']
    data = pd.read_csv('data_banknote_authentication.txt', header=None, names=column_names)
    X = data.drop('Class', axis=1)  # Ознаки
    y = data['Class']  # Цільова змінна
    return X, y


def split_dataset(X, y, test_ratio=0.2):
    """
    Розділення даних на тренувальний і тестовий набори.
    """
    return train_test_split(X, y, test_size=test_ratio, random_state=42)


def prepare_data_loaders():
    """
    Підготовка даталоадерів для PyTorch.
    """
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test.values, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, X_train.shape[1]
