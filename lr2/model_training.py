import torch
from metrics import calculate_model_accuracy


def train_model(model, loss_fn, optimizer, train_loader, test_loader, num_epochs=20):
    """
    Тренування моделі з фіксацією втрат для кожної епохи.
    """
    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_losses.append(total_train_loss / len(train_loader))

        # Оцінка втрат на тестових даних
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs.squeeze(), labels.float())
                total_test_loss += loss.item()

        test_losses.append(total_test_loss / len(test_loader))

    return train_losses, test_losses


def train_and_evaluate_models(models, optimizers, loss_functions, train_loader, test_loader, num_epochs=20):
    """
    Тренування і оцінка кількох моделей.
    """
    loss_histories = {}
    accuracies = {}

    for name, model in models.items():
        print(f'Тренування моделі {name}...')

        train_loss, test_loss = train_model(
            model, loss_functions[name], optimizers[name], train_loader, test_loader, num_epochs
        )
        loss_histories[name] = (train_loss, test_loss)

        accuracy = calculate_model_accuracy(model, test_loader)
        accuracies[name] = accuracy
        print(f'Точність моделі {name}: {accuracy:.4f}')

    return loss_histories, accuracies
