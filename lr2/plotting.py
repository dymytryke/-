import matplotlib.pyplot as plt

# Побудова графіків втрат для тренування і тестування
def plot_loss_curves(loss_histories):
    """
    Візуалізація динаміки втрат для тренувальних і тестових даних.
    """
    plt.figure(figsize=(12, 6))
    for name, (train_loss, test_loss) in loss_histories.items():
        plt.plot(train_loss, label=f'{name} - Тренувальні втрати')

    plt.xlabel('Епохи')
    plt.ylabel('Втрати')
    plt.title('Криві тренувальних втрат')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    for name, (train_loss, test_loss) in loss_histories.items():
        plt.plot(test_loss, label=f'{name} - Тестові втрати')

    plt.xlabel('Епохи')
    plt.ylabel('Втрати')
    plt.title('Криві тестових втрат')
    plt.legend()
    plt.show()
