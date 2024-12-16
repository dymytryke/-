import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from model_utils import train_eval_visualize_model


def main():
    data = pd.read_csv('bioresponse.csv')

    # Розділення набору даних на ознаки (X) та цільову змінну (y).
    x = data.iloc[:, 1:]
    y = data.Activity

    # random_state фіксує випадковість під час навчання моделі, щоб результати були відтворюваними між різними запусками
    random_state = 20

    # Розділення даних на тренувальний і тестовий набори.
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=random_state
                                                        )

    # Ініціалізація класифікаторів
    shallow_tree = DecisionTreeClassifier(max_depth=3,
                                          random_state=random_state
                                          )
    deep_tree = DecisionTreeClassifier(max_depth=15,
                                       random_state=random_state
                                       )
    shallow_forest = RandomForestClassifier(n_estimators=100,
                                            max_depth=3,
                                            random_state=random_state
                                            )
    deep_forest = RandomForestClassifier(n_estimators=100,
                                         max_depth=15,
                                         random_state=random_state
                                         )
    avoid_type_2_error_classifier = DecisionTreeClassifier(
        max_depth=5,
        class_weight={0: 1, 1: 5},
        random_state=random_state
    )

    # Навчання, оцінка та візуалізація роботи моделей
    train_eval_visualize_model(shallow_tree, "Shallow Tree", x_train, y_train, x_test, y_test)
    train_eval_visualize_model(deep_tree, "Deep Tree", x_train, y_train, x_test, y_test)
    train_eval_visualize_model(shallow_forest, "Shallow Forest", x_train, y_train, x_test, y_test)
    train_eval_visualize_model(deep_forest, "Deep Forest", x_train, y_train, x_test, y_test)
    train_eval_visualize_model(avoid_type_2_error_classifier, "Avoid Type II Error Tree", x_train, y_train, x_test,
                               y_test)


if __name__ == "__main__":
    main()
