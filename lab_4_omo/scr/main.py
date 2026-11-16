import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED) 
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Глобальные переменные для данных
X_train = X_test = y_train = y_test = None
numerical_columns = None


def load_and_prepare_data():
    """Пункт 1: загрузка и предварительная обработка данных."""
    global X_train, X_test, y_train, y_test, numerical_columns

    # 2. Загрузка данных
    df = pd.read_csv("adult.csv", skipinitialspace=True)

    print("Размер датасета:", df.shape)
    print("\nПервые 5 строк:")
    print(df.head())

    print("\nИнформация о датасете:")
    print(df.info())

    # 3. Предварительная обработка
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()

    df = df.replace("?", np.nan)
    df = df.dropna()
    df["income"] = df["income"].str.replace(".", "", regex=False)

    y = (df["income"] == ">50K").astype(int)
    X = df.drop(columns=["income"])

    print("\nРаспределение классов:")
    print(y.value_counts())

    # 4. Разделение на выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 5. Кодирование признаков
    categorical_columns = X.select_dtypes(include="object").columns
    numerical_columns = X.select_dtypes(exclude="object").columns
    X_train = pd.get_dummies(X_train, columns=categorical_columns, dtype=int)
    X_test = pd.get_dummies(X_test, columns=categorical_columns, dtype=int)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    scaler = StandardScaler()
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    X_train = X_train.astype("float32").to_numpy()
    X_test = X_test.astype("float32").to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    print("\nКоличество входных признаков:", X_train.shape[1])


def create_model(deep=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    if deep:
        model.add(tf.keras.layers.Dense(16, activation="relu"))
        model.add(tf.keras.layers.Dense(16, activation="relu"))
    else:
        model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_and_evaluate(model, name):
    print(f"\n{'=' * 50}")
    print(f"Обучение модели: {name}")
    print(f"{'=' * 50}")
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    probabilities = model.predict(X_test, verbose=0).flatten()
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print(f"\nРезультаты модели: {name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nМатрица ошибок:")
    print(confusion_matrix(y_test, predictions))
    print("\nОтчет классификации:")
    print(classification_report(y_test, predictions))

    return history, {
        "Модель": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": auc
    }


def run_both_models(show_plots=True):
    """Пункт 2: обучение обеих моделей + сравнение + графики."""
    # 8. Первая модель
    model_1 = create_model(deep=False)
    history_1, result_1 = train_and_evaluate(
        model_1, "Один скрытый слой (32 нейрона)"
    )

    # 9. Вторая модель
    model_2 = create_model(deep=True)
    history_2, result_2 = train_and_evaluate(
        model_2, "Два скрытых слоя (16 + 16 нейронов)"
    )

    # 10. Сравнение
    results = pd.DataFrame([result_1, result_2])
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛЕЙ")
    print("=" * 60)
    print(results.to_string(index=False))

    if show_plots:
        # 11. График сравнения точности
        plt.figure(figsize=(9, 5))
        plt.bar(results["Модель"], results["Accuracy"])
        plt.title("Сравнение точности моделей")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.xticks(rotation=10)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 12. График функции потерь
        plt.figure(figsize=(10, 5))
        plt.plot(history_1.history["val_loss"], label="32 нейрона")
        plt.plot(history_2.history["val_loss"], label="16 + 16 нейронов")
        plt.title("Сравнение функции потерь")
        plt.xlabel("Эпоха")
        plt.ylabel("Validation Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    print("\nКонец!")
    return results


def menu():
    while True:
        print("\n" + "=" * 50)
        print("МЕНЮ")
        print("=" * 50)
        print("1. Загрузить и подготовить данные")
        print("2. Обучить обе модели и сравнить")
        print("3. Полный прогон (данные + модели)")
        print("0. Выход")
        print("=" * 50)

        choice = input("Выберите пункт: ").strip()

        if choice == "1":
            try:
                load_and_prepare_data()
                print("\nДанные успешно подготовлены.")
            except Exception as e:
                print(f"\nОшибка при подготовке данных: {e}")

        elif choice == "2":
            if X_train is None:
                print("\nСначала загрузите данные (пункт 1).")
                continue
            try:
                run_both_models()
            except Exception as e:
                print(f"\nОшибка при обучении: {e}")

        elif choice == "3":
            try:
                load_and_prepare_data()
                run_both_models()
            except Exception as e:
                print(f"\nОшибка: {e}")

        elif choice == "0":
            print("\nВыход.")
            break

        else:
            print("\nНеверный пункт меню.")


if __name__ == "__main__":
    menu()
