
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# ЛАБОРАТОРНАЯ РАБОТА №2
# КЛАССИФИКАЦИЯ — ПРОГНОЗИРОВАНИЕ ОТТОКА КЛИЕНТОВ
# ============================================================


# 1. Загрузка данных

path = "Telco-Customer-Churn.csv"

try:
    df = pd.read_csv(path)

except FileNotFoundError:
    print("Файл не найден:", path)
    exit()

except Exception as e:
    print("Ошибка чтения файла:", e)
    exit()


# 2. Проверка наличия целевого столбца

if "Churn" not in df.columns:
    print("В датасете отсутствует столбец Churn.")
    print("Доступные столбцы:", df.columns.tolist())
    exit()


# 3. Подготовка данных

# Удаляем идентификатор клиента

df = df.drop(
    columns=["customerID"],
    errors="ignore"
)

# Преобразуем TotalCharges в числовой формат

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

# Удаляем строки без целевого класса

df = df.dropna(subset=["Churn"])

df = df[
    df["Churn"].isin(["Yes", "No"])
].copy()


# 4. Выбор признаков и целевой переменной

X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# 5. Определение числовых и категориальных признаков

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


# 6. Обработка числовых признаков

numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="mean")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# 7. Обработка категориальных признаков

categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# 8. Объединение обработчиков данных

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_transformer,
        numeric_features
    ),
    (
        "cat",
        categorical_transformer,
        categorical_features
    )
])


# 9. Создание модели логистической регрессии

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            C=1.0,
            penalty="l2",
            max_iter=2000,
            solver="lbfgs"
        )
    )
])


# 10. Разделение на обучающую и тестовую выборки

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 11. Обучение модели

model.fit(
    X_train,
    y_train
)

# Получение предсказаний

predictions = model.predict(X_test)

# Получение вероятностей для расчета AUC

probabilities = model.predict_proba(X_test)[:, 1]


# 12. Расчет метрик классификации

# Accuracy — доля правильных прогнозов

accuracy = accuracy_score(
    y_test,
    predictions
)

# Precision — взвешенная точность

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

# Recall — взвешенная полнота

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

# F1 — взвешенная F-мера

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

# AUC — площадь под ROC-кривой

auc = roc_auc_score(
    y_test,
    probabilities
)

# MCC — коэффициент корреляции Мэтьюса

mcc = matthews_corrcoef(
    y_test,
    predictions
)


# 13. Вывод результатов

print("\n========================================")
print("     РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
print("========================================")

print("Количество клиентов:", len(df))

print("Обучающая выборка:", len(X_train))
print("Тестовая выборка:", len(X_test))

print("----------------------------------------")

print("Accuracy:", round(accuracy, 4))
print("AUC:", round(auc, 4))
print("F1:", round(f1, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("MCC:", round(mcc, 4))

print("========================================")


# 14. Построение матрицы ошибок

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No", "Yes"]
)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Матрица ошибок — отток клиентов")

plt.tight_layout()
plt.show()


# 15. Вывод элементов матрицы ошибок

tn, fp, fn, tp = cm.ravel()

print("\n========================================")
print("           МАТРИЦА ОШИБОК")
print("========================================")

print("TN — верно предсказано No:", tn)
print("FP — ошибочно предсказано Yes:", fp)
print("FN — пропущен реальный уход:", fn)
print("TP — верно предсказано Yes:", tp)

print("========================================")


# 16. Дополнительный отчет по классам

print("\n========================================")
print("       ОТЧЕТ ПО КЛАССАМ")
print("========================================")

from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        predictions,
        target_names=["No", "Yes"],
        zero_division=0
    )
)


# 17. Вывод

print("\n========================================")
print("                 ВЫВОД")
print("========================================")

print("Accuracy — доля правильных прогнозов.")
print("Precision — точность положительных прогнозов.")
print("Recall — полнота обнаружения положительного класса.")
print("F1 — гармоническое среднее Precision и Recall.")
print("AUC — способность модели разделять классы.")
print("MCC — общая оценка качества классификации.")

print("\nМодель логистической регрессии обучена.")
print("Классификация оттока клиентов завершена.")
print("========================================")
