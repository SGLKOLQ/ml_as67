
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ЛАБОРАТОРНАЯ РАБОТА №2
# РЕГРЕССИЯ — ПРОГНОЗИРОВАНИЕ СЧАСТЬЯ В СТРАНАХ


# 1. Загрузка данных

path = "world_happiness_report .csv"

try:
    df = pd.read_csv(path)
except FileNotFoundError:
    print("Файл не найден:", path)
    exit()
except Exception as e:
    print("Ошибка чтения файла:", e)
    exit()


# 2. Поддержка разных версий датасета

aliases = {
    "Score": [
        "Score", "Ladder score",
        "Happiness score", "Life Ladder", "score"
    ],
    "GDP per capita": [
        "GDP per capita", "Logged GDP per capita",
        "Log GDP per capita"
    ],
    "Social support": [
        "Social support"
    ],
    "Healthy life expectancy": [
        "Healthy life expectancy",
        "Healthy life expectancy at birth"
    ]
}

rename = {}

for standard_name, variants in aliases.items():
    for variant in variants:
        if variant in df.columns:
            rename[variant] = standard_name
            break

df = df.rename(columns=rename)


# 3. Выбор признаков и целевой переменной

features = [
    "GDP per capita",
    "Social support",
    "Healthy life expectancy"
]

target = "Score"

missing = [
    col for col in features + [target]
    if col not in df.columns
]

if missing:
    print("Не найдены столбцы:", missing)
    print("Доступные столбцы:", df.columns.tolist())
    exit()

data = df[features + [target]].copy()

for col in features + [target]:
    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )

data = data.dropna()

if len(data) < 5:
    print("Недостаточно данных для обучения.")
    exit()

X = data[features]
y = data[target]


# 4. Разделение на обучающую и тестовую выборки

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 5. Обучение модели линейной регрессии

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# 6. Расчет MSE и R2

mse = mean_squared_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n=== РЕЗУЛЬТАТЫ РЕГРЕССИИ ===")

print("Количество записей:", len(data))
print("MSE:", round(mse, 4))
print("R2:", round(r2, 4))


# 7. Вывод коэффициентов модели

print("\nКоэффициенты модели:")

for name, coef in zip(features, model.coef_):
    print(f"{name}: {coef:.4f}")

print(
    "Свободный коэффициент:",
    round(model.intercept_, 4)
)


# 8. Визуализация зависимости Score от GDP per capita

x = data[["GDP per capita"]]
y_all = data["Score"]

graph_model = LinearRegression()

graph_model.fit(x, y_all)

x_line = pd.DataFrame({
    "GDP per capita": [
        x["GDP per capita"].min(),
        x["GDP per capita"].max()
    ]
})

y_line = graph_model.predict(x_line)

plt.figure(figsize=(9, 6))

plt.scatter(
    x["GDP per capita"],
    y_all,
    alpha=0.7,
    label="Страны"
)

plt.plot(
    x_line["GDP per capita"],
    y_line,
    color="red",
    label="Линия регрессии"
)

plt.xlabel("GDP per capita")
plt.ylabel("Score")

plt.title(
    "Зависимость счастья от ВВП на душу населения"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# 9. Вывод

print("\n=== ВЫВОД ===")
print("Модель линейной регрессии обучена.")
print("MSE показывает величину ошибки прогнозирования.")
print("R2 показывает, насколько модель объясняет вариацию Score.")
