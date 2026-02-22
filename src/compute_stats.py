import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from main4 import get_data


def compute_stats(data, column='AMZN'):
    """
    Вычисляет основные статистические характеристики для указанного столбца данных.

    Параметры:
    - data: pandas DataFrame, содержащий данные
    - column: str, название столбца с ценой (по умолчанию 'AMZN')

    Возвращает:
    - stats_dict: словарь с показателями
    """
    # Проверка наличия колонки
    if column not in data.columns:
        raise ValueError(f"Столбец '{column}' не найден в данных.")

    # Извлекаем значения, убираем пропуски (на всякий случай)
    values = data[column].dropna()

    # Расчёт статистик
    stats = {
        'Минимум': values.min(),
        'Максимум': values.max(),
        'Среднее арифметическое': values.mean(),
        'Медиана': values.median(),
        'Стандартное отклонение': values.std(),
        'Диапазон (макс - мин)': values.max() - values.min()
    }

    return stats


if __name__ == "__main__":
    filepath = "../data/portfolio_data.csv"

    data, window, limit = get_data(filepath, column='AMZN')

    stats_amzn = compute_stats(data, column='AMZN')

    print("Статистические характеристики для AMZN:")
    for key, value in stats_amzn.items():
        print(f"{key:25}: {value:.2f}")