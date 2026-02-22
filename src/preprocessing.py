import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess(filepath, column='AMZN', window=11, event_limit=0.10):
    """
    Загружает данные из CSV, выполняет предобработку и возвращает обработанный DataFrame.

    Параметры:
        filepath (str): путь к CSV-файлу
        column (str): название столбца с ценой актива
        window (int): размер окна для скользящего среднего
        event_limit (float): порог для определения аномальных изменений (в долях)

    Возвращает:
        pd.DataFrame: обработанные данные с дополнительными колонками
    """
    # Загрузка данных
    data = pd.read_csv(filepath)

    # Преобразование даты
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

    # Проверка на пропуски (для выбранного столбца)
    if data[column].isnull().any():
        print(f"Внимание: в столбце {column} есть пропуски. Они будут удалены.")
        data = data.dropna(subset=[column])
    else:
        print(f"Пропуски в столбце {column} отсутствуют.")

    # Логарифмирование цены
    data['Log_Price'] = np.log(data[column])

    # Сглаживание (скользящее среднее)
    data['Trend'] = data['Log_Price'].rolling(window=window, center=True, min_periods=1).mean()

    # Скорость изменения (разность тренда)
    data['Speed'] = data['Trend'].diff()

    # Нормализация скорости
    scaler = MinMaxScaler()
    # fillna(0) заменяет NaN, возникшие при diff() на первой строке
    data['Speed_Norm'] = scaler.fit_transform(data[['Speed']].fillna(0))

    # Выделение событий (аномальные изменения)
    data['Event'] = np.where(data['Speed'].abs() > event_limit, 1, 0)

    return data


def save_preprocessed(data, output_path):
    """
    Сохраняет обработанный DataFrame в CSV.
    """
    data.to_csv(output_path, index=False)
    print(f"Обработанные данные сохранены в {output_path}")


if __name__ == "__main__":
    input_file = "../data/portfolio_data.csv"
    output_file = "../data/preprocessed_amzn.csv"
    target_column = "AMZN"

    df = load_and_preprocess(input_file, column=target_column)

    # Оставляем только нужные колонки
    columns_to_keep = ['Date', 'AMZN', 'Log_Price', 'Trend', 'Speed', 'Speed_Norm', 'Event']
    df = df[columns_to_keep]

    save_preprocessed(df, output_file)
