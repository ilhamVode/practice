import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.io.wavfile import write
from .mapping import map_trend_and_speed_to_freq, map_speed_to_amplitude
from .synthesis import generate_continuous_tone_with_timbre, SAMPLE_RATE

def sonify(preprocessed_csv, output_wav="output/amzn_sonification.wav", total_duration=180):
    df = pd.read_csv(preprocessed_csv)
    required = {'Trend', 'Speed', 'Speed_Norm', 'Event'}
    if not required.issubset(df.columns):
        raise ValueError("Нет необходимых колонок в CSV")

    # Скорость и знак
    speed_sign = np.sign(df['Speed'].fillna(0))
    speed_norm = df['Speed_Norm'].values
    trend = df['Trend'].values

    # Частота для каждого дня (базовая + добавка от скорости)
    from .mapping import map_trend_and_speed_to_freq
    freqs_daily = map_trend_and_speed_to_freq(trend, speed_norm, speed_sign,
                                              base_min=200, base_max=600, delta_max=40)

    # Амплитуда для каждого дня (зависит от скорости)
    amps_daily = map_speed_to_amplitude(speed_norm, min_amp=0.2, max_amp=0.8)

    # Яркость тембра: зависит от скорости и событий
    # Базовая яркость пропорциональна speed_norm (чем активнее рынок, тем ярче звук)
    brightness_base = speed_norm * 0.7  # макс 0.7, чтобы не перегружать
    # Если событие (аномальное изменение), добавляем всплеск яркости
    event_factor = df['Event'].values * 0.5  # при событии добавляем 0.5
    # Общая яркость, ограниченная 1.0
    brightness_daily = np.clip(brightness_base + event_factor, 0.0, 1.0)

    # Временные метки для дней
    n_days = len(df)
    time_days = np.linspace(0, total_duration, n_days)

    # Интерполяторы
    freq_interp = interp1d(time_days, freqs_daily, kind='linear', fill_value='extrapolate')
    amp_interp = interp1d(time_days, amps_daily, kind='linear', fill_value='extrapolate')
    brightness_interp = interp1d(time_days, brightness_daily, kind='linear', fill_value='extrapolate')

    # Генерация звука с изменяемым тембром
    audio = generate_continuous_tone_with_timbre(freq_interp, amp_interp, brightness_interp, total_duration)

    # Сохранение
    audio_int16 = (audio * 32767).astype(np.int16)
    write(output_wav, SAMPLE_RATE, audio_int16)
    print(f"WAV сохранён: {output_wav} (длительность {total_duration} с)")