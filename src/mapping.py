# src/mapping.py
import numpy as np

# MIDI -> частота (Гц)
def midi_to_freq(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def build_major_scale(midi_low=60, midi_high=84):
    """
    Возвращает список MIDI-нотов, входящих в C-мажорную шкалу от midi_low до midi_high.
    По умолчанию C4(60) .. C6(84).
    """
    # ступени мажора: 0,2,4,5,7,9,11
    intervals = [0, 2, 4, 5, 7, 9, 11]
    notes = []
    base = 12 * (midi_low // 12)  # ближайший базовый октавный шаг
    # начинаем с C (но мы ориентируемся от midi_low)
    for m in range(midi_low, midi_high + 1):
        octave = (m % 12)
        # найдем ближайшую ступень мажора по модулю 12 относительно C
        if octave in intervals:
            notes.append(m)
    # Для гарантии — если notes пуст, построим простую последовательность
    if not notes:
        notes = list(range(midi_low, midi_high + 1))
    return notes

def map_log_to_midi(log_series, scale_notes):
    """Маппим лог-цены на ближайшую ступень в scale_notes"""
    vals = np.array(log_series, dtype=float)
    minv, maxv = vals.min(), vals.max()
    if maxv - minv < 1e-9:
        idxs = np.zeros_like(vals, dtype=int)
    else:
        norm = (vals - minv) / (maxv - minv)
        idxs = np.round(norm * (len(scale_notes) - 1)).astype(int)
    midi = [int(scale_notes[i]) for i in idxs]
    return np.array(midi, dtype=int)

def map_speed_to_amplitude(speed_norm, min_amp=0.08, max_amp=0.55):
    """speed_norm: 0..1 -> амплитуда в безопасных пределах"""
    return min_amp + (max_amp - min_amp) * np.clip(speed_norm, 0.0, 1.0)

def map_speed_to_duration(speed_norm, min_dur=0.12, max_dur=0.25):
    """
    Чем спокойнее (speed small), тем длиннее нота.
    Используем обратную зависимость.
    """
    inv = 1.0 - np.clip(speed_norm, 0.0, 1.0)
    return min_dur + (max_dur - min_dur) * inv

def map_speed_to_freq(speed_norm, speed_sign, base_freq=440, range_freq=220):
    """
    speed_norm: 0..1 (модуль скорости)
    speed_sign: -1, 0, 1 (направление)
    base_freq: частота при нулевой скорости (Гц)
    range_freq: максимальное отклонение вверх/вниз
    Возвращает частоту в Гц.
    """
    if speed_sign == 0:
        return base_freq
    # Знак определяет направление: +1 => выше, -1 => ниже
    delta = speed_norm * range_freq * speed_sign
    freq = base_freq + delta
    # Защита от слишком низких частот (ниже 50 Гц не слышно)
    if freq < 50:
        freq = 50
    return freq

def map_trend_and_speed_to_freq(trend_vals, speed_norm, speed_sign,
                                base_min=200, base_max=600, delta_max=70):
    """
    trend_vals : массив значений сглаженного тренда (Log_Price после сглаживания)
    speed_norm : массив значений скорости, нормализованных в [0,1]
    speed_sign : массив знаков скорости (-1, 0, 1)
    base_min, base_max : диапазон базовой частоты (Гц) от минимального к максимальному тренду
    delta_max : максимальное отклонение от базовой частоты за счёт скорости (Гц)
    """
    # Нормализуем тренд
    t_min = trend_vals.min()
    t_max = trend_vals.max()
    if t_max - t_min < 1e-9:
        norm_trend = np.zeros_like(trend_vals)
    else:
        norm_trend = (trend_vals - t_min) / (t_max - t_min)
    base_freq = base_min + norm_trend * (base_max - base_min)

    # Добавка от скорости: знак * модуль * коэффициент
    delta = speed_sign * speed_norm * delta_max
    freq = base_freq + delta
    # Защита от выхода за разумные пределы
    freq = np.clip(freq, 100, 1000)
    return freq