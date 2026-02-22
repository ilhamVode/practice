# src/synthesis.py
import numpy as np

SAMPLE_RATE = 44100


def midi_to_freq(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def adsr_envelope(n, sr, attack=0.01, decay=0.04, sustain_level=0.8, release=0.06):
    """
    Создаёт ADSR-огибающую строго длиной n сэмплов.
    """

    total_len = n / sr

    a = attack
    d = decay
    r = release

    # если суммарная длительность этапов больше длины ноты — масштабируем
    if a + d + r > total_len:
        factor = total_len / (a + d + r + 1e-9)
        a *= factor
        d *= factor
        r *= factor

    a_n = int(a * sr)
    d_n = int(d * sr)
    r_n = int(r * sr)

    s_n = n - (a_n + d_n + r_n)
    if s_n < 0:
        s_n = 0

    env = np.zeros(n, dtype=np.float32)

    # Attack
    if a_n > 0:
        env[:a_n] = np.linspace(0, 1, a_n, endpoint=False)

    # Decay
    if d_n > 0:
        start = a_n
        end = start + d_n
        env[start:end] = np.linspace(1, sustain_level, d_n, endpoint=False)

    # Sustain
    start = a_n + d_n
    end = start + s_n
    if s_n > 0:
        env[start:end] = sustain_level

    # Release
    start = end
    if r_n > 0 and start < n:
        env[start:] = np.linspace(sustain_level, 0, n - start, endpoint=False)

    return env


def generate_tone_by_freq(frequency, amplitude, duration, sr=SAMPLE_RATE):
    n = int(sr * duration)
    t = np.arange(n) / sr
    wave = (np.sin(2 * np.pi * frequency * t)
            + 0.35 * np.sin(2 * np.pi * frequency * 2 * t)
            + 0.12 * np.sin(2 * np.pi * frequency * 3 * t))
    env = adsr_envelope(n, sr, attack=0.01, decay=0.04, sustain_level=0.85, release=0.06)
    note = amplitude * wave * env
    return note.astype(np.float32)


import numpy as np

SAMPLE_RATE = 44100

def generate_continuous_tone(freq_interpolator, amp_interpolator, duration, sr=SAMPLE_RATE):
    """
    Генерирует непрерывный тон с изменяющейся частотой и амплитудой.
    freq_interpolator: функция времени (t от 0 до duration), возвращающая частоту в Гц.
    amp_interpolator: функция времени, возвращающая амплитуду (0..1).
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    # Фазовый аккумулятор для плавного изменения частоты
    phase = np.zeros(n)
    # Интегрируем мгновенную частоту для получения фазы
    freq_inst = freq_interpolator(t)
    # Защита от слишком низких частот
    freq_inst = np.maximum(freq_inst, 50)
    phase[0] = 0
    for i in range(1, n):
        phase[i] = phase[i-1] + 2 * np.pi * freq_inst[i] / sr
        # Не сбрасываем фазу, чтобы не было щелчков
    # Сигнал с добавлением небольшого количества гармоник для теплоты
    wave = (np.sin(phase)
            + 0.25 * np.sin(2 * phase)
            + 0.1 * np.sin(3 * phase))
    amp_inst = amp_interpolator(t)
    audio = amp_inst * wave
    # Нормализация на всякий случай
    max_amp = np.max(np.abs(audio))
    if max_amp > 1.0:
        audio = audio / max_amp
    return audio.astype(np.float32)

import numpy as np

SAMPLE_RATE = 44100

def generate_continuous_tone_with_timbre(freq_interp, amp_interp, brightness_interp, duration, sr=SAMPLE_RATE):
    """
    Генерирует непрерывный тон с изменяющейся частотой, амплитудой и тембром.
    brightness: от 0 (чистый синус) до 1 (насыщенный гармониками звук).
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    freq = freq_interp(t)
    amp = amp_interp(t)
    brightness = brightness_interp(t)

    # Фазовый аккумулятор для плавного изменения частоты
    phase = np.zeros(n)
    phase[0] = 0
    for i in range(1, n):
        phase[i] = phase[i-1] + 2 * np.pi * freq[i] / sr

    # Основная частота
    signal = np.sin(phase)

    # Добавление гармоник с весами, зависящими от brightness
    # Вторая гармоника с весом 0.3*brightness, третья – 0.15*brightness
    harmonic2 = 0.3 * brightness * np.sin(2 * phase)
    harmonic3 = 0.15 * brightness * np.sin(3 * phase)

    signal = signal + harmonic2 + harmonic3

    # Нормализация, чтобы сохранить примерно ту же громкость
    norm_factor = 1 + 0.3 * brightness + 0.15 * brightness
    signal = signal / norm_factor

    # Применяем амплитудную огибающую
    signal = amp * signal

    # Финальная нормализация на всякий случай
    max_amp = np.max(np.abs(signal))
    if max_amp > 1.0:
        signal = signal / max_amp

    return signal.astype(np.float32)
