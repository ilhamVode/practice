import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загрузка обработанных данных
data = pd.read_csv('../data/preprocessed_amzn.csv', parse_dates=['Date'])

# 1. График исходной цены AMZN
plt.figure(figsize=(12, 5))
plt.plot(data['Date'], data['AMZN'], color='blue', linewidth=1)
plt.xlabel('Дата')
plt.ylabel('Цена (USD)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('images/raw_price.png', dpi=150)
plt.close()

# 2. График логарифмированной цены
plt.figure(figsize=(12, 5))
plt.plot(data['Date'], data['Log_Price'], color='green', linewidth=1)
plt.xlabel('Дата')
plt.ylabel('ln(Цена)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('images/log_price.png', dpi=150)
plt.close()

# 3. График сглаженного тренда и скорости изменения (два подграфика)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Тренд
ax1.plot(data['Date'], data['Trend'], color='red', linewidth=1.5)
ax1.set_title('Сглаженный тренд (скользящее среднее, окно 11)')
ax1.set_ylabel('ln(Цена)')
ax1.grid(True, linestyle='--', alpha=0.6)

# Скорость изменения (нормализованная)
ax2.plot(data['Date'], data['Speed_Norm'], color='purple', linewidth=1)
ax2.set_title('Нормализованная скорость изменения цены')
ax2.set_xlabel('Дата')
ax2.set_ylabel('Speed_Norm')
ax2.grid(True, linestyle='--', alpha=0.6)

# Отметим события (аномальные изменения) на втором графике
event_dates = data.loc[data['Event'] == 1, 'Date']
event_values = data.loc[data['Event'] == 1, 'Speed_Norm']
ax2.scatter(event_dates, event_values, color='orange', s=20, label='События (>5%)')
ax2.legend()

plt.tight_layout()
plt.savefig('images/trend_speed.png', dpi=150)
plt.close()

print("Графики сохранены в папку images/")