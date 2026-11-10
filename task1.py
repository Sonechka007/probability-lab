"""
Задание 1. Bin(10; 0,5): генерация выборки и проверка закона больших чисел.
Запуск: python task1.py  (графики сохраняются в папку figs/)
Нужны библиотеки: numpy, scipy, matplotlib
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, chisquare

os.makedirs("figs", exist_ok=True)

# ---------- Параметры ----------
n, p = 10, 0.5      # распределение Bin(10; 0,5)
N = 1000            # объём одной выборки
M = 200             # число экспериментов для проверки ЗБЧ

EX = n * p                 # E[X] = 5
DX = n * p * (1 - p)       # D[X] = 2,5
print("E[X] =", EX, " D[X] =", DX, " sigma =", round(DX ** 0.5, 3))

rng = np.random.default_rng(42)   # генератор с фиксированным seed (результат воспроизводим)


# ---------- Генерация методом обратной функции ----------
def generate(size):
    u = rng.random(size)          # U ~ U(0,1)
    return binom.ppf(u, n, p)     # X = F^(-1)(U)


# ---------- 1. Выборка, гистограмма и теоретическая функция вероятности ----------
sample = generate(N)
print("Выборочное среднее:", sample.mean())
print("Выборочная дисперсия:", sample.var(ddof=1))

k = np.arange(0, n + 1)
freq = np.array([(sample == i).mean() for i in k])   # относительные частоты
theory_p = binom.pmf(k, n, p)                        # теоретические вероятности
print("k, частота, теория:")
for i in k:
    print(i, round(freq[i], 3), round(theory_p[i], 4))

# Критерий хи-квадрат: объединяем края (k=0,1 и k=9,10), чтобы ожидаемые частоты были >= 5
obs = np.bincount(sample.astype(int), minlength=n + 1)      # наблюдаемые числа
exp = N * theory_p                                         # ожидаемые числа
obs_g = np.r_[obs[0] + obs[1], obs[2:9], obs[9] + obs[10]]
exp_g = np.r_[exp[0] + exp[1], exp[2:9], exp[9] + exp[10]]
chi2, pval = chisquare(obs_g, exp_g)
print("chi2 =", round(chi2, 3), " df =", len(obs_g) - 1, " p-value =", round(pval, 3))

plt.figure(figsize=(8, 5))
plt.hist(sample, bins=np.arange(-0.5, n + 1.5, 1), density=True,
         edgecolor="black", alpha=0.6, label="Гистограмма выборки")
plt.plot(k, theory_p, "ro-", label="Теоретическая P(X=k)")
plt.xlabel("k")
plt.ylabel("Относительная частота")
plt.title("Bin(10; 0,5): гистограмма и теория")
plt.legend()
plt.savefig("figs/hist.png", dpi=200)
plt.show()

# ---------- 2. ЗБЧ: выборочные средние в M экспериментах ----------
means = np.array([generate(N).mean() for _ in range(M)])
print("Средние по экспериментам: min =", means.min(), " max =", means.max(),
      " std =", means.std(), " теория sigma/sqrt(N) =", (DX / N) ** 0.5)
print("Доля средних в пределах E[X] +- 2*sigma/sqrt(N):",
      np.mean(np.abs(means - EX) <= 2 * (DX / N) ** 0.5))

plt.figure(figsize=(8, 5))
plt.plot(range(1, M + 1), means, ".", label="Выборочное среднее")
plt.axhline(EX, color="red", label="E[X] = 5")
plt.xlabel("Номер эксперимента")
plt.ylabel("Выборочное среднее")
plt.title(f"ЗБЧ: средние по {M} экспериментам (N = {N})")
plt.legend()
plt.savefig("figs/lln_means.png", dpi=200)
plt.show()

# ---------- 3. ЗБЧ: среднее при росте числа наблюдений ----------
long_sample = generate(5000)
running_mean = np.cumsum(long_sample) / np.arange(1, 5001)

plt.figure(figsize=(8, 5))
plt.plot(running_mean, label="Среднее по первым n наблюдениям")
plt.axhline(EX, color="red", label="E[X] = 5")
plt.xlabel("n")
plt.ylabel("Выборочное среднее")
plt.title("Сходимость среднего к E[X]")
plt.legend()
for m_ in [10, 100, 1000, 5000]:
    print(f"Среднее по первым {m_} наблюдениям: {running_mean[m_ - 1]:.3f}")
plt.savefig("figs/lln_running.png", dpi=200)
plt.show()

# ---------- 4. Скорость сходимости: std выборочного среднего от N ----------
sizes = [10, 50, 100, 500, 1000, 5000]
stds = []
for size in sizes:
    means_size = generate((500, size)).mean(axis=1)   # 500 экспериментов
    stds.append(means_size.std())

theory = np.sqrt(DX / np.array(sizes))                # sigma / sqrt(N)

plt.figure(figsize=(8, 5))
plt.loglog(sizes, stds, "o-", label="Эксперимент")
plt.loglog(sizes, theory, "r--", label="Теория: σ/√N")
plt.xlabel("N")
plt.ylabel("Стандартное отклонение X̄_N")
plt.title("Скорость сходимости в ЗБЧ")
plt.legend()
plt.savefig("figs/lln_rate.png", dpi=200)
plt.show()

for size, s, t in zip(sizes, stds, theory):
    print(f"N = {size:5d}: эксперимент {s:.4f}, теория {t:.4f}")

slope = np.polyfit(np.log(sizes), np.log(stds), 1)[0]
print("Наклон прямой в логарифмическом масштабе:", round(slope, 3))
