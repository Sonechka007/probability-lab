"""

Задание 2. Bin(5, 0.3) и центральная предельная теорема.

Запуск: python task2_clt.py  -> графики PNG в папку figs/, таблицы в results.json

"""

import json

import os

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy import stats

SEED = 42

M, P = 5, 0.3                      # параметры одного слагаемого X ~ Bin(5, 0.3)

MU = M * P                         # E[X] = 1.5

VAR = M * P * (1 - P)              # D[X] = 1.05

K = 10_000                         # число сумм для каждого n

NS = [1, 2, 5, 10, 50, 100, 500]   # число слагаемых

ALPHA = 0.05

os.makedirs("figs", exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

def sums(rng, n, k, m=M, p=P):

    """k сумм S_n = X_1 + ... + X_n, X_i ~ Bin(m, p)."""

    return rng.binomial(m, p, size=(k, n)).sum(axis=1)

def z_of(s, n, m=M, p=P):

    mu, var = m * p, m * p * (1 - p)

    return (s - mu * n) / np.sqrt(var * n)

def chi2_test(s, n, m=M, p=P, min_exp=5):

    """Критерий χ² для гипотезы Z_n ~ N(0,1).

    Каждому целому значению S = j соответствует интервал Z

    [(j-0.5-μn)/σ√n, (j+0.5-μn)/σ√n]; соседние интервалы объединяются,

    пока ожидаемое число попаданий < min_exp. Крайние интервалы до ±∞."""

    mu, sd = m * p * n, np.sqrt(m * p * (1 - p) * n)

    jmax = m * n

    obs_j = np.bincount(s, minlength=jmax + 1)

    edges_z = (np.arange(-0.5, jmax + 1.5) - mu) / sd

    cdf = stats.norm.cdf(edges_z)

    cdf[0], cdf[-1] = 0.0, 1.0                 # хвосты нормального закона

    exp_j = len(s) * np.diff(cdf)

    obs, exp = [], []

    o_acc = e_acc = 0.0

    for o, e in zip(obs_j, exp_j):

        o_acc += o

        e_acc += e

        if e_acc >= min_exp:

            obs.append(o_acc); exp.append(e_acc)

            o_acc = e_acc = 0.0

    if e_acc > 0:                              # остаток присоединяем к последнему

        obs[-1] += o_acc; exp[-1] += e_acc

    obs, exp = np.array(obs), np.array(exp)

    exp *= obs.sum() / exp.sum()               # точная нормировка сумм

    stat, pval = stats.chisquare(obs, exp)     # df = r - 1 (параметры известны)

    return float(stat), float(pval), len(obs) - 1

def exact_distance(n, m=M, p=P, corrected=False):

    """Точное sup|F_{Z_n}(z) - Φ(z)| по закону S_n ~ Bin(mn, p).

    corrected=True: сравнение в серединах шагов (поправка на непрерывность)."""

    mu, sd = m * p * n, np.sqrt(m * p * (1 - p) * n)

    j = np.arange(m * n + 1)

    F = stats.binom.cdf(j, m * n, p)

    if corrected:

        return float(np.max(np.abs(F - stats.norm.cdf((j + 0.5 - mu) / sd))))

    Fl = np.concatenate(([0.0], F[:-1]))       # значение слева от скачка

    phi = stats.norm.cdf((j - mu) / sd)

    return float(max(np.max(np.abs(F - phi)), np.max(np.abs(Fl - phi))))

res = {}

rng = np.random.default_rng(SEED)

# ---------- 1. Генерация сумм, проверка моментов, критерии ----------

data, rows = {}, []

for n in NS:

    s = sums(rng, n, K)

    z = z_of(s, n)

    data[n] = (s, z)

    ks = stats.kstest(z, "norm")

    h = 1 / np.sqrt(VAR * n)                     # шаг решётки Z_n

    zj = z + rng.uniform(-h / 2, h / 2, size=K)  # «сглаживание» решётки

    ks_j = stats.kstest(zj, "norm")

    chi = chi2_test(s, n)

    # проверка кода: S_n должно иметь точно закон Bin(5n, 0.3)

    pmf = stats.binom.pmf(np.arange(M * n + 1), M * n, P)

    obs_b = np.bincount(s, minlength=M * n + 1)

    keep = pmf * K >= 5

    o_b = np.append(obs_b[keep], obs_b[~keep].sum())

    e_b = np.append(pmf[keep], pmf[~keep].sum()) * K

    if e_b[-1] < 1e-9:

        o_b, e_b = o_b[:-1], e_b[:-1]

    e_b *= o_b.sum() / e_b.sum()

    binom_p = float(stats.chisquare(o_b, e_b).pvalue)

    se_mean = np.sqrt(VAR * n / K)

    se_var = VAR * n * np.sqrt(2 / (K - 1))

    rows.append(dict(

        binom_p=binom_p,

        mean_se=float((s.mean() - MU * n) / se_mean),

        var_se=float((s.var(ddof=1) - VAR * n) / se_var),

        n=n, mean=float(s.mean()), mean_th=MU * n,

        var=float(s.var(ddof=1)), var_th=VAR * n,

        z_mean=float(z.mean()), z_var=float(z.var(ddof=1)),

        skew=float(stats.skew(z)), skew_th=(1 - 2 * P) / np.sqrt(VAR * n),

        chi2=chi[0], chi2_p=chi[1], chi2_df=chi[2],

        ks_D=float(ks.statistic), ks_p=float(ks.pvalue),

        ksj_D=float(ks_j.statistic), ksj_p=float(ks_j.pvalue),

        step=h, exact_D=exact_distance(n)))

res["table"] = rows

# ---------- 2. Гистограммы Z_n ----------

fig, axes = plt.subplots(2, 4, figsize=(13, 6))

xx = np.linspace(-4, 4, 400)

for ax, n in zip(axes.flat, NS):

    s, z = data[n]

    h = 1 / np.sqrt(VAR * n)

    lo, hi = s.min(), s.max()

    edges = (np.arange(lo - 0.5, hi + 1.5) - MU * n) / np.sqrt(VAR * n)  # бины по шагу h

    ax.hist(z, bins=edges, density=True, color="#7aa6d6", edgecolor="white",

            linewidth=0.3, label="Гистограмма $Z_n$")

    ax.plot(xx, stats.norm.pdf(xx), "r-", lw=1.5, label="Плотность N(0,1)")

    ax.set_xlim(-4, 4)

    ax.set_title(f"n = {n}, шаг = {h:.3f}".replace(".", ","))

    ax.set_xlabel("$z$"); ax.set_ylabel("плотность")

axes.flat[-1].axis("off")

h_, l_ = axes.flat[0].get_legend_handles_labels()

axes.flat[-1].legend(h_, l_, loc="center", fontsize=11)

fig.suptitle("Гистограммы нормированных сумм $Z_n$ для Bin(5; 0,3) и плотность N(0, 1)")

fig.tight_layout(); fig.savefig("figs/fig1_hist.png"); plt.close(fig)

# ---------- 3. Эмпирическая ФР против Φ (иллюстрация КС) ----------

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

for ax, n in zip(axes, [5, 50, 500]):

    z = np.sort(data[n][1])

    ax.step(z, np.arange(1, K + 1) / K, where="post", lw=1.2, label="Эмпирическая ФР $Z_n$")

    ax.plot(xx, stats.norm.cdf(xx), "r--", lw=1.2, label="Φ(z)")

    ax.set_xlim(-3, 3); ax.set_title(f"n = {n}")

    ax.set_xlabel("$z$"); ax.set_ylabel("F(z)"); ax.legend(fontsize=8)

fig.suptitle("Ступенчатая функция распределения $Z_n$ и нормальная Φ(z)")

fig.tight_layout(); fig.savefig("figs/fig2_ecdf.png"); plt.close(fig)

# ---------- 4. Скорость сходимости: точное расстояние ----------

grid = np.unique(np.round(np.logspace(0, 3, 40)).astype(int))

d03 = [exact_distance(n) for n in grid]

d05 = [exact_distance(n, p=0.5) for n in grid]

d03c = [exact_distance(n, corrected=True) for n in grid]

d05c = [exact_distance(n, p=0.5, corrected=True) for n in grid]

slope = lambda d: float(np.polyfit(np.log(grid[grid >= 10]), np.log(np.array(d)[grid >= 10]), 1)[0])

res["slopes"] = dict(p03=slope(d03), p05=slope(d05), p03c=slope(d03c), p05c=slope(d05c))

# эмпирическое расстояние (статистика КС) по выборке

emp = [stats.kstest(z_of(sums(rng, n, K), n), "norm").statistic for n in grid]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]

ax.loglog(grid, d03, "o-", ms=3, label="Точное $D_n$, Bin(5; 0,3)")

ax.loglog(grid, emp, "x", ms=4, color="gray", label=f"Выборочная статистика КС, K={K}")

c = d03[-1] * np.sqrt(grid[-1])

ax.loglog(grid, c / np.sqrt(grid), "k--", lw=1, label=r"$C/\sqrt{n}$ (наклон −1/2)")

ax.axhline(1.358 / np.sqrt(K), color="r", ls=":", label="Крит. значение КС, α=0,05")

ax.set_xlabel("n"); ax.set_ylabel(r"$\sup_z |F_{Z_n}(z)-\Phi(z)|$")

ax.set_title("Скорость сходимости в ЦПТ"); ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]

ax.loglog(grid, d03c, "o-", ms=3, label="Bin(5; 0,3), с поправкой")

ax.loglog(grid, d05c, "s-", ms=3, label="Bin(5; 0,5), с поправкой")

ax.loglog(grid, d03, "o:", ms=2, alpha=0.5, label="Bin(5; 0,3), без поправки")

ax.loglog(grid, d05, "s:", ms=2, alpha=0.5, label="Bin(5; 0,5), без поправки")

ax.set_xlabel("n"); ax.set_ylabel("максимальное отклонение")

ax.set_title("Несимметричный и симметричный случаи"); ax.legend(fontsize=8)

ax.grid(True, which="both", alpha=0.3)

fig.tight_layout(); fig.savefig("figs/fig3_rate.png"); plt.close(fig)

# ---------- 5. Сравнение с Bin(5, 0.5): критерии ----------

rng5 = np.random.default_rng(SEED + 1)

rows5 = []

for n in NS:

    s = sums(rng5, n, K, p=0.5)

    st, pv, df = chi2_test(s, n, p=0.5)

    rows5.append(dict(n=n, chi2=st, chi2_p=pv, chi2_df=df,

                      exact_D=exact_distance(n, p=0.5),

                      exact_Dc=exact_distance(n, p=0.5, corrected=True),

                      exact_Dc03=exact_distance(n, corrected=True)))

res["sym"] = rows5

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

for ax, p in zip(axes, [0.3, 0.5]):

    n = 5

    s = sums(np.random.default_rng(7), n, K, p=p)

    z = z_of(s, n, p=p)

    sd = np.sqrt(M * p * (1 - p) * n)

    edges = (np.arange(s.min() - 0.5, s.max() + 1.5) - M * p * n) / sd

    ax.hist(z, bins=edges, density=True, color="#7aa6d6", edgecolor="white")

    ax.plot(xx, stats.norm.pdf(xx), "r-", lw=1.5)

    ax.set_xlim(-4, 4); ax.set_title(f"Bin(5; {str(p).replace('.', ',')}), n = 5, асимметрия = {stats.skew(z):.2f}".replace("0.", "0,"))

    ax.set_xlabel("$z$"); ax.set_ylabel("плотность")

fig.suptitle("Влияние асимметрии слагаемого (n = 5)")

fig.tight_layout(); fig.savefig("figs/fig4_sym.png"); plt.close(fig)

# ---------- 6. Зависимость «порогового n» от K ----------

# Для каждого K повторяем критерий R раз и считаем долю отвержений (мощность).

ngrid = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]

R = 200

power = {}

rngp = np.random.default_rng(SEED + 2)

for KK in [1_000, 10_000, 100_000]:

    pw = []

    for n in ngrid:

        rej = 0

        for _ in range(R):

            # S_n ~ Bin(5n, 0.3) точно — используем это для скорости

            s = rngp.binomial(M * n, P, size=KK)

            rej += chi2_test(s, n)[1] < ALPHA

        pw.append(rej / R)

    power[KK] = pw

res["power"] = dict(ngrid=ngrid, R=R, **{str(k): v for k, v in power.items()})

fig, ax = plt.subplots(figsize=(7, 4))

for KK, mk in zip(power, ["o", "s", "^"]):

    ax.semilogx(ngrid, power[KK], mk + "-", label=f"K = {KK:,}".replace(",", " "))

ax.axhline(ALPHA, color="gray", ls="--", label="α = 0,05")

ax.set_xlabel("n"); ax.set_ylabel("доля отвержений $H_0$")

ax.set_title(f"Доля отвержений нормальности критерием χ² ({R} повторов)")

ax.legend(); ax.grid(True, which="both", alpha=0.3)

fig.tight_layout(); fig.savefig("figs/fig5_power.png"); plt.close(fig)

with open("results.json", "w", encoding="utf-8") as f:

    json.dump(res, f, ensure_ascii=False, indent=1, default=float)

print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
