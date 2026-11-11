# Сначала подключаем библиотеки и задаём общие настройки.
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SEED = 2026
rng = np.random.default_rng(SEED)
DAYS = 365
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False,
})
def save(fig, name):
    fig.savefig(OUT / name)

# 1. Точная формула и приближение
def p_exact(n, N=DAYS):
    """P(хотя бы у двоих из n совпали дни рождения), N дней в году."""
    if n > N:
        return 1.0
    q = 1.0
    for k in range(n):
        q *= (N - k) / N
    return 1.0 - q

def p_approx(n, N=DAYS):
    """Приближение: 1 - exp(-n(n-1)/(2N))."""
    return 1.0 - math.exp(-n * (n - 1) / (2 * N))

print("=== 1. Ключевые значения ===")
print(f"{'n':>4} {'точно':>8} {'прибл.':>8}")
for n in (10, 20, 23, 30, 40, 50, 57, 70, 100):
    print(f"{n:>4} {p_exact(n):8.4f} {p_approx(n):8.4f}")
n_half = next(n for n in range(1, 366) if p_exact(n) >= 0.5)
n_99 = next(n for n in range(1, 366) if p_exact(n) >= 0.99)
print(f"Минимальное n с p >= 0.5: {n_half};  с p >= 0.99: {n_99}")



# 2. Проверка методом Монте-Карло

def has_match(groups):
    """groups: массив (испытания, n). True, если в строке есть повтор."""
    s = np.sort(groups, axis=1)
    return (np.diff(s, axis=1) == 0).any(axis=1)

TRIALS = 10_000
ns = np.arange(2, 101)
p_mc = np.array([has_match(rng.integers(0, DAYS, (TRIALS, n))).mean() for n in ns])
p_th = np.array([p_exact(n) for n in ns])
p_ap = np.array([p_approx(n) for n in ns])
z = stats.norm.ppf(0.975)
# 95% доверительный интервал Уилсона: в отличие от p ± z·sqrt(p(1-p)/T)
# он не схлопывается в точку, когда оценка p равна 0 или 1.
center = (p_mc + z**2 / (2 * TRIALS)) / (1 + z**2 / TRIALS)
half = z * np.sqrt(p_mc * (1 - p_mc) / TRIALS + z**2 / (4 * TRIALS**2)) / (1 + z**2 / TRIALS)
lo, hi = center - half, center + half
inside = np.mean((p_th >= lo) & (p_th <= hi))

print("\n=== 2. Монте-Карло (10 000 групп на каждое n) ===")
for n in (10, 23, 50, 70):
    i = n - 2
    print(f"n={n:>3}: МК {p_mc[i]:.4f}, интервал [{lo[i]:.4f}; {hi[i]:.4f}], "
          f"теория {p_th[i]:.4f}")
print(f"Теория внутри 95%-интервала для {inside:.0%} значений n "
      f"(ожидаем около 95%)")
print(f"Макс. ошибка приближения exp: {np.max(np.abs(p_ap - p_th)):.4f}")

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(ns, p_th, color=BLUE, label="Точная формула")
ax.plot(ns, p_ap, color=ORANGE, ls="--", label="Приближение 1 − exp(−n(n−1)/730)")
step = 3
ax.errorbar(ns[::step], p_mc[::step],
            yerr=[p_mc[::step] - lo[::step], hi[::step] - p_mc[::step]], fmt="o", ms=4,
            color=AQUA, ecolor=AQUA, elinewidth=1, capsize=2,
            label="Монте-Карло, 95% интервал")
ax.axhline(0.5, color=INK2, lw=1, ls=":")
ax.axvline(23, color=INK2, lw=1, ls=":")
ax.annotate(f"n = 23: p ≈ {p_exact(23):.3f}", xy=(23, p_exact(23)),
            xytext=(32, 0.32), color=INK,
            arrowprops=dict(arrowstyle="->", color=INK2))
ax.set(xlabel="Число людей в группе, n",
       ylabel="Вероятность совпадения p(n)",
       title="Вероятность, что у двоих совпадёт день рождения",
       xlim=(0, 100), ylim=(0, 1.02))
ax.legend(loc="lower right")
save(fig, "01_p_of_n.png")



#  3. Оценка вероятностей редких событий

rng_rare = np.random.default_rng(SEED + 1)


def wilson(k, T, z=z):
    """95% интервал Уилсона для доли k/T."""
    p = k / T
    c = (p + z**2 / (2 * T)) / (1 + z**2 / T)
    h = z * np.sqrt(p * (1 - p) / T + z**2 / (4 * T**2)) / (1 + z**2 / T)
    return c - h, c + h


def count_events(n, want_match, T, chunk=100_000):
    """Сколько раз из T групп по n человек произошло событие.
    want_match=True — есть совпадение, False — у всех дни разные."""
    k = 0
    for start in range(0, T, chunk):
        m = min(chunk, T - start)
        hit = has_match(rng_rare.integers(0, DAYS, (m, n)))
        k += int(hit.sum() if want_match else (~hit).sum())
    return k


rare_events = [  # (название, n, событие «есть совпадение»?)
    ("совпадение при n = 3", 3, True),
    ("совпадение при n = 5", 5, True),
    ("нет совпадений при n = 60", 60, False),
    ("нет совпадений при n = 70", 70, False),
    ("нет совпадений при n = 80", 80, False),
    ("нет совпадений при n = 90", 90, False),
]
T_LIST = (10_000, 1_000_000)
rare_rows = []
print("\n=== 3. Оценка вероятностей редких событий ===")
print(f"{'событие':<27}{'точно':>10}{'T':>9}{'k':>7}{'оценка':>11}"
      f"{'95%-й интервал':>26}{'отн. ошибка':>13}")
for name, n, want in rare_events:
    p_true = p_exact(n) if want else 1 - p_exact(n)
    for T_ in T_LIST:
        k = count_events(n, want, T_)
        lo_r, hi_r = wilson(k, T_)
        p_hat = k / T_
        rel_err = abs(p_hat - p_true) / p_true
        rel_half = (hi_r - lo_r) / 2 / p_true
        rare_rows.append((name, p_true, T_, k, p_hat, lo_r, hi_r, rel_err, rel_half))
        print(f"{name:<27}{p_true:>10.2e}{T_:>9}{k:>7}{p_hat:>11.2e}"
              f"   [{lo_r:.2e}; {hi_r:.2e}]{rel_err:>12.0%}")
    T_need = z**2 * (1 - p_true) / (0.1**2 * p_true)
    print(f"{'':<27}для точности ±10% нужно T ≈ {T_need:,.0f}".replace(",", " "))
covered = np.mean([lo_ <= p <= hi_ for _, p, _, _, _, lo_, hi_, _, _ in rare_rows])
print(f"Точное значение внутри интервала Уилсона: {covered:.0%} строк")

fig, ax = plt.subplots(figsize=(9, 5.2))
pp = np.logspace(-6, -1.3, 200)
for T_, col in zip(T_LIST, (ORANGE, BLUE)):
    T_txt = f"{T_:,}".replace(",", " ")
    ax.loglog(pp, z * np.sqrt((1 - pp) / (pp * T_)), color=col, lw=2,
              label=f"Теория z·√((1−p)/(pT)), T = {T_txt}")
    pts = [(r[1], r[8]) for r in rare_rows if r[2] == T_]
    ax.loglog([a for a, _ in pts], [b for _, b in pts], "o", color=col, ms=8,
              mec="white", mew=1.5, label=f"Монте-Карло, T = {T_txt}")
ax.axhline(0.1, color=INK2, lw=1, ls=":")
ax.text(1.3e-6, 0.115, "точность ±10%", color=INK2)
ax.set(xlabel="Вероятность события p",
       ylabel="Относительная полуширина 95%-го интервала",
       title="Чем реже событие, тем больше опытов нужно для его оценки")
ax.legend(loc="upper right")
save(fig, "02_rare_events.png")


#  4. Проверка на нашей учебной группе
our_group = [
    (26, 5), (1, 12), (20, 3), (30, 8), (20, 12), (14, 3), (20, 5), (4, 8),
    (14, 6), (9, 1), (4, 8), (18, 1), (29, 4), (4, 3), (24, 11), (13, 3),
    (26, 1), (14, 8), (20, 11), (2, 6), (17, 8), (22, 1), (9, 6),
]
month_len = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
months = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн",
          "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
months_gen = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
              "августа", "сентября", "октября", "ноября", "декабря"]
starts = np.cumsum([0] + month_len[:-1])
# номер дня в году от 0 до 364
group = np.array([starts[m - 1] + d - 1 for d, m in our_group])
cnt = np.bincount(group, minlength=DAYS)
n_our = len(group)
dups = [f"{d + 1} {months_gen[m]}" for m in range(12)
        for d in range(month_len[m]) if cnt[starts[m] + d] > 1]
print("\n=== 4. Наша учебная группа ===")
print(f"Человек в группе: {n_our}; различных дней рождения: {len(set(group))}")
print(f"Совпадения: {', '.join(dups) if dups else 'нет'}")
print(f"Вероятность хотя бы одного совпадения в группе из {n_our}: {p_exact(n_our):.3f}")

fig, ax = plt.subplots(figsize=(11, 4.4))
for m in range(12):
    for d in range(month_len[m]):
        c = cnt[starts[m] + d]
        color = "#f1f0ec" if c == 0 else (BLUE if c == 1 else ORANGE)
        ax.add_patch(plt.Rectangle((d, 11 - m), 0.9, 0.9, color=color, lw=0))
        if c > 1:
            ax.text(d + 0.45, 11 - m + 0.45, str(d + 1), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
ax.set_xlim(-0.2, 31)
ax.set_ylim(-0.2, 12)
ax.set_yticks(np.arange(12) + 0.45, months[::-1])
ax.set_xticks(np.arange(0, 31, 5) + 0.45, [str(i + 1) for i in range(0, 31, 5)])
ax.grid(False)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_aspect("equal")
ax.set_title(f"Наша учебная группа ({n_our} человека): совпадение — {', '.join(dups)}")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE),
                   plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
          labels=["один день рождения", "совпадение"],
          loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2)
save(fig, "03_calendar.png")
print("\nГрафики сохранены в", OUT)

plt.show()
