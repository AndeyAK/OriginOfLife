import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Вещества: формальдегид, гликольальдегид,
# глицеральдегид, тетроза
SUBSTANCES = ["CH2O", "C2H4O2", "C3H6O3", "C4H8O4"]

# Условные константы скорости
K1 = 0.002
K2 = 0.01
K3 = 0.005

# Начальный состав: формальдегид и небольшое
# количество гликольальдегида
INITIAL = np.array([10.0, 0.1, 0.0, 0.0])

def derivatives(t, y):
    F, G, T3, T4 = np.maximum(y, 0)

    # Упрощённые суммарные превращения:
    # 2 CH2O -> C2H4O2
    # CH2O + C2H4O2 -> C3H6O3
    # CH2O + C3H6O3 -> C4H8O4

    v1 = K1 * F**2
    v2 = K2 * F * G
    v3 = K3 * F * T3

    return [
        -2 * v1 - v2 - v3,
        v1 - v2,
        v2 - v3,
        v3,
    ]

def carbon_count(y):
    F, G, T3, T4 = y
    return F + 2 * G + 3 * T3 + 4 * T4

def main():
    times = np.linspace(0, 100, 501)

    result = solve_ivp(
        derivatives,
        (0, 100),
        INITIAL,
        t_eval=times,
        rtol=1e-9,
        atol=1e-11,
    )

    if not result.success:
        raise RuntimeError(result.message)

    initial_carbon = carbon_count(INITIAL)
    carbon_history = np.array([
        carbon_count(state) for state in result.y.T
    ])

    max_error = np.max(
        np.abs(carbon_history - initial_carbon)
    )

    print("Начальное состояние:")
    print(dict(zip(SUBSTANCES, INITIAL)))

    print("\nКонечное состояние:")
    print(dict(zip(
        SUBSTANCES,
        result.y[:, -1].round(6),
    )))

    print(f"\nУглерод в начале: {initial_carbon:.8f}")
    print(f"Углерод в конце:  {carbon_history[-1]:.8f}")
    print(f"Максимальное отклонение углерода: {max_error:.3e}")

    if max_error < 1e-6:
        print("Баланс углерода: OK")
    else:
        print("Баланс углерода: ОШИБКА")

    data = pd.DataFrame(
        result.y.T,
        columns=SUBSTANCES,
    )
    data.insert(0, "Время", result.t)
    data["Углерод"] = carbon_history

    data.to_csv(
        "formose_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for i, name in enumerate(SUBSTANCES):
        plt.plot(result.t, result.y[i], label=name)

    plt.xlabel("Время (условные единицы)")
    plt.ylabel("Количество вещества")
    plt.title("Упрощённая модель образования сахаров")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("formose_results.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()