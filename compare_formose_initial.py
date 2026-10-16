import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

SUBSTANCES = ["CH2O", "C2H4O2", "C3H6O3", "C4H8O4"]

K1 = 0.002
K2 = 0.01
K3 = 0.005

INITIAL_G_VALUES = [0.0, 0.1, 1.0]
TIME_END = 100.0

def derivatives(t, y):
    F, G, T3, T4 = np.maximum(y, 0.0)

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
    times = np.linspace(0, TIME_END, 501)
    all_results = []
    summary = []

    for initial_g in INITIAL_G_VALUES:
        initial = np.array([10.0, initial_g, 0.0, 0.0])

        result = solve_ivp(
            derivatives,
            (0, TIME_END),
            initial,
            t_eval=times,
            rtol=1e-9,
            atol=1e-11,
        )

        if not result.success:
            raise RuntimeError(result.message)

        carbon_history = np.array([
            carbon_count(state) for state in result.y.T
        ])
        initial_carbon = carbon_count(initial)
        max_error = np.max(
            np.abs(carbon_history - initial_carbon)
        )

        print(f"\nНачальное количество C2H4O2: {initial_g}")
        print("Конечное состояние:")
        for name, value in zip(SUBSTANCES, result.y[:, -1]):
            print(f"  {name}: {value:.6f}")

        print(f"Начальный углерод: {initial_carbon:.8f}")
        print(f"Конечный углерод:  {carbon_history[-1]:.8f}")
        print(f"Максимальная ошибка баланса: {max_error:.3e}")

        if max_error > 1e-6:
            raise RuntimeError("Нарушен баланс углерода!")

        for i, name in enumerate(SUBSTANCES):
            all_results.append(pd.DataFrame({
                "Время": result.t,
                "Начальное C2H4O2": initial_g,
                "Вещество": name,
                "Количество": result.y[i],
            }))

        summary.append({
            "Начальное C2H4O2": initial_g,
            **{
                name: result.y[i, -1]
                for i, name in enumerate(SUBSTANCES)
            },
            "Начальный углерод": initial_carbon,
            "Конечный углерод": carbon_history[-1],
            "Максимальная ошибка": max_error,
        })

    pd.concat(all_results, ignore_index=True).to_csv(
        "formose_initial_timeseries.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        "formose_initial_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, initial_g in zip(axes, INITIAL_G_VALUES):
        subset = pd.concat(all_results, ignore_index=True)
        subset = subset[subset["Начальное C2H4O2"] == initial_g]

        for name in SUBSTANCES:
            line = subset[subset["Вещество"] == name]
            ax.plot(line["Время"], line["Количество"], label=name)

        ax.set_title(f"Начальное C2H4O2 = {initial_g}")
        ax.set_xlabel("Время (условные единицы)")
        ax.set_ylabel("Количество вещества")
        ax.grid(True, alpha=0.3)

    axes[-1].legend()
    fig.suptitle("Влияние начальных условий на модель")
    fig.tight_layout()
    fig.savefig("formose_initial_comparison.png", dpi=150)
    plt.show()

    print("\nСводка сохранена в formose_initial_summary.csv")
    print("Данные сохранены в formose_initial_timeseries.csv")
    print("График сохранён в formose_initial_comparison.png")

if __name__ == "__main__":
    main()