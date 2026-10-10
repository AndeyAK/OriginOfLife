import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Условные константы скорости
K_WATER = 0.002
K_CO = 0.001

SUBSTANCES = ["H2", "O2", "H2O", "CO", "CO2"]

# Сравниваем только начальное количество кислорода
SCENARIOS = {
    "A: мало O2": [6, 2, 0, 4, 0],
    "B: исходный O2": [6, 5, 0, 4, 0],
    "C: много O2": [6, 10, 0, 4, 0],
}

def derivatives(t, y):
    H2, O2, H2O, CO, CO2 = np.maximum(y, 0)

    # Условные скорости реакций
    v_water = K_WATER * H2**2 * O2
    v_co = K_CO * CO**2 * O2

    return [
        -2 * v_water,
        -v_water - v_co,
        2 * v_water,
        -2 * v_co,
        2 * v_co,
    ]

def atom_counts(y):
    H2, O2, H2O, CO, CO2 = y

    return np.array([
        2 * H2 + 2 * H2O,          # H
        2 * O2 + H2O + CO + 2 * CO2,  # O
        CO + CO2,                  # C
    ])

def simulate(initial):
    times = np.linspace(0, 100, 501)

    result = solve_ivp(
        derivatives,
        (0, 100),
        initial,
        t_eval=times,
        rtol=1e-8,
        atol=1e-10,
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result

def main():
    results = {}
    rows = []

    for name, initial in SCENARIOS.items():
        result = simulate(initial)
        results[name] = result

        start_atoms = atom_counts(np.array(initial))
        end_atoms = atom_counts(result.y[:, -1])

        balanced = np.allclose(
            start_atoms, end_atoms, atol=1e-6
        )

        rows.append({
            "Сценарий": name,
            "Начальный O2": initial[1],
            "Конечный H2": result.y[0, -1],
            "Конечный O2": result.y[1, -1],
            "Конечная H2O": result.y[2, -1],
            "Конечный CO": result.y[3, -1],
            "Конечный CO2": result.y[4, -1],
            "Баланс атомов": "OK" if balanced else "ОШИБКА",
        })

        print(f"\n{name}")
        print("Атомы в начале:", start_atoms)
        print("Атомы в конце: ", end_atoms)
        print("Баланс:", "OK" if balanced else "ОШИБКА")

    summary = pd.DataFrame(rows)

    print("\nСРАВНЕНИЕ КОНЕЧНЫХ СОСТОЯНИЙ")
    print(summary.round(4).to_string(index=False))

    summary.to_csv(
        "scenario_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Общая таблица временных рядов
    all_data = []

    for name, result in results.items():
        frame = pd.DataFrame(
            result.y.T, columns=SUBSTANCES
        )
        frame.insert(0, "Время", result.t)
        frame.insert(1, "Сценарий", name)
        all_data.append(frame)

    pd.concat(all_data, ignore_index=True).to_csv(
        "scenario_timeseries.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Три панели — по одной на сценарий
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5), sharey=True
    )

    for ax, (name, result) in zip(axes, results.items()):
        for i, substance in enumerate(SUBSTANCES):
            ax.plot(
                result.t,
                result.y[i],
                label=substance,
            )

        ax.set_title(name)
        ax.set_xlabel("Время (условные единицы)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(
        "Количество вещества (условные единицы)"
    )
    axes[-1].legend(loc="center right")

    fig.suptitle(
        "Влияние начального количества кислорода"
    )
    fig.tight_layout()
    fig.savefig(
        "scenario_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

if __name__ == "__main__":
    main()