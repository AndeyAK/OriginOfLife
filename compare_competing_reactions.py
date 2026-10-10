import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Порядок веществ: H2, O2, H2O, CO, CO2
INITIAL = [6, 5, 0, 4, 0]

# Скорость образования воды фиксирована
K_WATER = 0.002

# Меняем только скорость окисления CO
CO_SCENARIOS = {
    "CO медленно (×0.5)": 0.5,
    "CO базово (×1)": 1.0,
    "CO быстро (×2)": 2.0,
}

BASE_K_CO = 0.001
SUBSTANCES = ["H2", "O2", "H2O", "CO", "CO2"]

def derivatives(t, y, k_co):
    H2, O2, H2O, CO, CO2 = np.maximum(y, 0)

    # Скорости двух реакций
    v_water = K_WATER * H2**2 * O2
    v_co = k_co * CO**2 * O2

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
        2 * H2 + 2 * H2O,              # H
        2 * O2 + H2O + CO + 2 * CO2,   # O
        CO + CO2,                      # C
    ])

def simulate(k_co):
    times = np.linspace(0, 100, 501)

    result = solve_ivp(
        lambda t, y: derivatives(t, y, k_co),
        (0, 100),
        INITIAL,
        t_eval=times,
        rtol=1e-8,
        atol=1e-10,
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result

def main():
    results = {}
    summary_rows = []

    initial_atoms = atom_counts(np.array(INITIAL))

    for name, factor in CO_SCENARIOS.items():
        k_co = BASE_K_CO * factor
        result = simulate(k_co)
        results[name] = result

        final = result.y[:, -1]
        final_atoms = atom_counts(final)
        balanced = np.allclose(
            initial_atoms, final_atoms, atol=1e-6
        )

        summary_rows.append({
            "Сценарий": name,
            "k_CO": k_co,
            "H2": final[0],
            "O2": final[1],
            "H2O": final[2],
            "CO": final[3],
            "CO2": final[4],
            "Баланс атомов": "OK" if balanced else "ОШИБКА",
        })

        print(f"\n{name}")
        print("Атомы в начале:", initial_atoms)
        print("Атомы в конце: ", final_atoms)
        print("Баланс:", "OK" if balanced else "ОШИБКА")

    summary = pd.DataFrame(summary_rows)

    print("\nИТОГОВЫЙ СОСТАВ ПОСЛЕ 100 ЕДИНИЦ ВРЕМЕНИ")
    print(summary.round(4).to_string(index=False))

    summary.to_csv(
        "competing_reactions_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Сравниваем динамику во всех трёх сценариях
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

    axes[0].set_ylabel("Количество вещества")
    axes[-1].legend(loc="center right")

    fig.suptitle(
        "Как скорость окисления CO влияет на систему"
    )
    fig.tight_layout()
    fig.savefig(
        "competing_reactions.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

if __name__ == "__main__":
    main()