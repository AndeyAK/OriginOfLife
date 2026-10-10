import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Порядок веществ:
# H2, O2, H2O, CO, CO2

# Условные константы скорости
k_water = 0.002
k_co = 0.001

def reaction_rates(y):
    """Рассчитывает скорости двух реакций."""
    H2, O2, H2O, CO, CO2 = np.maximum(y, 0)

    rate_water = k_water * H2**2 * O2
    rate_co = k_co * CO**2 * O2

    return rate_water, rate_co

def derivatives(t, y):
    """Изменение количества каждого вещества со временем."""
    H2, O2, H2O, CO, CO2 = y

    rate_water, rate_co = reaction_rates(y)

    return [
        -2 * rate_water,
        -rate_water - rate_co,
        2 * rate_water,
        -2 * rate_co,
        2 * rate_co,
    ]

def main():
    # Начальное количество веществ в условных единицах
    initial = [6, 5, 0, 4, 0]

    time_points = np.linspace(0, 100, 501)

    result = solve_ivp(
        derivatives,
        (0, 100),
        initial,
        t_eval=time_points,
        rtol=1e-8,
        atol=1e-10,
    )

    if not result.success:
        raise RuntimeError(result.message)

    # Переносим результат в таблицу
    data = pd.DataFrame(
        result.y.T,
        columns=["H2", "O2", "H2O", "CO", "CO2"],
    )
    data.insert(0, "Время", result.t)

    print("Начальное состояние:")
    print(data.iloc[0].round(4).to_string())

    print("\nКонечное состояние:")
    print(data.iloc[-1].round(4).to_string())

    # Проверяем сохранение атомов
    initial_atoms = np.array([
        initial[0] * 2 + initial[2] * 2,
        initial[1] * 2 + initial[2] + initial[3] + initial[4] * 2,
        initial[3] + initial[4],
    ])

    final = result.y[:, -1]
    final_atoms = np.array([
        final[0] * 2 + final[2] * 2,
        final[1] * 2 + final[2] + final[3] + final[4] * 2,
        final[3] + final[4],
    ])

    print("\nАтомы H, O, C:")
    print("В начале:", initial_atoms.round(6))
    print("В конце: ", final_atoms.round(6))

    if np.allclose(initial_atoms, final_atoms, atol=1e-6):
        print("Проверка пройдена: атомы сохраняются.")
    else:
        print("ВНИМАНИЕ: обнаружено нарушение баланса.")

    # Сохраняем результаты
    data.to_csv("reaction_dynamics.csv", index=False)

    # Строим график
    for substance in ["H2", "O2", "H2O", "CO", "CO2"]:
        plt.plot(result.t, np.maximum(result.y[[
            "H2", "O2", "H2O", "CO", "CO2"
        ].index(substance)], 0), label=substance)

    plt.xlabel("Время (условные единицы)")
    plt.ylabel("Количество вещества (условные единицы)")
    plt.title("Динамика химической системы")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reaction_dynamics.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()