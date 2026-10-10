import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Порядок веществ: H2, O2, H2O, CO, CO2
INITIAL = np.array([6.0, 5.0, 0.0, 4.0, 0.0])

K_WATER = 0.002
K_CO = 0.001

START_TIME = 0.0
END_TIME = 100.0
NUMBER_OF_POINTS = 1001

def derivatives(t, y):
    H2, O2, H2O, CO, CO2 = y

    # Защита от очень малых отрицательных значений
    # при численном интегрировании
    H2 = max(H2, 0.0)
    O2 = max(O2, 0.0)
    CO = max(CO, 0.0)

    # Условные скорости реакций
    v_water = K_WATER * H2**2 * O2
    v_co = K_CO * CO**2 * O2

    return np.array([
        -2 * v_water,
        -v_water - v_co,
        2 * v_water,
        -2 * v_co,
        2 * v_co,
    ])

def atom_counts(y):
    """Возвращает число атомов H, O и C."""
    H2, O2, H2O, CO, CO2 = y

    return np.array([
        2 * H2 + 2 * H2O,             # H
        2 * O2 + H2O + CO + 2 * CO2,  # O
        CO + CO2,                     # C
    ])

def main():
    times = np.linspace(
        START_TIME,
        END_TIME,
        NUMBER_OF_POINTS,
    )

    result = solve_ivp(
        derivatives,
        (START_TIME, END_TIME),
        INITIAL,
        t_eval=times,
        rtol=1e-9,
        atol=1e-11,
    )

    if not result.success:
        raise RuntimeError(result.message)

    # Проверяем каждый сохранённый временной шаг
    initial_atoms = atom_counts(INITIAL)

    atom_history = np.array([
        atom_counts(state)
        for state in result.y.T
    ])

    deviations = atom_history - initial_atoms
    max_deviation = np.max(np.abs(deviations), axis=0)

    # Проверяем, не появились ли отрицательные количества
    min_amounts = np.min(result.y, axis=1)

    print("ПРОВЕРКА СОХРАНЕНИЯ АТОМОВ")
    print(f"Число временных точек: {len(result.t)}")
    print(f"Интервал: {START_TIME}–{END_TIME}")

    print("\nАтомы в начале:")
    print(f"H: {initial_atoms[0]:.10f}")
    print(f"O: {initial_atoms[1]:.10f}")
    print(f"C: {initial_atoms[2]:.10f}")

    print("\nМаксимальное отклонение за весь расчёт:")
    print(f"H: {max_deviation[0]:.3e}")
    print(f"O: {max_deviation[1]:.3e}")
    print(f"C: {max_deviation[2]:.3e}")

    print("\nМинимальное количество каждого вещества:")
    for name, value in zip(
        ["H2", "O2", "H2O", "CO", "CO2"],
        min_amounts,
    ):
        print(f"{name}: {value:.3e}")

    tolerance = 1e-6
    balance_ok = np.all(max_deviation < tolerance)
    nonnegative_ok = np.all(min_amounts >= -tolerance)

    print("\nИТОГ:")
    print(
        "Баланс атомов:",
        "OK" if balance_ok else "ОШИБКА",
    )
    print(
        "Неотрицательность:",
        "OK" if nonnegative_ok else "ОШИБКА",
    )

    # Сохраняем историю состояний и баланса
    data = pd.DataFrame(
        result.y.T,
        columns=["H2", "O2", "H2O", "CO", "CO2"],
    )
    data.insert(0, "Время", result.t)
    data["Атомы_H"] = atom_history[:, 0]
    data["Атомы_O"] = atom_history[:, 1]
    data["Атомы_C"] = atom_history[:, 2]

    data.to_csv(
        "conservation_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not balance_ok or not nonnegative_ok:
        raise RuntimeError(
            "Проверка не пройдена. Изучи отклонения выше."
        )

if __name__ == "__main__":
    main()