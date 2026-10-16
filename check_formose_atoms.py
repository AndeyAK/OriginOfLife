import numpy as np
import pandas as pd

SUBSTANCES = ["CH2O", "C2H4O2", "C3H6O3", "C4H8O4"]

# Число атомов каждого элемента в одной молекуле.
ATOMS = {
    "CH2O":    {"C": 1, "H": 2, "O": 1},
    "C2H4O2":  {"C": 2, "H": 4, "O": 2},
    "C3H6O3":  {"C": 3, "H": 6, "O": 3},
    "C4H8O4":  {"C": 4, "H": 8, "O": 4},
}

INITIAL_G_VALUES = [0.0, 0.1, 1.0]
ELEMENTS = ["C", "H", "O"]
TOLERANCE = 1e-6

def atom_counts(state):
    """Считает общее количество атомов каждого элемента."""
    totals = {element: 0.0 for element in ELEMENTS}

    for substance, amount in zip(SUBSTANCES, state):
        for element in ELEMENTS:
            totals[element] += ATOMS[substance][element] * amount

    return totals

def main():
    data = pd.read_csv("formose_initial_timeseries.csv")

    all_ok = True

    for initial_g in INITIAL_G_VALUES:
        scenario = data[
            data["Начальное C2H4O2"] == initial_g
        ]

        # Восстанавливаем состояние системы для каждого момента времени.
        states = (
            scenario.pivot(
                index="Время",
                columns="Вещество",
                values="Количество",
            )[SUBSTANCES]
        )

        initial_state = states.iloc[0].to_numpy(dtype=float)
        reference = atom_counts(initial_state)

        max_errors = {element: 0.0 for element in ELEMENTS}
        min_amount = float(states.to_numpy().min())

        for _, row in states.iterrows():
            current = atom_counts(row.to_numpy(dtype=float))

            for element in ELEMENTS:
                error = abs(current[element] - reference[element])
                max_errors[element] = max(max_errors[element], error)

        scenario_ok = (
            all(error <= TOLERANCE for error in max_errors.values())
            and min_amount >= -TOLERANCE
        )

        print(f"\nСценарий: начальное C2H4O2 = {initial_g}")
        print("Начальное количество атомов:", reference)

        for element in ELEMENTS:
            print(
                f"Максимальная ошибка {element}: "
                f"{max_errors[element]:.3e}"
            )

        print(f"Минимальное количество вещества: {min_amount:.3e}")
        print("Результат:", "OK" if scenario_ok else "ОШИБКА")

        if not scenario_ok:
            all_ok = False

    print("\nИТОГ:", "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ" if all_ok
          else "ОБНАРУЖЕНО НАРУШЕНИЕ")

    if not all_ok:
        raise SystemExit(1)

if __name__ == "__main__":
    main()