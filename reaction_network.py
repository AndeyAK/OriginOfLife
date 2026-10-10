from collections import Counter

# Состав молекул: количество атомов каждого элемента
MOLECULES = {
    "H2": {"H": 2},
    "O2": {"O": 2},
    "H2O": {"H": 2, "O": 1},
    "CO": {"C": 1, "O": 1},
    "CO2": {"C": 1, "O": 2},
}

# Реакции описаны коэффициентами перед формулами
REACTIONS = [
    {
        "name": "Образование воды",
        "reactants": {"H2": 2, "O2": 1},
        "products": {"H2O": 2},
    },
    {
        "name": "Окисление угарного газа",
        "reactants": {"CO": 2, "O2": 1},
        "products": {"CO2": 2},
    },
]

def atom_count(mixture):
    """Подсчитывает общее число атомов в смеси."""
    atoms = Counter()

    for molecule, quantity in mixture.items():
        for element, count in MOLECULES[molecule].items():
            atoms[element] += count * quantity

    return +atoms

def is_balanced(reaction):
    """Проверяет сохранение атомов в реакции."""
    left = Counter()
    right = Counter()

    for molecule, quantity in reaction["reactants"].items():
        for element, count in MOLECULES[molecule].items():
            left[element] += count * quantity

    for molecule, quantity in reaction["products"].items():
        for element, count in MOLECULES[molecule].items():
            right[element] += count * quantity

    return +left == +right

def max_reaction_extent(mixture, reactants):
    """Определяет, сколько раз можно провести реакцию."""
    possible = [
        mixture.get(molecule, 0) // needed
        for molecule, needed in reactants.items()
    ]
    return min(possible)

def run_reaction(mixture, reaction):
    """Проводит реакцию максимально возможное число раз."""
    extent = max_reaction_extent(
        mixture, reaction["reactants"]
    )

    if extent == 0:
        return 0

    for molecule, quantity in reaction["reactants"].items():
        mixture[molecule] -= quantity * extent

    for molecule, quantity in reaction["products"].items():
        mixture[molecule] = (
            mixture.get(molecule, 0) + quantity * extent
        )

    return extent

def main():
    # Начальное количество молекул
    mixture = {
        "H2": 6,
        "O2": 5,
        "CO": 4,
        "H2O": 0,
        "CO2": 0,
    }

    initial_atoms = atom_count(mixture)

    print("Начальная смесь:")
    print(mixture)
    print("\nНачальное количество атомов:")
    print(dict(initial_atoms))

    print("\nЗапуск реакций:")

    for reaction in REACTIONS:
        if not is_balanced(reaction):
            raise ValueError(
                f"Реакция не сбалансирована: {reaction['name']}"
            )

        extent = run_reaction(mixture, reaction)
        print(f"{reaction['name']}: выполнена {extent} раз(а)")

    final_atoms = atom_count(mixture)

    print("\nСмесь после реакций:")
    print(mixture)
    print("\nКонечное количество атомов:")
    print(dict(final_atoms))

    if initial_atoms == final_atoms:
        print("\nПроверка пройдена: атомы сохранены.")
    else:
        print("\nОШИБКА: количество атомов изменилось!")

if __name__ == "__main__":
    main()