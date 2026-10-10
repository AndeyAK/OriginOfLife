from collections import Counter

# Состав молекул: количество атомов каждого элемента.
MOLECULES = {
    "H2": {"H": 2},
    "O2": {"O": 2},
    "H2O": {"H": 2, "O": 1},
}

def count_atoms(side, molecules):
    """Считает атомы на одной стороне реакции."""
    total = Counter()

    for name, coefficient in side.items():
        if name not in molecules:
            raise ValueError(f"Неизвестная молекула: {name}")

        for element, amount in molecules[name].items():
            total[element] += amount * coefficient

    return total

def check_reaction(reactants, products, molecules):
    """Проверяет сохранение атомов в реакции."""
    left = count_atoms(reactants, molecules)
    right = count_atoms(products, molecules)

    print("Слева:", dict(left))
    print("Справа:", dict(right))

    if left == right:
        print("Реакция сбалансирована по атомам.")
        return True

    print("Баланс атомов нарушен!")
    return False

if __name__ == "__main__":
    # 2 H2 + O2 -> 2 H2O
    check_reaction(
        reactants={"H2": 2, "O2": 1},
        products={"H2O": 2},
        molecules=MOLECULES,
    )