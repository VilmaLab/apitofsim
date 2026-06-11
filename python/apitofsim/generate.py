from collections import Counter
from itertools import combinations
from typing import List, Optional, Tuple

from ase import Atoms


def atoms_to_counter(atoms: Atoms) -> Counter[str]:
    """Convert an Atoms object to a Counter of element symbols."""
    return Counter(atoms.get_chemical_symbols())


def generate_common_names(paths):
    result = {}
    level = 0
    while 1:
        for path in paths:
            name = path.stem
            for i in range(level):
                try:
                    name += "_" + path.parents[i].stem
                except IndexError:
                    raise ValueError(
                        f"Failed to generate unique common name for {path}, ran out of disambiguating parent directories"
                    )
            if name in result:
                result.clear()
                level += 1
                break
            result[name] = path
        else:
            break
    return result


def find_combination_triples(
    counters: List[Counter[str]],
    charges: Optional[List[int]],
    allow_neutral_parents: bool = False,
) -> List[Tuple[int, int, int]]:
    """
    Given a list of Atoms objects, find all triples (i, j, k) of indices such
    that atoms_list[i] + atoms_list[j] has exactly the same atoms as
    atoms_list[k] (i.e. i and j are reactants that combine to form product k).

    Returns a list of (reactant_a_index, reactant_b_index, product_index)
    tuples. Each unordered pair {i, j} appears at most once per product k,
    with i < j.
    """
    # Build a lookup from a frozen counter (i.e. a composition signature)
    # to the list of indices that share that composition.
    # This lets us do O(1) product lookups instead of scanning the whole list.
    from collections import defaultdict

    composition_to_indices: dict[frozenset[tuple[str, int]], list[int]] = defaultdict(
        list
    )
    for idx, c in enumerate(counters):
        key = frozenset(c.items())
        composition_to_indices[key].append(idx)

    results: List[Tuple[int, int, int]] = []

    # Enumerate all pairs of potential reactants
    for i, j in combinations(range(len(counters)), 2):
        # The combined composition is the element-wise sum of both counters
        combined = counters[i] + counters[j]
        combined_key = frozenset(combined.items())

        # Check whether any Atoms object in our list matches the combined
        # composition (those would be the products)
        for k in composition_to_indices.get(combined_key, []):
            if charges is not None:
                if (not allow_neutral_parents and charges[k] == 0) or (
                    charges[i] + charges[j] != charges[k]
                ):
                    continue
            results.append((i, j, k))

    return results


def viable_fragmentations(all_atoms, charges, allow_neutral_parents=False):
    counters = []
    for atoms in all_atoms:
        counters.append(atoms_to_counter(atoms))
    return find_combination_triples(counters, charges, allow_neutral_parents)
