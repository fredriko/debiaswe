import csv
from pathlib import Path
from typing import Set, Tuple, List


def compare_analogy_pair_lists(pair_file_1: str, pair_file_2: str, comparison_value: str) -> Tuple[
    int, int, int, List[str], List[str], List[str]]:
    """
    1) number of pairs that are marked comparison_value in both files
    2) number of pairs marked comparison_value in first file, but not second
    3) number of pairs marked comparison_value in second file, but not first
    4) list of all pairs agreed on (corresponding to 1)
    5) list of all pairs for (2)
    6) list of all pairs for (3)
    """
    set_1 = read_csv(pair_file_1, comparison_value)
    set_2 = read_csv(pair_file_2, comparison_value)
    common_pairs = sorted(list(set_1.intersection(set_2)))
    only_in_1 = sorted(list(set_1.difference(set_2)))
    only_in_2 = sorted(list(set_2.difference(set_1)))
    return len(common_pairs), len(only_in_1), len(only_in_2), common_pairs, only_in_1, only_in_2


def read_csv(file_name: str, comp_value: str) -> Set[str]:
    result: Set[str] = set({})
    with open(file_name) as fh:
        reader = csv.reader(fh)
        for row in reader:
            if comp_value is not None:
                if row[2] == comp_value:
                    result.add(row[0])
            else:
                if len(row[0].strip()) > 0:
                    result.add(row[0])
    return result


def print_results(model_name: str, comp_val: str,
                  results: Tuple[int, int, int, List[str], List[str], List[str]]) -> None:
    print(f"** {model_name}")
    print(f"Number of common pairs marked '{comp_val}': {results[0]}")
    print(f"Fredrik found {results[0] + results[1]} pairs")
    print(f"Magnus found {results[0] + results[2]} pairs")
    print("")


def annotated_analogy_pairs_comparison():
    root = Path(Path(__file__).parent, Path("annotated-analogy-pairs"))
    fredrik = Path(root, "fredrik")
    magnus = Path(root, "magnus")

    w2v = "analogy-pairs-hon-han-word2vec.csv"
    w2v_debiased = "analogy-pairs-hon-han-debiased-word2vec.csv"
    ftt = "analogy-pairs-hon-han-fasttext.csv"
    ftt_debiased = "analogy-pairs-hon-han-debiased-fasttext.csv"

    comp_val = "?"

    w2v_results = compare_analogy_pair_lists(Path(fredrik, w2v), Path(magnus, w2v), comp_val)
    w2v_debiased_results = compare_analogy_pair_lists(Path(fredrik, w2v_debiased), Path(magnus, w2v_debiased), comp_val)
    ftt_results = compare_analogy_pair_lists(Path(fredrik, ftt), Path(magnus, ftt), comp_val)
    ftt_debiased_results = compare_analogy_pair_lists(Path(fredrik, ftt_debiased), Path(magnus, ftt_debiased), comp_val)

    results = [
        ("word2vec original model", w2v_results),
        ("word2vec debiased model", w2v_debiased_results),
        ("fastText original model", ftt_results),
        ("fastText debiased model", ftt_debiased_results)]

    print("Comparing annotated analogy pairs")
    for result in results:
        print_results(result[0], comp_val, result[1])


def overlap_between_models():
    root = Path(Path(__file__).parent, Path("annotated-analogy-pairs"))
    fredrik = Path(root, "fredrik")

    w2v = "analogy-pairs-hon-han-word2vec.csv"
    w2v_debiased = "analogy-pairs-hon-han-debiased-word2vec.csv"
    ftt = "analogy-pairs-hon-han-fasttext.csv"
    ftt_debiased = "analogy-pairs-hon-han-debiased-fasttext.csv"

    w2v_set = read_csv(Path(fredrik, w2v), None)
    w2v_d_set = read_csv(Path(fredrik, w2v_debiased), None)

    print(f"Number of pairs common in top 150 of original and debiased word2vec model {len(w2v_set.intersection(w2v_d_set))}")

    ftt_set = read_csv(Path(fredrik, ftt), None)
    ftt_d_set = read_csv(Path(fredrik, ftt_debiased), None)

    print(f"Number of pairs common in top 150 of original and debiased fastText model: {len(ftt_set.intersection(ftt_d_set))}")


if __name__ == "__main__":
    overlap_between_models()
