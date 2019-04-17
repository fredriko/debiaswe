from pathlib import Path
from collections import Counter
from pprint import pprint

if __name__ == "__main__":
    root = Path(Path.home(), Path("Dropbox/RISE/research/bias-nlp-2019/data/models"))
    elmo_model_bias = Path(root, Path("bias_elmo.txt"))
    elmo_model_debiased = Path(root, Path("debiased_elmo.txt"))
    bert_model_bias = Path(root, Path("bias_bert.txt"))
    bert_model_debiased = Path(root, Path("debias_bert.txt"))

    c = Counter()
    with open(str(bert_model_bias), "r") as fh:
        for line in fh:
            term = line.split(" ", 1)
            c.update({term[0]: 1})

    pprint(c.most_common(100))