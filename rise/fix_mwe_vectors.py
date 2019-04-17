from pathlib import Path
import re

if __name__ == "__main__":
    root = Path(Path.home(), Path("Dropbox/RISE/research/bias-nlp-2019/data/models"))
    elmo_model_in = Path(root, Path("bias_elmo.txt"))
    elmo_model_out = Path(root, Path("bias_elmo_mwe.txt"))
    bert_model_in = Path(root, Path("bias_bert.txt"))
    bert_model_out = Path(root, Path("bias_bert_mwe.txt"))

    mwe_pattern = re.compile("^[^\d\-.]+(\s+)[^\d\-.]+.*")

    with open(str(elmo_model_in), "r") as fhi:
        with open(str(elmo_model_out), "w", encoding="utf-8") as fho:
            for line in fhi:
                line = line.strip()
                m = mwe_pattern.match(line)
                if m:
                    line = "_".join(line.split(" ", 1))
                fho.write(f"{line}\n")