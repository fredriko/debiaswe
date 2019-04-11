from pathlib import Path
import platform

if __name__ == "__main__":

    if platform.system() == "Darwin":
        root = Path("/Users/fredriko/Dropbox/RISE/research/bias-nlp-2019/data/models").resolve()
    else:
        root = Path("/data/research/bias-nlp-acl-2019/data/models/").resolve()

    debiased_w2v_model_file = Path(root, Path("sv-word2vec-vectors-nlpl-eu-69/sv-word2vec-model-debiased.txt"))
    debiased_ftt_model_file = Path(root, Path("sv-fasttext-crawl/sv-fasttext-model-debiased.txt"))
    debiased_normalized_w2v_model_file = Path(root, Path("normalized-vocabulary-models/filtered-sv-word2vec-model-debiased.txt"))
    debiased_normalized_ftt_model_file = Path(root, Path("normalized-vocabulary-models/filtered-sv-fasttext-model-debiased.txt"))

    configs = [
        (debiased_w2v_model_file, 3010472, 100),
        (debiased_ftt_model_file, 2000000, 300),
        (debiased_normalized_w2v_model_file, 681676, 100),
        (debiased_normalized_ftt_model_file, 681676, 300)
    ]

    for c in configs:
        print(f"Processing configuration: {c}")
        input_path = c[0]
        vocabulary_size = c[1]
        dimensionality = c[2]
        output_path = Path(input_path.parent, input_path.stem + "-header.txt")
        with open(str(input_path), "r", encoding="utf-8") as fhi:
            with open(str(output_path), "w") as fho:
                fho.write(f"{vocabulary_size} {dimensionality}\n")
                for line in fhi:
                    fho.write(f"{line}")
