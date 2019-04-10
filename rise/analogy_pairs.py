from debiaswe.we import WordEmbedding
from pathlib import Path
import json

if __name__ == "__main__":

    root = Path(Path.home(), Path("data/bias-nlp-2019/data/models/")).resolve()

    w2v_model_file = Path(root, Path("sv-word2vec-vectors-nlpl-eu-69/model.txt"))
    debiased_w2v_model_file = Path(root, Path("sv-word2vec-vectors-nlpl-eu-69/sv-word2vec-model-debiased.txt"))
    ftt_model_file = Path(root, Path("sv-fasttext-crawl/cc.sv.300.bin"))
    debiased_ftt_model_file = Path(root, Path("sv-fasttext-crawl/"))

    configs = {
        (w2v_model_file, "word2vec", Path(root, Path("analogy-pairs-hon-han-word2vec.txt"))),
        (debiased_w2v_model_file, "word2vec", Path(root, Path("analogy-pairs-hon-han-debiased-word2vec.txt"))),
        (ftt_model_file, "fasttext", Path(root, Path("analogy-pairs-hon-han-fasttext.txt"))),
        (debiased_ftt_model_file, "word2vec", Path(root, Path("analogy-pairs-hon-han-debiased-fasttext.txt"))),
    }

    for config in configs:
        print(f"Processing configuration: {config}")
        model = WordEmbedding(str(config[0]), model_type=config[1], max_size_voc=-1)
        gender_vector = model.diff("hon", "han")
        analogies = model.best_analogies_dist_thresh(gender_vector)
        with open(str(config[2]), "w", encoding="utf-8") as fh:
            json.dump(analogies, fh, indent=2)
