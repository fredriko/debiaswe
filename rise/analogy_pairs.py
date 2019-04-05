from debiaswe.we import WordEmbedding
from pathlib import Path

if __name__ == "__main__":

    root = Path(Path.home(), Path("Dropbox/RISE/research/bias-nlp-2019/data/models")).resolve()

    w2v_model_file = Path(root, Path("sv-word2vec-vectors-nlpl-eu-69/model.txt"))
    ftt_model_file = Path(root, Path("sv-fasttext-crawl/cc.sv.300.bin"))

    model = WordEmbedding(str(ftt_model_file), model_type="fasttext", max_size_voc=50000)
    gender_vector = model.diff("hon", "han")
    analogies = model.best_analogies_dist_thresh(gender_vector)
    for (a, b, c) in analogies:
        print(f"{a}-{b} (score: {c})")
