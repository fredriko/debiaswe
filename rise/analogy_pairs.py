from debiaswe.we import WordEmbedding
from pathlib import Path

if __name__ == "__main__":

    root = Path(Path.home(), Path("data/bias-nlp-2019/data/models/normalized-vocabulary-models")).resolve()

    # Use vocabulary normalized versions of the Swedish word2vec and FastText models. The models are on the
    # same format.
    w2v_model_file = Path(root, Path("filtered-sv-word2vec-model.txt"))
    ftt_model_file = Path(root, Path("filtered-sv-fasttext-model.txt"))

    model = WordEmbedding(str(ftt_model_file), model_type="word2vec", max_size_voc=-1)
    gender_vector = model.diff("hon", "han")
    analogies = model.best_analogies_dist_thresh(gender_vector)
    for (a, b, c) in analogies:
        print(f"{a} - {b} (score: {c})")
