from debiaswe.debias import debias
from debiaswe.we import WordEmbedding

from pathlib import Path
import json
from pprint import pprint

if __name__ == "__main__":

    process_word2vec = False

    root = Path(Path.home(), Path("Dropbox/RISE/research/bias-nlp-2019/data/models"))
    w2v_model_file = Path(root, Path("sv-word2vec-vectors-nlpl-eu-69/model.txt"))
    ftt_model_file = Path(root, Path("sv-fasttext-crawl/cc.sv.300.bin"))
    w2v_model_debiased_file = Path(root, Path("sv-word2vec-model-debiased.txt"))
    ftt_model_debiased_file = Path(root, Path("sv-fasttext-model-debiased.txt"))


    if process_word2vec:
        model_file = w2v_model_file
        model_debiased_file = w2v_model_debiased_file
        model_type = "word2vec"
    else:
        model_file = ftt_model_file
        model_debiased_file = ftt_model_debiased_file
        model_type = "fasttext"


    """
    root = Path(Path.home(), Path("Dropbox/RISE/research/bias-nlp-2019/data/models/normalized-vocabulary-models"))
    w2v_model_file = Path(root, Path("filtered-sv-word2vec-model.txt"))
    ftt_model_file = Path(root, Path("filtered-sv-fasttext-model.txt"))
    w2v_model_debiased_file = Path(root, Path("filtered-sv-word2vec-model-debiased.txt"))
    ftt_model_debiased_file = Path(root, Path("filtered-sv-fasttext-model-debiased.txt"))
    """

    definitional_pairs_file = Path("../data/definitional_pairs_sv.json")
    gender_specific_words_file = Path("../data/gender_specific_full_sv_fasttext.json")

    with open(str(definitional_pairs_file), "r", encoding="utf-8") as fh:
        definitional_pairs = json.load(fh)
    print(f"Got {len(definitional_pairs)} definitional pairs")
    pprint(definitional_pairs)

    with open(str(gender_specific_words_file), "r", encoding="utf-8") as  fh:
        gender_specific_words = json.load(fh)
    num_print = 10
    print(f"Got {len(gender_specific_words)} gender specific words. First {num_print}:")
    pprint(gender_specific_words[:num_print])

    model = WordEmbedding(str(model_file), model_type=model_type, max_size_voc=-1)
    print("Debiasing model...")
    debias(model, gender_specific_words, definitional_pairs, [])
    print(f"Saving debiased model to file {model_debiased_file}")
    model.save(str(model_debiased_file))

