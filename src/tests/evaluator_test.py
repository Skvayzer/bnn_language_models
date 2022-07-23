import unittest

import pandas as pd
import yaml

from src.skip_gram_hard_neg import SkipGramModelHardNeg
from src import evaluator


class TestEvaluator(unittest.TestCase):

    @classmethod
    def test_measure_words_correlation(cls):
        config = yaml.load(
            open("/Users/somov-od/Documents/projects/AIRI school/bnn_language_models/src/config.yaml", 'r'),
        Loader=yaml.Loader)
        validation_corr_test_df = config['val_corr_test_data_path']

        validation_corr_df = pd.read_csv(validation_corr_test_df, sep='\t')
        total_words = set(validation_corr_df['# Word1'].to_list() + validation_corr_df['Word2'].to_list())
        test_word2idx_dict = {word: idx for idx, word in enumerate(total_words)}

        model = SkipGramModelHardNeg(vocab_size=70460, embed_dim=15, dev='cpu')

        evaluation_data = evaluator.measure_words_correlation(corr_test_df=validation_corr_df,
                                                              model_word_matrix=model.central_embeddings,
                                                              word2idx_dict=test_word2idx_dict)
        print(evaluation_data)

        assert list(evaluation_data.keys()) == ["pearson score", "p-val", "oov pairs"]

    @classmethod
    def test_measure_word_analogy_accuracy(cls):
        pass


if __name__ == '__main__':
    unittest.main()
