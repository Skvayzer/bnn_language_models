import os
import unittest

import pandas as pd
import yaml

from src.skip_gram_hard_neg import SkipGramModelHardNeg
from src import evaluator


class TestEvaluator(unittest.TestCase):

    @classmethod
    def test_measure_words_correlation(cls):
        config = yaml.load(
            open(os.path.join(os.environ["PROJECT_PATH"], "src", "config.yaml"), 'r'),
            Loader=yaml.Loader)
        validation_corr_test_df = os.path.join(os.environ["PROJECT_PATH"], config['val_corr_test_data_path'])

        validation_analogy_df = pd.read_csv(validation_corr_test_df, sep='\t')
        total_words = set(validation_analogy_df['# Word1'].to_list() + validation_analogy_df['Word2'].to_list())
        test_word2idx_dict = {word: idx for idx, word in enumerate(total_words)}

        model = SkipGramModelHardNeg(vocab_size=70460, embed_dim=15, dev='cpu')

        evaluation_data = evaluator.measure_words_correlation(corr_test_df=validation_analogy_df,
                                                              model_word_matrix=model.central_embeddings,
                                                              word2idx_dict=test_word2idx_dict,
                                                              dev='cpu')
        print(evaluation_data)


    @classmethod
    def test_measure_word_analogy_accuracy(cls):
        config = yaml.load(
            open(os.path.join(os.environ["PROJECT_PATH"], "src", "config.yaml"), 'r'),
            Loader=yaml.Loader)
        validation_analogy_test_df = os.path.join(os.environ["PROJECT_PATH"], config['val_analogy_test_data_path'])

        validation_analogy_df = pd.read_csv(validation_analogy_test_df, sep=' ')

        total_words = set(validation_analogy_df['w1'].to_list() + validation_analogy_df['w2'].to_list()
                          + validation_analogy_df['q'].to_list())
        test_word2idx_dict = {word: idx for idx, word in enumerate(total_words)}

        model = SkipGramModelHardNeg(vocab_size=70460, embed_dim=15, dev='cpu')

        evaluation_data = evaluator.measure_word_analogy_accuracy(analogy_test_df=validation_analogy_df,
                                                              model_word_matrix=model.central_embeddings,
                                                              word2idx_dict=test_word2idx_dict,
                                                              dev='cpu')
        print(evaluation_data)



if __name__ == '__main__':
    unittest.main()
