import pandas as pd
from torch.nn import CosineSimilarity
from scipy import stats
import torch


def measure_words_correlation(corr_test_df, model_word_matrix, word2idx_dict):
    """
    for each pair in eval corr test dataset calculate cosine simularity
    then spearman test to given values
    :return:
    dict with corr value, p-val, pairs with unknown words
    """
    oov_word_pair_count = 0
    simularity_score_array = []
    gold_similarity = corr_test_df['Average Score'].to_list()

    for word1, word2 in zip(corr_test_df['# Word1'].to_list(), corr_test_df['Word2'].to_list()):
        word1_idx = word2idx_dict.get(word1)
        word2_idx = word2idx_dict.get(word2)

        if word1_idx is not None and word2_idx is not None:
            word1_tensor_idx = torch.tensor(word1_idx)
            word2_tensor_idx = torch.tensor(word2_idx)

            word1_emb = model_word_matrix(word1_tensor_idx).reshape(1,-1)
            word2_emb = model_word_matrix(word2_tensor_idx).reshape(1,-1)

            cosine_similarity = CosineSimilarity(eps=1e6)(word1_emb, word2_emb).detach().numpy()[0]
            simularity_score_array.append(cosine_similarity)

        else:
            oov_word_pair_count += 1
            simularity_score_array.append(0.0)
    pearson = stats.pearsonr(gold_similarity, simularity_score_array)
    return {
        "pearson score": pearson[0],
        "p-val": pearson[1],
        "oov pairs": oov_word_pair_count
    }

def measure_word_analogy_accuracy(analogy_test_df, model, word2idx_dict):
    # TODO: Implement word analogy eval acc
    pass
