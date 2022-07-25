import numpy as np
import pandas as pd
from torch.nn import CosineSimilarity
from scipy import stats
import torch


def measure_words_correlation(corr_test_df, model_word_matrix, word2idx_dict, dev):
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
            word1_tensor_idx = torch.tensor(word1_idx).to(dev)
            word2_tensor_idx = torch.tensor(word2_idx).to(dev)

            word1_emb = model_word_matrix(word1_tensor_idx).reshape(1, -1)
            word2_emb = model_word_matrix(word2_tensor_idx).reshape(1, -1)

            if dev == 'cpu':
                cosine_similarity = CosineSimilarity(eps=1e6)(word1_emb, word2_emb).detach().numpy()[0]
            else:
                cosine_similarity = CosineSimilarity(eps=1e6)(word1_emb, word2_emb).cpu().detach().numpy()[0]
            simularity_score_array.append(cosine_similarity)

        else:
            oov_word_pair_count += 1
            simularity_score_array.append(0.0)
    spearman_score = stats.spearmanr(gold_similarity, simularity_score_array)
    return {
        "pearson score": spearman_score[0],
        "p-val": spearman_score[1],
        "oov pairs": oov_word_pair_count
    }


def measure_word_analogy_accuracy(analogy_test_df, model_word_matrix, word2idx_dict, dev):
    """
    for each pair in eval analogy test dataset calculate cosine simularity
    :return:
    the accuracy of the word embedding arithmetic (king - man + woman = queen)
    """
    good_cnt = 0
    all_cnt = 0

    for word1, word2, word3, word4 in zip(analogy_test_df['w1'].to_list(),
                                          analogy_test_df['w2'].to_list(),
                                          analogy_test_df['q'].to_list(),
                                          analogy_test_df['a'].to_list()
                                          ):

        word1_idx = word2idx_dict.get(word1)
        word2_idx = word2idx_dict.get(word2)
        word3_idx = word2idx_dict.get(word3)
        word4_idx = word2idx_dict.get(word4)

        if word1_idx is not None and word2_idx is not None and word3_idx is not None and word4_idx is not None:

            all_cnt += 1

            word1_tensor_idx = torch.tensor(word1_idx).to(dev)
            word2_tensor_idx = torch.tensor(word2_idx).to(dev)
            word3_tensor_idx = torch.tensor(word3_idx).to(dev)

            word1_emb = model_word_matrix(word1_tensor_idx).reshape(1, -1)
            word2_emb = model_word_matrix(word2_tensor_idx).reshape(1, -1)
            word3_emb = model_word_matrix(word3_tensor_idx).reshape(1, -1)

            analogue_emb = word2_emb - word1_emb + word3_emb

            cosine_similarity = CosineSimilarity(eps=1e6, dim=1)(model_word_matrix._parameters['weight'],
                                                                 analogue_emb)
            idx_of_max = torch.argmax(cosine_similarity, dim=0).item()
            if idx_of_max == word4_idx:
                good_cnt += 1

    return good_cnt / all_cnt
