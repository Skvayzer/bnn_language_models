import random

import torch
from torch.utils.data import Dataset


class Word2vecDataset(Dataset):
    def __init__(self, dataset, window_size):
        self.dataset = dataset
        self.word_list = " ".join(dataset).split()
        self.window_size = window_size
        self.corpus_words = list(set(self.word_list))
        self.word2idx = {word: idx for idx, word in enumerate(self.corpus_words)}
        self.word2idx['UNK'] = len(self.corpus_words)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def __getitem__(self, idx):
        result_dict = {"original": [], "hard_negs": []}

        # допустим что мы подготовили такой формат данных - чтобы подходить под формат датасета
        sentence_array = self.dataset[idx].split()
        random_idx = random.choice(range(len(sentence_array)))
        central_word_idx = self.word2idx[sentence_array[random_idx]]
        left_window_start = max(0, random_idx - self.window_size // 2)
        rigth_window_end = min(random_idx + self.window_size // 2 + 1, len(sentence_array))

        for context_word in sentence_array[left_window_start:rigth_window_end]:
            context_word_idx = self.word2idx[context_word]
            if central_word_idx != context_word_idx:
                result_dict['original'].append([central_word_idx, context_word_idx])

        for _ in result_dict['original']:

            if idx != len(self.dataset):
                remaining_samples = self.dataset[:idx] + self.dataset[idx+1:]
            else:
                remaining_samples = self.dataset[:idx]

            central, context = random.sample(remaining_samples, 2)

            random_central_sentence = central.split()
            random_idx = random.choice(range(len(random_central_sentence)))
            central_word_idx = self.word2idx[random_central_sentence[random_idx]]

            random_context_sentence = context.split()
            random_idx = random.choice(range(len(random_context_sentence)))
            context_word_idx = self.word2idx[random_context_sentence[random_idx]]

            result_dict['hard_negs'].append([central_word_idx, context_word_idx])

        return result_dict

    def __len__(self):
        return len(self.dataset)
