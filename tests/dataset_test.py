import os
import unittest
import yaml
import pandas as pd
from src.dataset import Word2vecDataset
from torch.utils.data import DataLoader


class TestDataset(unittest.TestCase):

    def test_dataset(self):
        config = yaml.load(
            open(os.path.join(os.environ["PROJECT_PATH"], "src", "config.yaml"), 'r'),
            Loader=yaml.Loader)
        training_corpus_path = config['train_data_path']
        context_window = int(config['contex_window_size'])
        lenta_corpus_df = pd.read_csv(os.path.join(os.environ["PROJECT_PATH"], training_corpus_path)).sample(n=100, random_state=42)
        corpus = lenta_corpus_df['text'].tolist()
        w2v_dataset = Word2vecDataset(corpus, context_window)
        print(w2v_dataset[0])


if __name__ == '__main__':
    unittest.main()
