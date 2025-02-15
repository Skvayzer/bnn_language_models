import os
import sys

import torch.optim as optim
import yaml
import pandas as pd
import torch
import wandb
import evaluator
from tqdm import tqdm
import utils

from dataset import Word2vecDataset
from skip_gram_hard_neg import SkipGramModelHardNeg

if __name__ == "__main__":

    if os.environ.get('PROJECT_PATH') is None:
        #path to project
        os.environ['PROJECT_PATH'] = os.getcwd().split('/')[0]
        os.environ['WANDB_KEY'] = ''
        os.environ['WANDB_CONFIG_DIR'] = '/tmp'

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    config = yaml.load(open(os.path.join(os.environ['PROJECT_PATH'], 'src', 'config.yaml'), 'r'), Loader=yaml.Loader)

    model_name = config['model_name']

    learning_rate = float(config['learning_rate'])
    model_dim = int(config['model_dimension'])
    context_window = int(config['contex_window_size'])
    epoch_num = int(config['epoch_num'])
    embeddings_path = os.path.join(os.environ['PROJECT_PATH'], config['embeddings_dump_path'],
                                          f"embs_matrix_{model_name}_{model_dim}.pkl")

    training_corpus_path = os.path.join(os.environ['PROJECT_PATH'], config['train_data_path'])
    val_corr_data_path = os.path.join(os.environ['PROJECT_PATH'], config['val_corr_test_data_path'])
    val_analogy_data_path = os.path.join(os.environ['PROJECT_PATH'], config['val_analogy_test_data_path'])

    # read train data
    lenta_corpus_df = pd.read_csv(training_corpus_path)
    corpus = lenta_corpus_df['text'].tolist()

    # read val data
    validation_corr_df = pd.read_csv(val_corr_data_path, sep='\t')
    validation_analogy_df = pd.read_csv(val_analogy_data_path, sep=' ')
    w2v_dataset = Word2vecDataset(corpus, context_window)

    w2v_model = None
    optimizer = None
    if model_name == 'vanilla':
        w2v_model = SkipGramModelHardNeg(vocab_size=len(w2v_dataset.word2idx), embed_dim=model_dim, dev=dev).to(dev)
        optimizer = optim.Adam(w2v_model.parameters(), lr=learning_rate)
    # TODO: implement BNN model & optimizer
    # elif model_name == 'bnn':
    #     w2v_model = None
    #     optimizer = None

    loss_array = []

    wandb.init(
        project="bnn_lm",
        name=f"experiment_{model_name}_{model_dim}",
        config={
            "learning_rate": learning_rate,
            "model_name": model_name,
            "epochs": epoch_num,
            "window_size": context_window,
            "model_dim": model_dim
        })

    try:
        for epoch_num in range(epoch_num):
            epoch_loss = 0
            for idx in tqdm(range(len(w2v_dataset))):
                sample = w2v_dataset[idx]

                #значит натокнулись на потовряющиеся токены в одном окне
                if len(sample['original']) != 0:
                    pos_pair = torch.tensor(sample['original']).to(dev)
                    neg_pair = torch.tensor(sample['hard_negs']).to(dev)

                    pos_central, pos_context = pos_pair[:, 0], pos_pair[:, 1]
                    neg_central, neg_context = neg_pair[:, 0], neg_pair[:, 1]

                    loss = w2v_model.forward(pos_central, pos_context, neg_central, neg_context).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                if idx % 100 == 0:
                    corr_score_dict = evaluator.measure_words_correlation(corr_test_df=validation_corr_df,
                                                                          model_word_matrix=w2v_model.central_embeddings,
                                                                          word2idx_dict=w2v_dataset.word2idx,
                                                                          dev=dev)
                    corr_score, p_val = corr_score_dict['pearson score'], corr_score_dict['p-val']

                    # TODO: Score word analogy dataset
                    analogy_score = evaluator.measure_word_analogy_accuracy(analogy_test_df=validation_analogy_df,
                                                                          model_word_matrix=w2v_model.central_embeddings,
                                                                          word2idx_dict=w2v_dataset.word2idx,
                                                                          dev=dev)

                    wandb.log({"corr_score": corr_score, "p_val": p_val})
                    wandb.log({"analogy_accuracy": analogy_score})


            #epoch per sample loss
            epoch_avg_per_sample_loss = epoch_loss / len(w2v_dataset)
            wandb.log({"epoch_sample_loss": epoch_avg_per_sample_loss})

    except KeyboardInterrupt:

        hdd_size = utils.get_embeddings_hdd_size(w2v_model.central_embeddings, embeddings_path)
        wandb.log({"embeddings_size": hdd_size})
        print(f'Dumped embeds to disk - total size for dim {model_dim} '
              f'with vocab size {len(w2v_dataset.word2idx)} == {hdd_size}')

    hdd_size = utils.get_embeddings_hdd_size(w2v_model.central_embeddings, embeddings_path)
    wandb.log({"embeddings_size": hdd_size})
    print(f'Dumped embeds to disk - total size for dim {model_dim} '
          f'with vocab size {len(w2v_dataset.word2idx)} == {hdd_size}')
