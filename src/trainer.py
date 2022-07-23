import os

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

    wandb.login(key=os.environ['WANDB_KEY'])

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

    training_corpus_path = os.path.join(os.environ['PROJECT_PATH'], config['train_data_path'])
    val_corr_data_path = os.path.join(os.environ['PROJECT_PATH'], config['val_corr_test_data_path'])
    # TODO Read word analogy dataset

    # read train data
    lenta_corpus_df = pd.read_csv(training_corpus_path).sample(n=100, random_state=42)
    corpus = lenta_corpus_df['text'].tolist()

    # read val data
    validation_corr_df = pd.read_csv(val_corr_data_path, sep='\t')
    # TODO Read analogy data

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
            steps_counter = 0
            epoch_loss = 0
            for idx in tqdm(range(len(w2v_dataset))):
                sample = w2v_dataset[idx]
                window_loss = 0
                for pos_pair, neg_pair in (zip(sample['original'], sample['hard_negs'])):
                    pos_central, pos_context = pos_pair
                    neg_central, neg_context = neg_pair

                    loss = w2v_model.forward(pos_central, pos_context, neg_central, neg_context)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    window_loss += loss.item()
                    steps_counter += 1

                    epoch_loss += window_loss

                if idx % 100 == 0:
                    corr_score_dict = evaluator.measure_words_correlation(corr_test_df=validation_corr_df,
                                                                          model_word_matrix=w2v_model.central_embeddings,
                                                                          word2idx_dict=w2v_dataset.word2idx)
                    corr_score, p_val = corr_score_dict['pearson score'], corr_score_dict['p-val']

                    # TODO: Score word analogy dataset

                    wandb.log({"corr_score": corr_score, "p_val": p_val, "current_step_loss": window_loss})

            epoch_avg_loss = epoch_loss / steps_counter
            wandb.log({"epoch_sample_loss": epoch_avg_loss})

    except KeyboardInterrupt:
        hdd_dump_path = utils.dump_embeddings_to_hdd(w2v_model.central_embeddings)
        hdd_size = utils.eval_data_size(hdd_dump_path)
        wandb.log({"embeddings_size": hdd_size})
        print(f'Dumped embeds to disk - total size for dim {model_dim} '
              f'with vocab size {len(w2v_dataset.word2idx)} == {hdd_size}')

    hdd_dump_path = utils.dump_embeddings_to_hdd(w2v_model.central_embeddings)
    hdd_size = utils.eval_data_size(hdd_dump_path)
    wandb.log({"embeddings_size": hdd_size})
    print(
        f'Dumped embeds to disk - total size for dim {model_dim} '
        f'with vocab size {len(w2v_dataset.word2idx)} == {hdd_size}')


