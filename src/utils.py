import pickle
import os


def get_embeddings_hdd_size(model_embeddings, path):
    with open(path, 'wb') as f:
        pickle.dump(model_embeddings, f)
        hdd_size = os.fstat(f.fileno()).st_size
    return hdd_size
