import pickle
import os

def dump_embeddings_to_hdd(model_embeddings):
    f = open('model_embeddings', 'wb')
    pickle.dump(model_embeddings, f)
    path = os.path.realpath(f.name)
    f.close()
    return path

def eval_data_size(path):
    f = open(path, 'rb')
    size = os.fstat(f.fileno()).st_size
    f.close()
    return size