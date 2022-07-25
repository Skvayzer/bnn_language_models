import pickle
import os

def dump_embeddings_to_hdd(model_embeddings):
    # TODO: Implelent pickle dump to HDD -> return path to dump
    f = open('model_embeddings', 'w')
    pickle.dump(model_embeddings, f)
    f.close()
    return os.path.realpath(f.name)

def eval_data_size(path):
    # TODO: Eval HDD embeddings size
    f = open(path, 'r')
    f.close()
    return os.fstat(f.fileno()).st_size