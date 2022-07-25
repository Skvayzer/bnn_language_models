import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SkipGramModelHardNeg(nn.Module):
    def __init__(self, vocab_size, embed_dim, dev):
        super(SkipGramModelHardNeg, self).__init__()
        self.central_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.device = dev

        initrange = 1.0 / embed_dim
        init.uniform_(self.central_embeddings.weight.data, -initrange, initrange)
        init.uniform_(self.context_embeddings.weight.data, -initrange, initrange)

    def forward(self, central_word, context_word, neg_central_word, neg_context_word):

        emb_central = self.central_embeddings(central_word)
        emb_context = self.context_embeddings(context_word)

        neg_emb_central = self.central_embeddings(neg_central_word)
        neg_emb_context = self.context_embeddings(neg_context_word)

        pos_score = torch.sum(torch.mul(emb_central, emb_context), dim=1)
        neg_score = torch.sum(torch.mul(neg_emb_central, neg_emb_context), dim=1)

        pos_score = torch.clamp(pos_score, max=10, min=-10)
        neg_score = torch.clamp(neg_score, max=10, min=-10)

        score = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score))
        return score
