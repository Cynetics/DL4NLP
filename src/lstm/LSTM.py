# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
#from Embedding import Embedder

class KL(nn.Module):
    def __init__(self, divisor=2):
        super().__init__()
        self.c1 = 1.16145124
        self.c2 = -1.50204118
        self.c3 = 0.58629921
        self.divisor = divisor

    def forward(self, log_alpha):
        alpha = torch.exp(log_alpha)
        divergence = -(0.5*log_alpha + self.c1*alpha + self.c2*alpha**2 + self.c3*alpha**3) / self.divisor
        return divergence.mean()

class BayesianDropout(nn.Module):
    """
    We follow Max and Kingma paper "Variational Dropout and the Local Reparameterization Trick" (2015)
    Here alpha should be p / (1-p), as shown in the paper.

    """
    def __init__(self, dropout_rate, weight_dim, cuda=True):
        super().__init__()
        self.alpha = dropout_rate / (1 - dropout_rate)
        self._log_alpha = nn.Parameter(torch.log(torch.ones(weight_dim)*self.alpha), requires_grad=True)

        # init log_alpha params
        sd = 1. / torch.sqrt(torch.tensor(weight_dim).float())
        self._log_alpha.data.uniform_(-sd, sd)

    def forward(self, x, eval=False):
        # Gaussian prior
        if eval: return x
        noise = torch.randn(x.shape).to(torch.device('cuda' if self.cuda else "cpu")) + 1

        # max=1.0 corresponds with a dropout rate of 0.5 (section 3.3)
        #self._log_alpha.data = torch.clamp(self._log_alpha.data, max=1.0)
        self._log_alpha.data = self._log_alpha.data.masked_fill(self._log_alpha.data > 1.0, 0)
        self._log_alpha.data = self._log_alpha.data.masked_fill(self._log_alpha.data < -1.0, 0)
        noise *= torch.exp(self._log_alpha)
        return x * noise


class Embedder(nn.Module):
    def __init__(self, input_dim, embedding_dim, train_embedding=True, load_embeddings=False):
        super().__init__()

        self._embedding = nn.Embedding(input_dim, embedding_dim)
        if load_embeddings:
            weight_path = "./models/embeddings/character_embeddings_256.pt"
            print("Pre-Trained Embeddings Loaded")
            self._embedding.load_state_dict({'weight':torch.load(weight_path)})
        self._embedding.weight.requires_grad = train_embedding

    def forward(self, inputs):
        emb = self._embedding(inputs)
        return emb


class Model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, bidirectional=False, lang_amount=235, load_embeddings=False):
        super().__init__()
        self._bidirectional = 2 if bidirectional else 1
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim
        self._linear_dim = self._hidden_dim*self._num_layers*self._bidirectional
        hidden_layer_dim = 512
        self._embedding = Embedder(input_dim, embedding_dim, train_embedding=True, load_embeddings=load_embeddings)

        self._lstm = nn.LSTM(embedding_dim,
                             hidden_dim,
                             num_layers,
                             bidirectional=bidirectional,
                             batch_first=True, dropout=0.4)

        #self._bayesian1 = BayesianDropout(0.5, 512)
        self._linear1 = nn.Linear(self._linear_dim, hidden_layer_dim)
        #self._bayesian = dropout(0.5, 300, "variational")
        self._bayesian = BayesianDropout(0.5, hidden_layer_dim)
        self._relu = nn.LeakyReLU(0.3)

        self._linear = nn.Linear(hidden_layer_dim, lang_amount)

    def forward(self, inputs, eval=False):

        inputs = self._embedding(inputs)

        # use the individual characters for additional classification?
        _, (h_n, _) = self._lstm(inputs)
        h_n = h_n.permute(1,0,2).contiguous().view(h_n.shape[1], self._linear_dim)
        #h_n = self._bayesian1(h_n)
        h_n = self._linear1(h_n)
        bayesian_dropout = self._bayesian(h_n, eval)
        bayesian_dropout = self._relu(bayesian_dropout)

        #logits = self._linear(h_n)
        logits = self._linear(bayesian_dropout)

        return logits
