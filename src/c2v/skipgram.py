import numpy as np
import torch
from torch import nn

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, char_frequency):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        char_frequency = sorted([(k,v) for k,v in char_frequency.items()], key=lambda x: x[0])
        word_freqs = np.array([freqs/self.vocab_size for (_, freqs) in char_frequency])
        unigram_dist = word_freqs/word_freqs.sum()
        self.noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))

        #Layer
        self.central_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)

        #Uniformly distributed weight initialisation
        nn.init.uniform_(self.central_embedding.weight, -1.0, 1.0)
        nn.init.uniform_(self.context_embedding.weight, -1.0, 1.0)

    def vocabulary_similarities(self):
        """
        Generates a large tensor carrying the cosine similarities between all words in the
        vocabulary.
        """
        emb = self.central_embedding.weight
        #Gives a (embedding_dim, embedding_dim) matrix
        dot = emb @ emb.t()
        #2 denotes euclidean norm, 1 denotes shape as (n_embeddings, vector_dim)
        norm = torch.norm(emb, 2, 1)
        similarities = torch.div(dot,norm)
        similarities = torch.div(similarities, torch.unsqueeze(norm, 0))

        return similarities

    def forward_input(self, data):
        return self.central_embedding(data)

    def forward_output(self, data):
        return self.context_embedding(data)

    def forward_noise(self, batch_size, n_samples=10, device=torch.device("cuda")):
        noise = self.noise_dist.to(device)
        negative_words = torch.multinomial(noise, batch_size * n_samples, replacement=True)

        return self.context_embedding(negative_words).view(batch_size,n_samples,self.embedding_dim)

class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):

        batch_size, embed_size = input_vectors.shape

        # Input vectors should be a batch of column vectors
        input_vectors = input_vectors.view(batch_size, embed_size, 1)

        # Output vectors should be a batch of row vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size)

        # bmm = batch matrix multiplication
        # correct log-sigmoid loss
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()

        # incorrect log-sigmoid loss
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

        # negate and sum correct and noisy log-sigmoid losses
        # return average batch loss
        return -(out_loss + noise_loss).mean()
