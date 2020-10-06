# Deep Learning for NLP 2020
Here we do a project on language identification for the UvA master's course DL4NLP.

# Code and Files

Each folder contains a model used for training and analysis in the paper.

That is, src/baseline contains the BOC model, while src/lstm and src/gpt are the autoregressive models.

We had additionally experimented with a Skipgram model (src/cv2) to obtain semantic representations for the characters. Although we trained this successfully, it did not provide a significant improvement on our models.

If the data is in place (e.g., src/lstm/data/wili-2018 exists), a model can be trained by running, e.g., train python /src/lstm/train.py --batch_size 2048 --hidden_dim 512, similarly for GPT. See train.py -h or /src/lstm/utils/config.py for the configuration details.

We expanded on the repo https://github.com/graykode/gpt-2-Pytorch for the GPT implementation.
