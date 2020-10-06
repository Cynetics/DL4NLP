to train
python train.py --batch_size 2048 --hidden_dim 512

train file only validates on a random subset of 1000
paragraphs

to validate on the whole data
python predict.py --model_checkpoint ./models/LSTM/99_0.958.pt --batch_size 2048 --hidden_dim 512

predict file validates on the whole validation set
