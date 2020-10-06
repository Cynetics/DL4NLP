import argparse

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='chars', help='Kind of input: [chars|bytes]')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--sequence_length', type=int, default=64)

    parser.add_argument('--input_dim', type=int, default=12300)
    parser.add_argument('--embedding_dim', type=int, default=256)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_checkpoint', type=str, default=None)

    parser.add_argument('--data_path', type=str, default="./data/wili-2018/x_train_sub_clean.txt")
    parser.add_argument('--label_path', type=str, default="./data/wili-2018/y_train_clean.txt")

    parser.add_argument('--val_data_path', type=str, default="./data/wili-2018/x_val_sub_clean.txt")
    parser.add_argument('--val_label_path', type=str, default="./data/wili-2018/y_val_clean.txt")

    parser.add_argument('--prediction_type', type=str, default="deterministic")


    args = parser.parse_args()
    print(args)
    with open("config.txt", 'w') as file:
        file.write(str(args))
    return args
