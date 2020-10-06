import torch
import numpy as np

def show_results(show_chars, similarities, char_to_idx, idx_to_char, top_chars=10):
    """
    Receives a list of chars that need to be presented. Converts them to index and looks
    up the 5 most similar chars in the similarities matrix.
    """
    head = "Char\t\t\t Most similar to\n" + '-'*80
    fmt = "{char:s}{space: <4}\t\t {similar:s}"
    print('Overview of some char similarities in descending order.')
    print('-'*55)
    print(head)

    char_idx = [char_to_idx[char] for char in show_chars]
    for idx in char_idx:
        most_similar = similarities[idx,:].topk(top_chars).indices.cpu().numpy()
        test = most_similar.tolist()
        test = [idx_to_char[i] for i in test]
        print(fmt.format(char=idx_to_char[idx], space=' ', similar=str(", ".join(test))))
