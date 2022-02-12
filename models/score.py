import numpy as np
import nltk
from nltk.metrics import distance
from typing import Set
import editdistance
import numpy as np
import torch
from torch import Tensor



def exact_match_score(preds, targets, ignore_indices: Set[int], *args):
    """Computes exact match scores.
    Args:
        preds: list of list of tokens (one ref)
        targets: list of list of tokens (one hypothesis)
    Returns:
       (float) 1 is perfect
    """
    N = preds.shape[0]
    exactMatch = 0.0
    for i in range(N):
        pred = [token for token in preds[i].tolist() if token not in ignore_indices]
        target = [token for token in targets[i].tolist() if token not in ignore_indices]
        if np.array_equal(pred, target):
            exactMatch += 1

    return exactMatch / N




def edit_distance(preds, targets, ignore_indices: Set[int], *args):
    """Computes Levenshtein distance between two sequences.
    Args:
        preds: list of list of token (one hypothesis)
        targets: list of list of token (one hypothesis)
    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)
    """
    d_leven, len_tot = 0, 0
    N = preds.shape[0]
    for i in range(N):
        pred = [token for token in preds[i].tolist() if token not in ignore_indices]
        target = [token for token in targets[i].tolist() if token not in ignore_indices]
        d_leven += editdistance.distance(pred, target)
        len_tot += float(max(len(pred), len(target)))
        # print(editdistance.distance(pred, target), float(max(len(pred), len(target))))
    return 1. - d_leven / len_tot