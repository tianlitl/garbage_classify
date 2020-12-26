import torch
import numpy as np

def accuracy(predictions, targets):
	correct = []
	for i in range(len(predictions)):
		if torch.equal(predictions.cpu().long()[i], targets.cpu().long()[i]):
			correct.append(1)
		else:
			correct.append(0)
	return sum(correct) / len(predictions)

def Cmatrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat
