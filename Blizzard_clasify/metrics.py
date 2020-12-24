import torch


def accuracy(predictions, targets):
    correct = []
    for i in range(len(predictions)):
        if torch.equal(predictions.cpu().long()[i], targets.cpu().long()[i]):
            correct.append(1)
        else:
            correct.append(0)
    return sum(correct) / len(predictions)
