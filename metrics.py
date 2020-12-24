import torch

def accuracy(predictions, targets):
    index = torch.nonzero(predictions == targets).squeeze()
    correct = index.shape.numel()
    # for i in range(len(predictions)):
    #     if torch.equal(predictions.cpu().long()[i], targets.cpu().long()[i]):
    #         correct += 1
    return correct / len(predictions)
