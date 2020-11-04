
def accuracy(y: list, y_hat: list):

    mask = y == y_hat

    correct = sum(mask)

    percent = correct / len(y)

    return percent
