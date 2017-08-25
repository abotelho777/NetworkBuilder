import numpy as np
from sklearn.metrics import f1_score


def auc(actual, predicted, average_over_labels=True):
    assert len(actual) == len(predicted)

    ac = np.array(actual).reshape((len(actual),-1))
    pr = np.array(predicted).reshape((len(predicted),-1))

    # print(ac.shape)

    label_auc = []
    for i in range(ac.shape[-1]):
        a = np.array(ac[:,i])
        p = np.array(pr[:,i])

        pos = np.argwhere(a[:] == np.max(a))
        neg = np.argwhere(a[:] != np.max(a))

        eq = np.ones((np.alen(neg), np.alen(pos))) * p[pos].T == np.ones((np.alen(neg), np.alen(pos))) * p[neg]
        geq = np.array(np.ones((np.alen(neg), np.alen(pos))) *
                       p[pos].T >= np.ones((np.alen(neg), np.alen(pos))) * p[neg],
                       dtype=np.float32)
        geq[eq[:, :] == True] = 0.5
        label_auc.append(np.mean(geq))

    if average_over_labels:
        return np.mean(label_auc)
    else:
        return label_auc


def f1(actual, predicted):
    return f1_score(np.array(actual), np.round(predicted))


if __name__ == "__main__":

    x = np.array([.2, .3, .5, .3, .6])
    y = np.array([0, 1, 0, 0, 1])

    print(auc(y,x))