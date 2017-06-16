import numpy as np
from sklearn.metrics import f1_score


def auc(actual, predicted):
    assert len(actual) == len(predicted)

    a = np.array(actual)
    p = np.array(predicted)

    pos = np.argwhere(a[:] == np.max(a))
    neg = np.argwhere(a[:] != np.max(a))

    eq = np.ones((np.alen(neg), np.alen(pos))) * p[pos].T == np.ones((np.alen(neg), np.alen(pos))) * p[neg]
    geq = np.array(np.ones((np.alen(neg), np.alen(pos))) *
                   p[pos].T >= np.ones((np.alen(neg), np.alen(pos))) * p[neg],
                   dtype=np.float32)
    geq[eq[:, :] == True] = 0.5
    return np.mean(geq)


def f1(actual, predicted):
    return f1_score(np.array(actual), np.round(predicted))


if __name__ == "__main__":

    x = np.array([.2, .3, .5, .3, .6])
    y = np.array([0, 1, 0, 0, 1])

    print(f1(y,x))