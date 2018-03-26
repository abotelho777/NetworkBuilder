import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from skll.metrics import kappa as kpa


def auc(actual, predicted, average_over_labels=True):
    assert len(actual) == len(predicted)

    ac = np.array(actual).reshape((len(actual),-1))
    pr = np.array(predicted).reshape((len(predicted),-1))

    na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()

    if len(na) == 0:
        return np.nan

    ac = ac[na]
    pr = pr[na]

    # for i in range(len(ac)):
    #     print(ac[i],'-',pr[i])


    label_auc = []
    for i in range(ac.shape[-1]):
        a = np.array(ac[:,i])
        p = np.array(pr[:,i])

        pos = np.argwhere(a[:] == np.max(a))
        neg = np.argwhere(a[:] != np.max(a))

        # print(pos)
        # print(neg)

        eq = np.ones((np.alen(neg), np.alen(pos))) * p[pos].T == np.ones((np.alen(neg), np.alen(pos))) * p[neg]
        geq = np.array(np.ones((np.alen(neg), np.alen(pos))) *
                       p[pos].T >= np.ones((np.alen(neg), np.alen(pos))) * p[neg],
                       dtype=np.float32)
        geq[eq[:, :] == True] = 0.5
        label_auc.append(np.mean(geq))

    if average_over_labels:
        return np.nanmean(label_auc)
    else:
        return label_auc


def f1(actual, predicted):
    return f1_score(np.array(actual), np.round(predicted))


def rmse(actual, predicted, average_over_labels=True):
    assert len(actual) == len(predicted)

    ac = np.array(actual, dtype=np.float32).reshape((len(actual), -1))
    pr = np.array(predicted, dtype=np.float32).reshape((len(predicted), -1))

    na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()

    if len(na) == 0:
        return np.nan

    ac = ac[na]
    pr = pr[na]

    label_rmse = []
    for i in range(ac.shape[-1]):
        dif = np.array(ac[:, i]) - np.array(pr[:, i])
        sqdif = dif**2
        mse = np.nanmean(sqdif)
        label_rmse.append(np.sqrt(mse))


    if average_over_labels:
        return np.nanmean(label_rmse)
    else:
        return label_rmse


def cohen_kappa(actual, predicted, split=0.5, average_over_labels=True):
    assert len(actual) == len(predicted)

    ac = np.array(actual).reshape((len(actual), -1))
    pr = np.array(predicted).reshape((len(predicted), -1))

    na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()

    if len(na) == 0:
        return np.nan

    ac = np.array(np.array(ac[na]) > split, dtype=np.int32)
    pr = np.array(np.array(pr[na]) > split, dtype=np.int32)

    label_kpa = []
    if hasattr(split, '__iter__'):
        assert len(split) == ac.shape[-1]
    else:
        split = np.ones(ac.shape[1]) * split

    for i in range(ac.shape[-1]):
        label_kpa.append(kpa(np.array(np.array(ac[:, i]) > split[i], dtype=np.int32),
                np.array(np.array(pr[:, i]) > split[i], dtype=np.int32)))

    if average_over_labels:
        return np.nanmean(label_kpa)
    else:
        return label_kpa


def cohen_kappa_multiclass(actual, predicted):
    assert len(actual) == len(predicted)

    ac = np.array(actual).reshape((len(actual), -1))
    pr = np.array(predicted).reshape((len(predicted), -1))

    na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()

    if len(na) == 0:
        return np.nan

    aci = np.argmax(np.array(np.array(ac[na]), dtype=np.int32), axis=1)
    pri = np.argmax(np.array(np.array(pr[na]), dtype=np.float32), axis=1)

    # for i in range(len(aci)):
    #     print(aci[i],'--',pri[i],':',np.array(pr[na])[i])

    return kpa(aci,pri)

# def kappa(actual, predicted, split=0.5):
#     # pred = normalize(list(predicted), method='uniform')
#     return kpa(actual, [p > split for p in predicted])


if __name__ == "__main__":

    x = np.array([.2, .3, .5, .3, .6])
    y = np.array([0, 1, 0, 0, 1])

    print(auc(y,x))