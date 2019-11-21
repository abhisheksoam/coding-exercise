import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import KBinsDiscretizer

k_fold = KFold(n_splits=10)
Accuracy_Fold = []
for n_bins in range(2, 20):
    Accuracy = []
    print(k_fold.split(X))
    for train_indices, test_indices in k_fold.split(X):
        print("Train: %s | test: %s" % (train_indices, test_indices))
        X_train, X_test, y_train, y_test = (
            X[train_indices],
            X[test_indices],
            y[train_indices],
            y[test_indices],
        )
        # discretise
        est = KBinsDiscretizer(n_bins=n_bins, encode="onehot", strategy="uniform")
        est.fit(X_train)
        Xd_train = est.transform(X_train)
        Xd_test = est.transform(X_test)
        clf = MultinomialNB()
        # training
        clf.fit(Xd_train, y_train)

        #     #prediction
        y_test_pred = clf.predict(Xd_test)
        # accuracy
        Accuracy.append(accuracy_score(y_test, y_test_pred))
        # print("Average 10-fold cross-validation accuracy=",np.mean(np.array(Accuracy)))
        Accuracy_Fold.append([n_bins, np.mean(np.array(Accuracy))])
        Accuracy_Fold = np.array(Accuracy_Fold)
        print(Accuracy_Fold)
        ind = np.argmax(Accuracy_Fold[:, 1])
        print("Best n_bins=", Accuracy_Fold[ind, :])
