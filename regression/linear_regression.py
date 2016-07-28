import numpy as np
from sklearn import datasets, linear_model

from sklearn.cross_validation import train_test_split, KFold

diabetes = datasets.load_diabetes()


class LinearRegression:
    def __init__(self, X, y, test_split):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_split)

    # Create linear regression object
    def train(self, model):
        # K-fold cross validation. Chose the one which maximises the model score
        folds = KFold(len(self.X_train), n_folds=10)
        scores = []
        for train, test in folds:
            X, y = self.X_train, self.y_train
            model.fit(X[train], y[train])
            score = model.score(X[test], y[test])
            print("Residual sum of squares: %.2f" % float(np.mean(model.predict(X=X[test]) - y[test]) ** 2))
            print('Variance score: %.2f' % score)
            scores.append(score)
        print "----- Variance in scores %f" % np.var(scores)
        print


def main():
    X = diabetes.data
    y = diabetes.target
    regression = LinearRegression(X, y, 0.9)

    print "Linear Regression"
    # Should have more variance in scores
    linear_regression = linear_model.LinearRegression()
    regression.train(linear_regression)

    print "Ridge Regression"
    # Lesser variance in scores
    ridge_regression = linear_model.Ridge()
    regression.train(ridge_regression)

    print "Lasso Regression"
    # Lesser variance in scores
    lasso_regression = linear_model.Lasso()
    regression.train(lasso_regression)


if __name__ == "__main__":
    main()
