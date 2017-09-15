from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
import matplotlib.pyplot as plt
import pandas as pd


def linear_regression(data, y):
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    def rmse_cv_train(model):
        rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring=scorer, cv=10))
        return (rmse)

    def rmse_cv_test(model):
        rmse = np.sqrt(-cross_val_score(model, X_test, y_test, scoring=scorer, cv=10))
        return (rmse)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Look at predictions on training and validation set
    print("RMSE on Training set :", rmse_cv_train(lr).mean())
    print("RMSE on Test set :", rmse_cv_test(lr).mean())
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)

    # Plot residuals
    plt.scatter(y_train_pred, y_train_pred - y_train, c="blue", marker="s", label="Training data")
    plt.scatter(y_test_pred, y_test_pred - y_test, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
    plt.show()

    # Plot predictions
    plt.scatter(y_train_pred, y_train, c="blue", marker="s", label="Training data")
    plt.scatter(y_test_pred, y_test, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc="upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
    plt.show()

    return lr


def ridge_model(data, y):
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    def rmse_cv_train(model):
        rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring=scorer, cv=10))
        return (rmse)

    def rmse_cv_test(model):
        rmse = np.sqrt(-cross_val_score(model, X_test, y_test, scoring=scorer, cv=10))
        return (rmse)

    ridge = RidgeCV(alphas=[0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
    ridge.fit(X_train, y_train)
    alpha = ridge.alpha_
    print("Best alpha :", alpha)

    print("Try again for more precision with alphas centered around " + str(alpha))
    ridge = RidgeCV(alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,
                            alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                            alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],
                    cv=10)
    ridge.fit(X_train, y_train)
    alpha = ridge.alpha_
    print("Best alpha :", alpha)

    print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())
    print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())
    y_train_rdg = ridge.predict(X_train)
    y_test_rdg = ridge.predict(X_test)

    # Plot residuals
    plt.scatter(y_train_rdg, y_train_rdg - y_train, c="blue", marker="s", label="Training data")
    plt.scatter(y_test_rdg, y_test_rdg - y_test, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression with Ridge regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
    plt.show()

    # Plot predictions
    plt.scatter(y_train_rdg, y_train, c="blue", marker="s", label="Training data")
    plt.scatter(y_test_rdg, y_test, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression with Ridge regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc="upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
    plt.show()

    # Plot important coefficients
    coefs = pd.Series(ridge.coef_, index=X_train.columns)
    print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " + \
          str(sum(coefs == 0)) + " features")
    imp_coefs = pd.concat([coefs.sort_values().head(10),
                           coefs.sort_values().tail(10)])
    imp_coefs.plot(kind="barh")
    plt.title("Coefficients in the Ridge Model")
    plt.show()

    return ridge

def lasso_model(data, y):
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    def rmse_cv_train(model):
        rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring=scorer, cv=10))
        return (rmse)

    def rmse_cv_test(model):
        rmse = np.sqrt(-cross_val_score(model, X_test, y_test, scoring=scorer, cv=10))
        return (rmse)

    lasso = LassoCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1,
                            0.3, 0.6, 1],
                    max_iter=50000, cv=10)
    lasso.fit(X_train, y_train)
    alpha = lasso.alpha_
    print("Best alpha :", alpha)

    print("Try again for more precision with alphas centered around " + str(alpha))
    lasso = LassoCV(alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8,
                            alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05,
                            alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35,
                            alpha * 1.4],
                    max_iter=50000, cv=10)
    lasso.fit(X_train, y_train)
    alpha = lasso.alpha_
    print("Best alpha :", alpha)

    print("Lasso RMSE on Training set :", rmse_cv_train(lasso).mean())
    print("Lasso RMSE on Test set :", rmse_cv_test(lasso).mean())
    y_train_las = lasso.predict(X_train)
    y_test_las = lasso.predict(X_test)

    # Plot residuals
    plt.scatter(y_train_las, y_train_las - y_train, c="blue", marker="s", label="Training data")
    plt.scatter(y_test_las, y_test_las - y_test, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression with Lasso regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
    plt.show()

    # Plot predictions
    plt.scatter(y_train_las, y_train, c="blue", marker="s", label="Training data")
    plt.scatter(y_test_las, y_test, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression with Lasso regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc="upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
    plt.show()

    # Plot important coefficients
    coefs = pd.Series(lasso.coef_, index=X_train.columns)
    print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " + \
          str(sum(coefs == 0)) + " features")
    imp_coefs = pd.concat([coefs.sort_values().head(10),
                           coefs.sort_values().tail(10)])
    imp_coefs.plot(kind="barh")
    plt.title("Coefficients in the Lasso Model")
    plt.show()

    return lasso


def elasticNet_model(data, y):
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    def rmse_cv_train(model):
        rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring=scorer, cv=10))
        return (rmse)

    def rmse_cv_test(model):
        rmse = np.sqrt(-cross_val_score(model, X_test, y_test, scoring=scorer, cv=10))
        return (rmse)

    elasticNet = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                              alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006,
                                      0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
                              max_iter=50000, cv=10)
    elasticNet.fit(X_train, y_train)
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha)

    print("Try again for more precision with l1_ratio centered around " + str(ratio))
    elasticNet = ElasticNetCV(
        l1_ratio=[ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],
        alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
        max_iter=50000, cv=10)
    elasticNet.fit(X_train, y_train)
    if (elasticNet.l1_ratio_ > 1):
        elasticNet.l1_ratio_ = 1
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha)

    print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) +
          " and alpha centered around " + str(alpha))
    elasticNet = ElasticNetCV(l1_ratio=ratio,
                              alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,
                                      alpha * .9,
                                      alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25,
                                      alpha * 1.3,
                                      alpha * 1.35, alpha * 1.4],
                              max_iter=50000, cv=10)
    elasticNet.fit(X_train, y_train)
    if (elasticNet.l1_ratio_ > 1):
        elasticNet.l1_ratio_ = 1
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha)

    print("ElasticNet RMSE on Training set :", rmse_cv_train(elasticNet).mean())
    print("ElasticNet RMSE on Test set :", rmse_cv_test(elasticNet).mean())
    y_train_ela = elasticNet.predict(X_train)
    y_test_ela = elasticNet.predict(X_test)

    # Plot residuals
    plt.scatter(y_train_ela, y_train_ela - y_train, c="blue", marker="s", label="Training data")
    plt.scatter(y_test_ela, y_test_ela - y_test, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression with ElasticNet regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
    plt.show()

    # Plot predictions
    plt.scatter(y_train, y_train_ela, c="blue", marker="s", label="Training data")
    plt.scatter(y_test, y_test_ela, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression with ElasticNet regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc="upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
    plt.show()

    # Plot important coefficients
    coefs = pd.Series(elasticNet.coef_, index=X_train.columns)
    print("ElasticNet picked " + str(sum(coefs != 0)) + " features and eliminated the other " + str(
        sum(coefs == 0)) + " features")
    imp_coefs = pd.concat([coefs.sort_values().head(10),
                           coefs.sort_values().tail(10)])
    imp_coefs.plot(kind="barh")
    plt.title("Coefficients in the ElasticNet Model")
    plt.show()

    return elasticNet






