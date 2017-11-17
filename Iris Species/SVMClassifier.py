import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    iris = pd.read_csv('./data/Iris.csv')
    # print(iris.describe())
    y = np.array(iris[['Species']])
    x = np.array(iris.drop(['Id', 'Species'], axis=1))
    print("y")
    print(y[0:5])
    print("x")
    print(x[0:5])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, stratify=y)

    from sklearn.svm import LinearSVC

    svmLinearModel = LinearSVC()
    svmLinearModel.fit(x_train, y_train.ravel())
    y_pred = svmLinearModel.predict(x_test)
    print(svmLinearModel.score(x_test, y_test.ravel()))

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    import matplotlib.pyplot as plt
    import seaborn as sn

    df_cm = pd.DataFrame(cm, index=[i for i in np.unique(y)], columns=[i for i in np.unique(y)])
    plt.figure(figsize= (5, 5))
    sn.heatmap(df_cm, annot=True)
    plt.show()

    # Non-linear SVM
    from sklearn.svm import SVC

    svcModel = SVC()
    svcModel.fit(x_train, y_train.ravel())
    y_scvpred = svcModel.predict(x_test)
    print(svcModel.score(x_test,y_test.ravel()))

    cm2 = confusion_matrix(y_test, y_scvpred)
    df_cm2 = pd.DataFrame(cm2, index=[i for i in np.unique(y)], columns=[i for i in np.unique(y)])
    plt.figure(figsize=(5, 5))
    sn.heatmap(df_cm2, annot=True)
    plt.show()

