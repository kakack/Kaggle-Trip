import pandas as pd
import numpy as np
from Data_Processing import data_basic_processing, dif_num_and_cat
from scipy.stats import skew
from Modeling import linear_regression, lasso_model, ridge_model, elasticNet_model


if __name__ == '__main__':
    train = pd.read_csv("./dataset/train.csv")
    test = pd.read_csv("./dataset/test.csv")
    train.SalePrice = np.log1p(train.SalePrice)
    y = train.SalePrice

    train_count = train.shape[0]
    test_count = test.shape[0]

    data_whole = pd.concat([train, test])
    data_basic_processing(data_whole)

    data_whole, whole_num, whole_cat = dif_num_and_cat(data_whole)

    skewness = whole_num.apply(lambda x: skew(x))
    skewness = skewness[abs(skewness) > 0.5]
    print(str(skewness.shape[0]) + " skewed numerical features to log transform")
    skewed_features = skewness.index
    whole_num[skewed_features] = np.log1p(whole_num[skewed_features])

    print("NAs for categorical features in train : " + str(whole_cat.isnull().values.sum()))
    whole_cat = pd.get_dummies(whole_cat)
    print("Remaining NAs for categorical features in train : " + str(whole_cat.isnull().values.sum()))

    whole_final = pd.concat([whole_num, whole_cat], axis = 1)
    print("New number of features : " + str(train.shape[1]))

    train_final = whole_final[:train_count]
    test_final = whole_final[whole_final.shape[0] - train_count +1:]

    lr = linear_regression(train_final, y)
    predicted_0 = lr.predict(test_final)
    result_0 = pd.DataFrame({'Id': test['Id'].as_matrix(), 'SalePrice': np.expm1(predicted_0)})
    result_0.to_csv("./output/logistic_regression.csv", index=False)

    ridge = ridge_model(train_final, y)
    predicted_1 = ridge.predict(test_final)
    result_1 = pd.DataFrame({'Id': test['Id'].as_matrix(), 'SalePrice': np.expm1(predicted_1)})
    result_1.to_csv("./output/ridge_model.csv", index=False)

    lasso = lasso_model(train_final, y)
    predicted_2 = lasso.predict(test_final)
    result_2 = pd.DataFrame({'Id': test['Id'].as_matrix(), 'SalePrice': np.expm1(predicted_2)})
    result_2.to_csv("./output/lasso_model.csv", index=False)

    elastic = elasticNet_model(train_final, y)
    predicted_3 = elastic.predict(test_final)
    result_3 = pd.DataFrame({'Id': test['Id'].as_matrix(), 'SalePrice': np.expm1(predicted_3)})
    result_3.to_csv("./output/elasticNet_model.csv", index=False)




