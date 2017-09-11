# import sklearn.preprocessing as preprocessing
# import pandas as pd
# import Set_Attribution as sa
#
# def attr_scale(df):
#     scaler = preprocessing.StandardScaler()
#     age_scale_param = scaler.fit([df['Age']])
#     df['Age_scaled'] = scaler.fit_transform([df['Age']], [age_scale_param])
#     # fare_scale_param = scaler.fit([df['Fare']])
#     # df['Fare'] = scaler.fit_transform(df['Fare', fare_scale_param])
#     return  df
#
# def main():
#     df = pd.read_csv("./dataset/train.csv")
#     data_train, rfr = sa.set_missing_ages(df)
#     data_train = attr_scale(data_train)
#     print(data_train)
#
# if __name__ == '__main__':
#     main()
