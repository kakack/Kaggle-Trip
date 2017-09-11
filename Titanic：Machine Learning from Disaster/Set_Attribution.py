from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def set_missing_ages(df):
    age_df = df[["Age", "Fare", "Parch", "SibSp", "Pclass"]]

    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y = known_age[:, 0]
    X = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    predictedAge = rfr.predict(unknown_age[:, 1::])
    df.loc[(df.Age.isnull()), 'Age'] = predictedAge

    return df, rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 1
    df.loc[(df.Cabin.isnull()), 'Cabin'] = -1
    df['Cabin'] = df['Cabin'].astype(int)
    return df

def set_Sex_type(df):
    df.loc[(df.Sex == 'female'), 'Sex'] = 1
    df.loc[(df.Sex == 'male'), 'Sex'] = -1
    df['Sex'] = df['Sex'].astype(int)
    return df

def set_Embarked_type(df):
    df.loc[(df.Embarked == 'C'), 'Embarked'] = 1
    df.loc[(df.Embarked == 'Q'), 'Embarked'] = 2
    df.loc[(df.Embarked == 'S'), 'Embarked'] = 3
    df.loc[(df.Embarked.isnull()), 'Embarked'] = 0
    df['Embarked'] = df['Embarked'].astype(int)
    return df

def main():
    df = pd.read_csv("./dataset/train.csv")
    data_train, rfr = set_missing_ages(df)
    data_train = set_Cabin_type(data_train)
    data_train = set_Sex_type(data_train)
    data_train = set_Embarked_type(data_train)
    print(data_train)

if __name__ == '__main__':
    main()