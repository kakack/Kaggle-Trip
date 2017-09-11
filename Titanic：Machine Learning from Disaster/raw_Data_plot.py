import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def raw_Data_plot(data):
    fig = plt.figure()
    fig.set(alpha=0.2)

    plt.subplot2grid((2, 3), (0, 0))
    data.Survived.value_counts().plot(kind='bar')
    plt.title("Total Survived situation")
    plt.ylabel("Head count")

    plt.subplot2grid((2, 3), (0, 1))
    data.Pclass.value_counts().plot(kind='bar')
    plt.title("Passenger Class")
    plt.ylabel("Head count")

    plt.subplot2grid((2, 3), (0, 2))
    plt.scatter(data.Age, data.Survived)
    plt.ylabel("Survived situation")
    plt.xlabel("Age")
    plt.grid(b=True,
             which='major',
             axis='x')
    plt.title("Survived situation on Ages")

    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    for i in range(1, 4):
        data.Age[data.Pclass == i].plot(kind='kde')
    # data.Age[data.Pclass == 1].plot(kind='kde')
    # data.Age[data.Pclass == 2].plot(kind='kde')
    # data.Age[data.Pclass == 3].plot(kind='kde')
    plt.xlabel("Age")
    plt.ylabel("Densty")
    plt.title("Different Ages in Class")
    plt.legend(('1-Class', '2-Class', '3-CLass'), loc ='best')

    plt.subplot2grid((2, 3), (1, 2))
    data.Embarked.value_counts().plot(kind='bar')
    plt.title("Different Port boarding count")
    plt.ylabel("Head Count")

    plt.show()


def main():
    data = pd.read_csv("./dataset/train.csv")
    raw_Data_plot(data)


if __name__ == "__main__":
    main()

