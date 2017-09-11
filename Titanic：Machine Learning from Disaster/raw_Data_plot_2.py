import matplotlib.pyplot as plt
import pandas as pd

def raw_Data_plot_2(data):
    fig = plt.figure()
    fig.set(alpha=0.2)

    Survived_0 = data.Pclass[data.Survived == 0].value_counts()
    Survived_1 = data.Pclass[data.Survived == 1].value_counts()
    df = pd.DataFrame({"Survived": Survived_1, "Unsurvived": Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title("Survived situation on classes")
    plt.xlabel("Classes")
    plt.ylabel("Head Count")
    plt.show()

    Survived_f = data.Survived[data.Sex == 'female'].value_counts()
    Survived_m = data.Survived[data.Sex == 'male'].value_counts()
    df = pd.DataFrame({"male": Survived_m, "female": Survived_f})
    df.plot(kind='bar', stacked=True)
    plt.title("Survived situation on Sex")
    plt.xlabel("Sex")
    plt.ylabel("Head count")
    plt.show()

    Survived_0 = data.Embarked[data.Survived == 0].value_counts()
    Survived_1 = data.Embarked[data.Survived == 1].value_counts()
    df = pd.DataFrame({'Survived': Survived_1, 'Unsurvived': Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title("Port Boarding Situation")
    plt.xlabel("Boarding port")
    plt.ylabel("Head Count")
    plt.show()

def main():
    data = pd.read_csv("./dataset/train.csv")
    raw_Data_plot_2(data)

if __name__ == '__main__':
    main()



