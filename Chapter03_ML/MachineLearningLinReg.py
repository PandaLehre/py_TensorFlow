import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from tf_utils.dummyData import regression_data


def main() -> None:
    x, y = regression_data()
    x = x.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    regr = LinearRegression()
    regr.fit(x_train, y_train)  # train
    score = regr.score(x_test, y_test)
    print(f"R2-Score: {score}")
    print(f"Coefs: {regr.coef_}")  # m
    print(f"Intercept: {regr.intercept_}")  # b

    y_pred = regr.predict(x_test)

    plt.scatter(x, y)
    plt.plot(x_test, y_pred, color="green")
    plt.show()


if __name__ == "__main__":
    main()
