import util
import matplotlib.pyplot as plt


def plot(x, y):
    neg_x = x[y == -1, :]
    pos_x = x[y == 1, :]

    plt.scatter(neg_x[:, 0], neg_x[:, 1], marker='x', color='red')
    plt.scatter(pos_x[:, 0], pos_x[:, 1], marker='o', color='blue')

    plt.show()


if __name__ == '__main__':
    x_a, y_a = util.load_csv('../data/ds1_a.csv')
    plot(x_a, y_a)
    x_b, y_b = util.load_csv('../data/ds1_b.csv')
    plot(x_b, y_b)
