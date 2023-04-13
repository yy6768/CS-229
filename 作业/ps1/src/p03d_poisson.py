import numpy as np
import matplotlib.pyplot as plt
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    # *** START CODE HERE ***
    model = PoissonRegression(step_size=lr)
    # Fit a Poisson Regression model
    model.fit(x_train, y_train)
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_test, y_test = util.load_dataset(eval_path, add_intercept=False)
    h_theta = model.predict(x_test)
    np.savetxt(pred_path, np.array(list(zip(h_theta, y_test))), fmt="%d")

    plt.figure()
    plt.plot(y_test, h_theta, 'bx')
    plt.xlabel('true counts')
    plt.ylabel('predict counts')
    plt.savefig('output/p03d.png')
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        k = 0
        if self.theta is None:
            self.theta = np.zeros(n)
        while k < self.max_iter:
            k += 1
            j = np.random.randint(n)
            # print(self.step_size)
            delta = (y - np.exp(x.dot(self.theta))).dot(x[:, j]) * self.step_size / m
            # print(delta)
            self.theta[j] += delta
            if delta < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***
