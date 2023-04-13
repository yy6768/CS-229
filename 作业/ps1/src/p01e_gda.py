import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    gda = GDA()
    gda.fit(x_train, y_train)

    x_test, y_test = util.load_dataset(eval_path, add_intercept=False)
    y_hat = gda.predict(x_test)
    util.plot(x_train, y_train, gda.theta, 'output/p01e_{}.png'.format(pred_path[-5]))
    np.savetxt(pred_path, np.array(list(zip(y_hat > 0.5, y_test))), fmt='%d')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=0.2, max_iter=100, eps=1e-5,
                 theta_0=None, verbose=True):
        super().__init__(step_size, max_iter, eps, theta_0, verbose)

        self.theta = None

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n + 1)
        y_sum = np.sum(y)
        phi = y_sum / m
        mu0 = np.sum(x[y == 0], axis=0) / (m - y_sum)
        mu1 = np.sum(x[y == 1], axis=0) / y_sum
        sigma = ((x[y == 0] - mu0).T.dot(x[y == 0] - mu0) + (x[y == 1] - mu1).T.dot(x[y == 1] - mu1)) / m
        sigma_inv = np.linalg.inv(sigma)
        self.theta[0] = 0.5 * (mu0 + mu1).dot(sigma_inv).dot(mu0 - mu1) - np.log((1 - phi) / phi)
        self.theta[1:] = sigma_inv.dot(mu1 - mu0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        x0 = np.ones(m)
        x = np.insert(x, n, x0, axis=1)
        p = 1 / (1 + np.exp(-x.dot(self.theta)))
        return p
        # *** END CODE HERE
