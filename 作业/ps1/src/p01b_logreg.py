import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_hat = logistic_regression.predict(x_eval)
    util.plot(x_train, y_train, logistic_regression.theta, 'output/p01b_{}.png'.format(pred_path[-5]))
    np.savetxt(pred_path, np.array(list(zip(y_hat > 0.5, y_eval))), fmt='%d')
    # *** END CODE HERE ***


def logistic(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=0.2, max_iter=100, eps=1e-5, theta_0=None, verbose=True):
        super().__init__(step_size, max_iter, eps, theta_0, verbose)

    def logistic_hessian(self, x):
        m, _ = x.shape
        h_theta = logistic(x @ self.theta)
        hessian = (x.T * h_theta * (1 - h_theta)).dot(x) / m
        return hessian

    def grad(self, x, y):
        m, n = x.shape
        gradient = np.zeros(n)
        for i in range(m):
            gradient += (y[i] - logistic(np.dot(self.theta, x[i]))) * x[i]
        return - 1 / m * gradient

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # init
        m, n = x.shape

        if self.theta is None:
            self.theta = np.zeros(n)

        # iteration
        k = 0

        while k < self.max_iter:
            hessian = self.logistic_hessian(x)
            ini_hessian = np.linalg.inv(hessian)
            gradient = self.grad(x, y)
            delta = ini_hessian @ gradient
            self.theta -= delta

            k += 1
            if np.linalg.norm(delta, ord=1) < self.eps:
                break

    # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return logistic(x @ self.theta)
        # *** END CODE HERE ***
