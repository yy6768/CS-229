import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    logistic_t = LogisticRegression()
    x_train, y_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    logistic_t.fit(x_train, y_train)
    y_hat_c = logistic_t.predict(x_test)
    # Make sure to save outputs to pred_path_c
    np.savetxt(pred_path_c, np.array(list(zip(y_hat_c > 0.5, t_test))), fmt="%d")
    # Part (d): Train on y-labels and test on true labels
    logistic_y = LogisticRegression()
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)
    logistic_y.fit(x_train, y_train)
    y_hat_d = logistic_y.predict(x_test)
    # Make sure to save outputs to pred_path_d
    np.savetxt(pred_path_d, np.array(list(zip(y_hat_d > 0.5, y_test))), fmt="%d")
    # Part (e): Apply correction factor using validation set and test on true labels
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    alpha = np.mean(logistic_y.predict(x_valid))

    y_hat_e = y_hat_d / alpha
    correction = 1 + np.log(2 / alpha - 1) / logistic_y.theta[0]

    # Plot and use np.savetxt to save outputs to pred_path_e

    np.savetxt(pred_path_e, y_hat_e, fmt="%d")
    util.plot(x_test, t_test, logistic_t.theta, 'output/p02c.png')
    util.plot(x_test, y_test, logistic_y.theta, 'output/p02d.png')
    util.plot(x_test, t_test, logistic_y.theta, 'output/p02e.png', correction)
    # *** END CODER HERE
