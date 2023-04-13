import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    best_tau = 0
    best_MSE = np.Inf
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_valid)
        MSE = np.mean((y_valid - y_pred) ** 2)
        if MSE < best_MSE:
            best_tau = tau
            best_MSE = MSE
    # Fit a LWR model with the best tau value
    best_model = LocallyWeightedLinearRegression(best_tau)
    # Run on the test set to get the MSE value
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    # Save predictions to pred_path
    mse = np.mean((y_pred - y_test) ** 2)
    print(f'test set: best_tau={best_tau}, MSE={mse}')
    np.savetxt(pred_path, np.array(list(zip(y_pred, y_test))), fmt="%d")
    # Plot data
    plt.figure()
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_valid, y_pred, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05c.png')
    # *** END CODE HERE ***
