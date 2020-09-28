"""
Sidharth Lakshmanan
CSE 163 AA

This file tests the implementation of the
WeatherModels class. Thus, by extension,
it accesses the private fields of the class.
In practice, this should not be done.
"""
from weather_models import WeatherModels
from timeit import default_timer as timer
import pandas as pd
import warnings
from os import listdir, unlink


def main():
    """
    This file tests the implementation of the
    WeatherModels class. Thus, by extension,
    it accesses the private fields of the class.
    In practice, this should not be done.
    Also, this program will print out cumulative
    time since start of program to make sure
    run time is not absurdly large.

    Note: This will clear out and use the "test"
    folder
    """
    n = 4
    # Folder for test graphs
    folder = './test'
    # Surpressing warnings because the nueral network
    # will show warnings as there is too little data
    # (fine for testing on sample.csv)
    warnings.filterwarnings("ignore")
    # Starting timer to make sure no method runs for an
    # absurd amount of time
    start = timer()
    # Initialize data
    weather = WeatherModels("sample.csv", n)
    verify_init(weather, n)
    print("Data Initialized: " + str(timer() - start) + "sec")
    # Clean out folder just in case
    verify_tune(-1, folder)
    # Tune models
    weather.tune_regression_tree(1, folder=folder)
    verify_tune(2, folder)
    weather.tune_regression_tree(1, hparams=['max_features'], folder=folder)
    verify_tune(1, folder)
    weather.tune_regression_tree(1, hparams=['max_depth'], folder=folder)
    verify_tune(1, folder)
    print("Regression Tree Tuned: " + str(timer() - start) + "sec")
    weather.tune_neural_net(1, folder=folder)
    verify_tune(2, folder)
    weather.tune_neural_net(1, hparams=['max_iter'], folder=folder)
    verify_tune(1, folder)
    weather.tune_neural_net(1, hparams=['hidden_layer_sizes'], folder=folder)
    verify_tune(1, folder)
    print("Neural Network Tuned: " + str(timer() - start) + "sec")
    # Train models : Note that this part will take about 1 minute to complete
    weather_full = WeatherModels("data.csv", n)
    reg_model = weather_full.train_linear_regressor()
    tree_model = weather_full.train_regression_tree(
        max_features=35,
        max_depth=8
    )
    nn_model = weather_full.train_neural_net(
        hidden_layer_sizes=(100, 100),
        max_iter=500
    )
    verify_train(weather_full, reg_model, tree_model, nn_model, n)
    print("\nModels Trained: " + str(timer() - start) + "sec")
    print("Success!!!")


def verify_init(weather, n):
    """
    Will verify the intialization of the data

    Input: WeatherModels instance, int
        weather -   This is the instance of the WeatherModels
                    class to test.
        n       -   This is the int representing how many days
                    back the WeatherModels intance was given.

    Output:
        Will throw errors if the class does not meet specifications
    """
    test = pd.read_csv('sample.csv')
    cols = test.shape[1]
    rows = test.shape[0]
    assert n == weather._n, "N is not the same"
    features = weather._features.shape[1]
    labels = weather._labels.shape[1]
    assert features == ((cols - 3) * n) + 3, \
        "Features count is not correct"
    assert labels == cols - 3, \
        "Labels count is not correct"
    rowsf = weather._features.shape[0]
    rowsl = weather._labels.shape[0]
    assert rowsf == rowsl, "Feature and Label rows do not match"
    assert rowsf == rows - n, "Row count is not correct"


def verify_tune(files, folder):
    """
    Will verify the tuning of the models

    Input: int, String
        files   -   This is the int representing how many files
                    should be created in the folder. If files is negative,
                    will only delete all files and not check correctness.
        folder  -   This is the path to the folder to check

    Output:
        Will throw errors if the class does not meet specifications
        Will also remove all files in folder.
    """
    count = 0
    for filename in listdir(folder):
        file_path = folder + "/" + filename
        unlink(file_path)
        count += 1
    if files >= 0:
        assert count == files, "Number of Files created is wrong"


def verify_train(weather, reg_model, tree_model, nn_model, n):
    """
    Will verify the training of the models

    Input: Tuple, Tuple, Tuple, int
        weather     -   This is the instance of the WeatherModels
                        class to test.
        reg_model   -   Tuple as returned by train_linear_regressor()
        tree_model  -   Tuple as returned by train_regression_tree()
        nn_model    -   Tuple as returned by train_neural_net()
        n           -   Number of days back as used in the intialization
                        of the WeatherModels class

    Output:
        Will throw errors if the class does not meet specifications
    """
    test = pd.read_csv('data.csv', nrows=n + 1)
    real_data = test.iloc[[-1], 3:15]
    assert reg_model[0] > 0.3 and tree_model[0] > 0.3 and nn_model[0] > 0.3, \
        "Training errors are too low"
    assert reg_model[1] > 0.3 and tree_model[1] > 0.3 and nn_model[1] > 0.3, \
        "Testing errors are too low"
    reg_df = weather.predict(test, reg_model[2])
    tree_df = weather.predict(test, tree_model[2])
    nn_df = weather.predict(test, nn_model[2])
    print("Requires manual inspection for accuracy:")
    print("\tActual Data:\n")
    print(real_data)
    print("\n\tLinear Regressor:\n")
    print(reg_df)
    print("\n\tRegression Tree:\n")
    print(tree_df)
    print("\n\tNeural Network:\n")
    print(nn_df)


if __name__ == '__main__':
    main()
