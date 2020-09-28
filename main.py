"""
Sidharth Lakshmanan
CSE 163 AA

This code produces a graph comparing each of the models trained.
If the commented code is uncommented, this code has the potential
to tune each of the models, and compare their accuracies.
"""

import altair as alt
from altair_saver import save
from weather_models import WeatherModels
from timeit import default_timer as timer
import pandas as pd
from math import floor


def main():
    """
    This code produces a graph comparing each of the models trained.
    If the commented code is uncommented, this code has the potential
    to tune each of the models, and compare their accuracies.
    """
    # Actual tunings (n = 4) are located in the
    # parameter tunings folder. Do not modify that folder.
    # Note that the following code can take a long time to
    # run. (Ignore any warnings thrown for the sample.csv data.)

    start = timer()
    weather = WeatherModels("sample.csv", 4)
    print("Data Initialized: " + str(timer() - start) + "sec")

    # Code used to produce graphs in the parameter_tunings folder
    # Takes around 5000 seconds
    # If planning to uncomment this code, please change the folder
    # to a different folder name:
    #
    # tune_models(weather, './parameter_tunings')
    # print("Models Tuned: " + str(timer() - start) + "sec")
    #

    # The following code was used in the creation of the model_comparison
    # graph in the model_comparison folder. Code take around 1000 seconds
    # to run. Don't uncomment this code.
    #
    # compare_models(weather, "./model_comparison")
    #
    compare_models(weather, "./test")
    print("Models Compared: " + str(timer() - start) + "sec")


def tune_models(weather, folder):
    '''
    This produces the graphs of the tuning hyperparameters for each model

    Input:
        weather -   This is the instance of the WeatherModels class
        folder  -   This is the path of the folder the graphs should be stored
                    Should not include the final backslash
    Output:
        Will store the graphs in the folder specified
    '''
    weather.tune_regression_tree(10, folder=folder)
    weather.tune_neural_net(10, folder=folder)


def compare_models(weather, folder):
    '''
    This produces a graph comparing each of the models for accuracy (R^2)
    This will use the best hyperparameters for the data.csv dataset

    Input:
        weather -   This is the instance of the WeatherModels class
        folder  -   This is the path of the folder the graphs should be stored
                    Should not include the final backslash
    Output:
        Will store the graphs in the folder specified
    '''
    acc_df = pd.DataFrame(columns=['Model', 'test_acc'])
    index = 0
    for i in range(10):
        model = weather.train_linear_regressor(model_get=False)
        acc_df.loc[index] = pd.Series({'Model': 'Linear Regression',
                                       'test_acc': model[1]})
        index += 1
        model = weather.train_regression_tree(max_features=35,
                                              max_depth=8,
                                              model_get=False)
        acc_df.loc[index] = pd.Series({'Model': 'Decision Tree Regression',
                                       'test_acc': model[1]})
        index += 1
        model = weather.train_neural_net(max_iter=500,
                                         hidden_layer_sizes=(100, 100, 100),
                                         model_get=False)
        acc_df.loc[index] = pd.Series({'Model': 'Neural Network',
                                       'test_acc': model[1]})
        index += 1
    # Create graph
    test_df_min = floor(acc_df['test_acc'].min() * 100) / 100.0
    test_df_max = floor(acc_df['test_acc'].max() * 100) / 100.0 + 0.01
    test_chart = alt.Chart(acc_df).mark_boxplot(size=100).encode(
        x='Model',
        y=alt.Y('test_acc:Q',
                axis=alt.Axis(title='Accuracy (R^2)'),
                scale=alt.Scale(domain=[test_df_min, test_df_max]))
    ).properties(height=400, width=800, title='Accuracy per Model')
    save(test_chart, folder + "/" + "model_comparison.html")


if __name__ == '__main__':
    main()
