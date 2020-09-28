"""
Sidharth Lakshmanan
CSE 163 AA

This file contains the WeatherModels class
and allows the user to train and tune machine
learning models to predict future weather
based on weather from the past "n" days.

Initializor for class:
    WeatherModels(self, String, int)

Functions Included in this class:

    train_linear_regressor(self, Boolean)
    train_neural_net(self, int, Tuple, Boolean)
    train_regression_tree(self, int, int, Boolean)
    tune_neural_net(self, int, list<String>, String)
    tune_regression_tree(self, int, list<String>, String)
    predict(self, DataFrame, model)
"""

import pandas as pd
import altair as alt
from math import floor
from altair_saver import save
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class WeatherModels:
    """
    This class allows the user to train:
        1) Linear Regression model
                LinearRegression from sklearn.linear_model
        2) Decision Tree Regression model
                DecisionTreeRegressor from sklearn.tree
        3) Neural Network model
                MLPRegressor from sklearn.neural_network

    Initializor for class:
        WeatherModels(self, String, int)

    Functions Included in this class:

        train_linear_regressor(self, Boolean)
        train_neural_net(self, int, Tuple, Boolean)
        train_regression_tree(self, int, int, Boolean)
        tune_neural_net(self, int, list<String>, String)
        tune_regression_tree(self, int, list<String>, String)
        predict(self, DataFrame, model)
    """

    def __init__(self, path, n):
        """
        Input: String, int
            path -  this string represents the path
                    of the csv file that contains the
                    weather data
            n    -  this int represents how many days
                    back the model should account for
                    (i.e., if the model needs to predict
                    weather based on the past 3 days, N=3)
        """
        self._reg_model = None
        self._n = n
        weather = pd.read_csv(path)
        cols = weather.columns
        # Extend dataframe with past n days weather
        for i in range(3, len(cols)):
            for j in range(1, self._n + 1):
                self._add_nth_day(weather, j, cols[i])
        data = weather.dropna()
        self._labels = data.iloc[:, 3:15]
        exclude = set(range(3, 15))
        self._features = data.iloc[:, [i for i in range(data.shape[1])
                                       if i not in exclude]]

    def _add_nth_day(self, weather, n, feature):
        """
        Input: DataFrame, int, string
            weather -   this DataFrame represents the weather data
            n       -   this int represents which day
                        back the column should be added
            feature -   this string represents the weather feature
                        that should be operated on
        Output:
            Will append the Nth day back's feature as another column
        Note: This function is private so it may not work as expected
        """
        rows = weather.shape[0]
        # Set first cells to NaN
        newcol = [None] * n
        # Set rest to shifted values
        for i in range(n, rows):
            newcol.append(weather[feature][i - n])
        # Append to dataframe
        weather[feature + '_' + str(n)] = newcol

    def train_linear_regressor(self, model_get=True):
        """
        This will train a linear regression model

        Optional Input: Boolean
            model_get   -   Requires a boolean for whether to
                            return the model or not (true is default)
        Output:
        Will return a tuple: (Training Accuracy, Testing Accuracy, Model)
        Both accuracies are R^2 scores
        The model is the model that was trained (will only return if model_get
        is True)
        """
        features_train, features_test, labels_train, labels_test = \
            train_test_split(self._features, self._labels, test_size=0.2)
        self._reg_model = LinearRegression().fit(features_train, labels_train)
        train_acc = r2_score(labels_train,
                             self._reg_model.predict(features_train),
                             multioutput='variance_weighted')
        test_acc = r2_score(labels_test,
                            self._reg_model.predict(features_test),
                            multioutput='variance_weighted')
        if model_get:
            return (train_acc, test_acc, self._reg_model)
        else:
            return (train_acc, test_acc)

    def train_neural_net(self, max_iter=500, hidden_layer_sizes=(100,),
                         model_get=True):
        """
        This will train a neural network model

        Optional Inputs: int, Tuple, Boolean
            max_iter    -   Requires an int specifying how many iterations
                            to do when training the model. Defaults to 500.
            hidden_layer_sizes  - Requires a tuple representing the hidden
                                  layers of the model where the ith element
                                  represents the number of neurons of the
                                  ith hidden layer. Will default to (100,)
            model_get   -   Requires a boolean for whether to
                            return the model or not (true is default)
        Output:
        Will return a tuple: (Training Accuracy, Testing Accuracy, Model)
        Both accuracies are R^2 scores
        The model is the model that was trained (will only return if model_get
        is True)
        """
        features_train, features_test, labels_train, labels_test = \
            train_test_split(self._features, self._labels, test_size=0.2)
        self._nn_model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                      max_iter=max_iter)
        self._nn_model.fit(features_train, labels_train)
        train_acc = r2_score(labels_train,
                             self._nn_model.predict(features_train),
                             multioutput='variance_weighted')
        test_acc = r2_score(labels_test,
                            self._nn_model.predict(features_test),
                            multioutput='variance_weighted')
        if model_get:
            return (train_acc, test_acc, self._nn_model)
        else:
            return (train_acc, test_acc)

    def train_regression_tree(self, max_features=None,
                              max_depth=None, model_get=True):
        """
        This will train a regression tree model

        Optional Inputs: int, int, Boolean
            max_features-   Requires an int specifying how many features
                            to consider. Defaults to include all of the
                            features.
            max_depth   -   Requires an int specifying how deep to make the
                            tree.
                            Defaults to max depth.
            model_get   -   Requires a boolean for whether to
                            return the model or not (true is default)
        Output:
        Will return a tuple: (Training Accuracy, Testing Accuracy, Model)
        Both accuracies are R^2 scores
        The model is the model that was trained (will only return if model_get
        is True)
        """
        features_train, features_test, labels_train, labels_test = \
            train_test_split(self._features, self._labels, test_size=0.2)
        self._tree_model = DecisionTreeRegressor(max_features=max_features,
                                                 max_depth=max_depth)
        self._tree_model = self._tree_model.fit(features_train, labels_train)
        train_acc = r2_score(labels_train,
                             self._tree_model.predict(features_train),
                             multioutput='variance_weighted')
        test_acc = r2_score(labels_test,
                            self._tree_model.predict(features_test),
                            multioutput='variance_weighted')
        if model_get:
            return (train_acc, test_acc, self._tree_model)
        else:
            return (train_acc, test_acc)

    def tune_neural_net(self, sample_size,
                        hparams=['max_iter', 'hidden_layer_sizes'],
                        folder="."):
        """
        Input: int
            sample_size -   This represents the number of samples per state
                            of tuning. Note that the more samples, the more
                            time it takes.
        Optional Input: list of strings, String
            hparams     -   This represents which parameters to tune.
                            Possible elements of list are:
                            ["max_iter", "hidden_layer_sizes"]
                            By default, this will tune all of these parameters.
            folder      -   path of the folder where the graphs should be saved
                            defaults to the current folder where this program
                            is located. Do not include final backslash.
        Output:
            Will save plots to the local directory representing the parameters
            that were tuned.
        """
        # Tuning parameters
        if 'max_iter' in hparams:
            acc_df = pd.DataFrame(columns=['Number of Iterations',
                                           'test_acc',
                                           'train_acc'])
            iters = list(range(200, 2001, 200))
            index = 0
            for i in iters:
                for j in range(sample_size):
                    train, test = self.train_neural_net(max_iter=i,
                                                        model_get=False)
                    acc_df.loc[index] = pd.Series({'Number of Iterations': i,
                                                   'train_acc': train,
                                                   'test_acc': test})
                    index += 1
            # Make plot
            iter_plot = self._plot_accuracy(acc_df, 'Number of Iterations')
            save(iter_plot, folder + "/neural_net_iterations.html")
        if 'hidden_layer_sizes' in hparams:
            acc_df = pd.DataFrame(columns=['Hidden Layer Configuration',
                                           'test_acc',
                                           'train_acc'])
            layers = [(50,), (100,), (50, 50), (100, 50), (100, 100),
                      (100, 50, 50), (100, 100, 50), (100, 100, 100)]
            index = 0
            for i in layers:
                for j in range(sample_size):
                    train, test = self.train_neural_net(hidden_layer_sizes=i,
                                                        model_get=False)
                    acc_df.loc[index] = \
                        pd.Series({'Hidden Layer Configuration': i,
                                   'train_acc': train,
                                   'test_acc': test})
                    index += 1
            # Make plot
            depth_plot = self._plot_accuracy(acc_df,
                                             'Hidden Layer Configuration')
            save(depth_plot, folder + "/neural_net_layers.html")

    def tune_regression_tree(self, sample_size,
                             hparams=['max_features', 'max_depth'],
                             folder="."):
        """
        Input: int
            sample_size -   This represents the number of samples per state
                            of tuning. Note that the more samples, the more
                            time it takes.
        Optional Input: list of strings, String
            hparams     -   This represents which parameters to tune.
                            Possible elements of list are:
                            ["max_features", "max_depth"]
                            By default, this will tune all of these parameters.
            folder      -   path of the folder where the graphs should be saved
                            defaults to the current folder where this program
                            is located. Do not include final backslash.
        Output:
            Will save plots to the local directory representing the parameters
            that were tuned.
        """
        # Tuning parameters
        if 'max_features' in hparams:
            acc_df = pd.DataFrame(columns=['Number of Features',
                                           'test_acc',
                                           'train_acc'])
            max_features = list(range(self._n + 3, self._features.shape[1],
                                      self._n))
            index = 0
            for i in max_features:
                for j in range(sample_size):
                    train, test = self.train_regression_tree(max_features=i,
                                                             max_depth=10,
                                                             model_get=False)
                    acc_df.loc[index] = pd.Series({'Number of Features': i,
                                                   'train_acc': train,
                                                   'test_acc': test})
                    index += 1
            # Make plot
            feat_plot = self._plot_accuracy(acc_df, 'Number of Features')
            save(feat_plot, folder + "/regression_tree_features.html")
        if 'max_depth' in hparams:
            acc_df = pd.DataFrame(columns=['Number of Levels Deep',
                                           'test_acc',
                                           'train_acc'])
            max_depth = list(range(2, 20, 2))
            index = 0
            for i in max_depth:
                for j in range(sample_size):
                    train, test = self.train_regression_tree(max_depth=i,
                                                             model_get=False)
                    acc_df.loc[index] = pd.Series({'Number of Levels Deep': i,
                                                   'train_acc': train,
                                                   'test_acc': test})
                    index += 1
            # Make plot
            depth_plot = self._plot_accuracy(acc_df, 'Number of Levels Deep')
            save(depth_plot, folder + "/regression_tree_depth.html")

    def _plot_accuracy(self, acc_df, hparam):
        """
        Input: DataFrame, String
            acc_df    -   this DataFrame is the training and
                            testing accuracies
            hparam      -   this is the column name in the dataframes
                            that needs to be plotted against accuracy
        Output: Altair chart
            Will return a chart with the plots 2, regular, and 2 zoomed
            in, of the training and testing data
        Note: This function is private so it may not work as expected
        """
        test_df_min = floor(acc_df['test_acc'].min() * 10) / 10.0
        test_df_max = floor(acc_df['test_acc'].max() * 10) / 10.0 + 0.1
        train_df_min = floor(acc_df['train_acc'].min() * 10) / 10.0
        train_df_max = floor(acc_df['train_acc'].max() * 10) / 10.0 + 0.1
        # Regular Graphs
        test_meds = alt.Chart(acc_df).mark_line(color="red").encode(
            x=hparam,
            y=alt.Y('median(test_acc):Q',
                    axis=alt.Axis(title='Accuracy (R^2)'))
        ).properties(height=200, width=400, title='Testing Data Accuracy')
        train_meds = alt.Chart(acc_df).mark_line(color="red").encode(
            x=hparam,
            y=alt.Y('median(train_acc):Q',
                    axis=alt.Axis(title='Accuracy (R^2)'))
        ).properties(height=200, width=400, title='Training Data Accuracy')
        test_chart = alt.Chart(acc_df).mark_boxplot().encode(
            x=hparam,
            y=alt.Y('test_acc:Q',
                    axis=alt.Axis(title='Accuracy (R^2)'))
        ).properties(height=200, width=400, title='Testing Data Accuracy')
        train_chart = alt.Chart(acc_df).mark_boxplot().encode(
            x=hparam,
            y=alt.Y('train_acc:Q',
                    axis=alt.Axis(title='Accuracy (R^2)'))
        ).properties(height=200, width=400, title='Training Data Accuracy')
        total = ((test_chart + test_meds) | (train_chart + train_meds))
        # Zoomed in Graphs
        test_meds = alt.Chart(acc_df).mark_line(color="red").encode(
            x=hparam,
            y=alt.Y('median(test_acc):Q',
                    axis=alt.Axis(title='Accuracy (R^2)'),
                    scale=alt.Scale(domain=[test_df_min, test_df_max]))
        ).properties(height=200, width=400,
                     title='Testing Data Accuracy (Zoomed)')
        train_meds = alt.Chart(acc_df).mark_line(color="red").encode(
            x=hparam,
            y=alt.Y('median(train_acc):Q',
                    axis=alt.Axis(title='Accuracy (R^2)'),
                    scale=alt.Scale(domain=[train_df_min, train_df_max]))
        ).properties(height=200, width=400,
                     title='Training Data Accuracy (Zoomed)')
        test_chart = alt.Chart(acc_df).mark_boxplot().encode(
            x=hparam,
            y=alt.Y('test_acc:Q',
                    axis=alt.Axis(title='Accuracy (R^2)'),
                    scale=alt.Scale(domain=[test_df_min, test_df_max]))
        ).properties(height=200, width=400,
                     title='Testing Data Accuracy (Zoomed)')
        train_chart = alt.Chart(acc_df).mark_boxplot().encode(
            x=hparam,
            y=alt.Y('train_acc:Q',
                    axis=alt.Axis(title='Accuracy (R^2)'),
                    scale=alt.Scale(domain=[train_df_min, train_df_max]))
        ).properties(height=200, width=400,
                     title='Training Data Accuracy (Zoomed)')
        total &= ((test_chart + test_meds) | (train_chart + train_meds))
        return total

    def predict(self, weather, model):
        """
        Input: DataFrame, model
            weather -   this DataFrame represents the weather data
                        (only include n+1 lines with the n+1'st line
                        including the data at minimum to predict)
            model   -   this is the model returned by the train_*
                        functions
        Output: DataFrame
            Will return a dataframe of the predicted values of the n+1'st
            row. If there are more, or less than n+1 rows, the behavior is
            undefined.
        """
        weather = weather.copy()
        cols = weather.columns
        # Extend dataframe with past n days weather
        for i in range(3, len(cols)):
            for j in range(1, self._n + 1):
                self._add_nth_day(weather, j, cols[i])
        labels = weather.iloc[:, 3:15]
        exclude = set(range(3, 15))
        features = weather.iloc[:, [i for i in range(weather.shape[1])
                                if i not in exclude]]
        features = features.dropna()
        return pd.DataFrame(data=model.predict(features),
                            columns=labels.columns)
