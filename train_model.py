# for data
import numpy as np
import pandas as pd

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# for training testing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# for saving regression model as file for reuseability
from joblib import Parallel, delayed
import joblib
import os.path


# make directories if it does not exist
def check_folder_exists(path_to_folder):
    if not os.path.exists(path_to_folder):
        os.makedirs("./" + path_to_folder)


def main():
    random_state_value = 1
    gold_data = pd.read_csv("gld_price_data.csv")

    check_folder_exists("pickle_files")
    check_folder_exists("media_plots")
    # get a overview of csv data

    # print first and last five rows of data to
    print(gold_data.head(), end="\n\n")
    print(gold_data.tail(), end="\n\n")

    print("Number of columns and rows -", gold_data.shape, end="\n\n")
    print("Column name, type, null count and data type - ", end="\n\n")
    print(gold_data.info(), end="\n\n")

    # check number of missing values
    missing_values = gold_data.isnull().sum()
    print("Missing values -")
    print(missing_values, end="\n\n")

    # statiscal measure of data
    print("Statiscal measures of data-")
    print(gold_data.describe(), end="\n\n")

    gold_data = gold_data.drop(["Date"], axis=1)  # date is irrelevant

    # check what values correlate with each other
    correlation = gold_data.corr()
    print("Corelation of all values with eachother")
    print(correlation, end="\n\n")

    # visualuize this correlation
    sns.heatmap(
        correlation,
        cbar=True,
        square=True,
        fmt=".1f",
        annot=True,
        annot_kws={"size": 8},
        cmap="rocket",
    )
    plt.savefig("media_plots/correlation_heatmap.png", bbox_inches="tight")
    # plt.show()

    # correaltion values for gold
    print("Correlation of gold to the rest of the values -")
    print(correlation["GLD"], end="\n\n")

    # how gold prices are distributed
    sns.displot(gold_data["GLD"], color="gold")
    plt.savefig("media_plots/distribution_plot.png", bbox_inches="tight")
    # plt.show()

    X = gold_data.drop(["GLD"], axis=1)  # date is irrelevant
    Y = gold_data["GLD"]

    print("datatype of non-gold values -", type(X))
    print("datatype of gold values -", type(Y))

    print("Value of non gold entites - ")
    print(X, end="\n\n")  # silver, uso , sp500 , usd/eur
    print("Value of gold entites - ")
    print(Y, end="\n\n")  # gold column

    # split data into training data and testing data - test_size is in percentages OF 80-20%, change random state later
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random_state_value
    )

    # check count of training and testing data ie if data is split 80-20
    print("size of training data for non-gold values -", X_train.shape[0])
    print("size of testing data for non-gold values -", X_test.shape[0])
    print("size of training data for gold values -", Y_train.shape[0])
    print("size of testing data for non-gold values -", Y_test.shape[0], end="\n\n")

    # MODeL Training - n_estimators means number of decision trees ----------------------------------------------

    regressor = RandomForestRegressor(n_estimators=100, random_state=random_state_value)
    regressor.fit(X_train, Y_train)

    # model trainig done ---------------------------------------------------------------------------------------

    #  save the model as a pickle file
    joblib.dump(regressor, "pickle_files/regressor.pkl")
    X_test.to_pickle("pickle_files/X_test.pkl")
    Y_test.to_pickle("pickle_files/Y_test.pkl")


if __name__ == "__main__":
    main()
