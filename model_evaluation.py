from joblib import Parallel, delayed
import joblib
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
import pandas as pd
import numpy as np
import os.path
from train_model import check_folder_exists

def check_pickle_exists():
    if (
        not os.path.exists("pickle_files/X_test.pkl")
        or not os.path.exists("pickle_files/Y_test.pkl")
        or not os.path.exists("pickle_files/regressor.pkl")
    ):
        from train_model import main
        main()

def import_pickle_files():
    regressor = joblib.load("pickle_files/regressor.pkl")
    X_test = joblib.load("pickle_files/X_test.pkl")
    Y_test = joblib.load("pickle_files/Y_test.pkl")
    return regressor,X_test,Y_test

def master():
    check_folder_exists("pickle_files")
    check_folder_exists("media_plots")

    # Load the model from the file
    try:
        regressor,X_test,Y_test = import_pickle_files()
    except FileNotFoundError:
        check_pickle_exists()
        regressor,X_test,Y_test = import_pickle_files()

    # Use the loaded model to make predictions and test
    test_data_prediciton = regressor.predict(X_test)
    
    #  load test data as dataframe to pcikle for later use
    pd.DataFrame(test_data_prediciton).to_pickle("pickle_files/test_data_prediciton.pkl")
    
    print("datatype of prediction made - ", type(test_data_prediciton), end="\n\n")
    print(
        "actual prediction made using non-gold values (first five values for simplicity) -"
    )
    print(test_data_prediciton[0:5], end="\n\n")

    # R squared error to computer error rate
    error_score = metrics.r2_score(Y_test, test_data_prediciton)
    print("ACCURACY - ", (error_score * 100), "%", sep="")

    # visualize error rate
    Y_test = list(Y_test) 
    test_data_prediciton = list(test_data_prediciton)
    
    # fix below repition
    plt.plot(Y_test, color="blue", label="Actual Value")
    plt.title("Number of values")
    plt.ylabel("GOLD prices")
    plt.legend()
    plt.tight_layout()
    plt.savefig("media_plots/actual_values.png", bbox_inches="tight")
    plt.close()
    # plt.show()

    plt.plot(test_data_prediciton, color="red", label="predicted Value")
    plt.title("Number of values")
    plt.ylabel("GOLD prices")
    plt.legend()
    plt.tight_layout()
    plt.savefig("media_plots/predicted_values.png", bbox_inches="tight")
    plt.close()
    # plt.show()

    plt.plot(test_data_prediciton, color="red", label="predicted Value",alpha = 0.5)
    plt.plot(Y_test, color="blue", label="Actual Value",alpha = 0.5)
    plt.title("Number of values")
    plt.ylabel("GOLD prices")
    plt.legend()
    plt.tight_layout()
    plt.savefig("media_plots/price_comparison.png", bbox_inches="tight")
    plt.close()
    # plt.show()
    
    # residual plot 
    fig,ax = plt.subplots()
    sns.set_style("whitegrid")
    fig.set_facecolor('black')
    ax.set_facecolor('black')
    ax.axes.set_xlabel('actual')
    ax.axes.set_ylabel('predicted')
    
    for tick_label in ax.axes.get_yticklabels():
        tick_label.set_color("white")
    for tick_label in ax.axes.get_xticklabels():
        tick_label.set_color("white")
    
    for _,s in ax.spines.items():
        s.set_color('cyan')
    plt.grid()
    
    sns.residplot(x = Y_test, y = test_data_prediciton, color = 'cyan',label= 'Gold prices')
    fig.savefig(r'media_plots/error_rate.png')
    plt.show()
    
if __name__ == '__main__':
    master()