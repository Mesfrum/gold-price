import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from train_model import check_folder_exists
from model_evaluation import import_pickle_files, check_pickle_exists
import os


def check_prediction_data_exists():
    if os.path.exists(r"pickle_files/test_data_prediciton.pkl") == False:
        from model_evaluation import master
        master()

    test_data_prediction = joblib.load(r"pickle_files/test_data_prediciton.pkl")
    return test_data_prediction


def on_click_event(usd_price, silver_price, uso_price, eur_usd_price):
    prediction = regression.predict(
        [[usd_price, silver_price, uso_price, eur_usd_price]]
    )
    return prediction


check_folder_exists("pickle_files")
test_data_prediction = check_prediction_data_exists()
test_data_prediction = np.array(test_data_prediction)

try:
    regressor, X_test, Y_test = import_pickle_files()
except FileNotFoundError:
    check_pickle_exists()
    regressor, X_test, Y_test = import_pickle_files()

Y_test = list(Y_test)

data = pd.read_csv("gld_price_data.csv", parse_dates=["Date"], dayfirst=False)
data["Date"] = pd.to_datetime(data["Date"]).dt.date

# load regresion model
regression = joblib.load(r"pickle_files/regressor.pkl")

st.set_page_config(page_title="goldey")

with st.container():
    st.subheader("A TE MINI PROJECT")
    st.title("Predicting the prices of gold")

with st.container():
    st.subheader("The parameters are USD, SILVER, US OIL, EUR/USD")
    usd_price = st.number_input("Insert a price for USD:")
    st.write("the number is ", usd_price)
    silver_price = st.number_input("Insert a price for SILVER:")
    st.write("the number is ", silver_price)
    uso_price = st.number_input("Insert a price for US Oil:")
    st.write("the number is ", uso_price)
    eur_usd_price = st.number_input("Insert a price for EUR vs USD:")
    st.write("the number is ", eur_usd_price)

    if st.button("Calculate"):
        prediction = on_click_event(usd_price, silver_price, uso_price, eur_usd_price)
        st.write("The predicted price of GOLD is")
        st.subheader(float(prediction))
        st.subheader('USD/OUNCE')

with st.container():
    st.write("-----------")
    st.write("DATA")
    st.dataframe(data, width=704, height=300)

with st.container():
    st.write("-----------")
    st.write("Gold Data Distribution plot")
    st.image(r"media_plots/distribution_plot.png", width=704)

    st.write("-----------")
    st.write("Corelation of gold with other parameters heatmap -")
    st.image(r"media_plots/correlation_heatmap.png", width=704)

    st.write("-----------")
    st.write("Actual data")
    st.line_chart(Y_test, width=704)

    st.write("-----------")
    st.write("Predicted data")
    st.line_chart(test_data_prediction.tolist(), width=1204, height=400)

    st.write("-----------")
    st.write("Price Difference between actual and predicted price vs Gold price Residual plot")
    st.image(r"media_plots/error_rate.png", width=704)
