import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from train_model import check_folder_exists
from model_evaluation import import_pickle_files,check_pickle_exists
import os

def check_prediction_data_exists():
    if os.path.exists(r"pickle_files/test_data_prediciton.pkl") == False:
        from model_evaluation import master
        master()
        
    test_data_prediction = joblib.load(r"pickle_files/test_data_prediciton.pkl")
    return test_data_prediction


def on_click_event(usd_price,silver_price,uso_price,eur_usd_price):
    prediction = regression.predict([[usd_price,silver_price,uso_price,eur_usd_price]])
    flag = 1
    return [prediction,flag]

check_folder_exists("pickle_files")
test_data_prediction =  check_prediction_data_exists()
test_data_prediction =  np.array(test_data_prediction)

try:
    regressor,X_test,Y_test = import_pickle_files()
except FileNotFoundError:
    check_pickle_exists()
    regressor,X_test,Y_test = import_pickle_files()

Y_test = list(Y_test)

data = pd.read_csv("gld_price_data.csv", parse_dates=["Date"], dayfirst=False)
data['Date'] = pd.to_datetime(data['Date']).dt.date 

parameters = X_test.columns.values
print(parameters)

fig,ax = plt.subplots()
sns.set_palette("bright")
sns.set_style("dark")

sns.set_style(rc = {'axes.facecolor': 'black', 'figure.facecolor' : 'black'}) 
for tick_label in ax.axes.get_yticklabels():
    tick_label.set_color("white")
for tick_label in ax.axes.get_xticklabels():
    tick_label.set_color("white")
sns.residplot(x = Y_test, y =test_data_prediction, color = 'lightblue')
     
regression = joblib.load(r'pickle_files/regressor.pkl')

st.set_page_config(page_title="goldey")
              
with st.container():
    st.subheader("A TE MINI PROJECT")
    st.title("Predicting the prices of gold")

with st.container():
    st.subheader('The parameters are USD, SILVER, US OIL, EUR/USD')
    usd_price = st.number_input('Insert a price for USD:')
    st.write('the number is ', usd_price)
    silver_price = st.number_input('Insert a price for SILVER:')
    st.write('the number is ', silver_price)
    uso_price = st.number_input('Insert a price for US Oil:')
    st.write('the number is ', uso_price)
    eur_usd_price = st.number_input('Insert a price for EUR vs USD:')
    st.write('the number is ', eur_usd_price)
    
    if st.button('Calculate'):
        prediction = on_click_event(usd_price,silver_price,uso_price,eur_usd_price)
        st.write('The predicted price of GOlD is')
        st.subheader(float(prediction[0]))

with st.container():
    st.write("-----------")
    st.write("DATA")
    st.dataframe(data, width=704, height=300)

with st.container():
    st.write("-----------")
    st.write('Actual data')
    st.line_chart(Y_test,width=704)
    
    st.write("-----------")
    st.write('Predicted data')
    st.line_chart(test_data_prediction.tolist(),width=1204,height=400)

    st.write("-----------")
    st.write('Actual data vs predicted data Residual plot')
    st.pyplot(fig= fig)
