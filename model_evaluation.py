from joblib import Parallel, delayed
import joblib
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
import pandas as pd
import os.path
  
def check_folder_exists(path_to_folder):
    if not os.path.exists(path_to_folder):
        os.makedirs("./"+path_to_folder)
       
check_folder_exists('pickle_files')
check_folder_exists('media_plots')

# Load the model from the file
regressor = joblib.load('pickle_files/regressor.pkl')
X_test = joblib.load('pickle_files/X_test.pkl')
Y_test = joblib.load('pickle_files/Y_test.pkl')

# Use the loaded model to make predictions and test

test_data_prediciton = regressor.predict(X_test)
print('datatype of prediction made - ', type(test_data_prediciton),end='\n\n')
print('actual prediction made using non-gold values (first five values for simplicity) -')
print(test_data_prediciton[0:5],end='\n\n')

# R squared error to computer error rate
error_score = metrics.r2_score(Y_test,test_data_prediciton)
print('ACCURACY - ',(error_score*100),'%',sep='')

#  FIX BELOW ----------------------------------------------------------------------------------------------------

# visualize error rate
Y_test = list(Y_test)
test_data_prediciton = list(test_data_prediciton)


plt.plot(test_data_prediciton, color='red',label = 'predicted Value')
plt.title('Number of values')
plt.ylabel('GOLD prices')
plt.legend()
plt.tight_layout()
plt.savefig('media_plots/predicted_values.png',bbox_inches='tight')
plt.show()

plt.plot(Y_test,color='blue',label = 'Actual Value')
plt.title('Number of values')
plt.ylabel('GOLD prices')
plt.legend()
plt.tight_layout()
plt.savefig('media_plots/actual_values.png',bbox_inches='tight')
plt.show()


plt.plot(test_data_prediciton, color='red',label = 'predicted Value')
plt.plot(Y_test,color='blue',label = 'Actual Value')
plt.title('Number of values')
plt.ylabel('GOLD prices')
plt.legend()
plt.tight_layout()
plt.savefig('media_plots/price_comparison.png',bbox_inches='tight')
plt.show()