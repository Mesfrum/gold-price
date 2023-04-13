from joblib import Parallel, delayed
import joblib
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
  
# # Save the model as a pickle in a file
# joblib.dump(knn, 'filename.pkl')
  
# # Load the model from the file
regressor = joblib.load('pickle_files/regressor.pkl')
X_test = joblib.load('pickle_files/X_test.pkl')
Y_test = joblib.load('pickle_files/Y_test.pkl')


# # Use the loaded model to make predictions and test

test_data_prediciton = regressor.predict(X_test)
print('datatype of prediction made - ', type(test_data_prediciton),end='\n\n')
print('actual prediction made using non-gold values (first five values for simplicity) -')
print(test_data_prediciton[0:5],end='\n\n')

# R squared error to computer error rate
error_score = metrics.r2_score(Y_test,test_data_prediciton)
print('Error score -',error_score)

#  FIX BELOW ----------------------------------------------------------------------------------------------------
# visualize error rate
Y_test = list(Y_test)

# graph_error_rate = plt.figure(3)
plt.plot(test_data_prediciton, color='red',label = 'predicted Value')
plt.plot(Y_test,color='blue',label = 'Actual Value')
plt.title('NUMber of values')
plt.ylabel('GOLD prices')
plt.legend()
plt.savefig('media_plots/actual_vs_test_prices.png',bbox_inches='tight')
plt.show()
# plt.close()


