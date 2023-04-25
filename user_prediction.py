from joblib import Parallel, delayed
import joblib
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
import pandas as pd
import numpy as np

regression = joblib.load(r'pickle_files/regressor.pkl')

prediction = regression.predict([[80,280,40,60]])

print(prediction)