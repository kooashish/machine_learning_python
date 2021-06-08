from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import urllib.request as rq

rq.urlretrieve("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv", "FuelConsumption.csv")
df = pd.read_csv("FuelConsumption.csv")
cdf = df[['ENGINESIZE','CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Method to split a data set in train and test
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Method to split a data set in test and train set
x = cdf[['ENGINESIZE']]
y = cdf[['CO2EMISSIONS']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
regr = LinearRegression()
regr.fit(x_train,y_train)

# Not the best method as there could be a overlap but this is only for demonstration purpose
# Best method is to use the test dataset created which is test_x and predict pred_y and compare the results
# with y_test to show the model's efficacy
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)


print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )
