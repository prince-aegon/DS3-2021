"""
NAME       - SARTHAK SAPTAMI KUMAR JHA
ROLL NO.   - B20317
MOBILE NO. - 8825319259
BRANCH     - Computer Science and Engineering (CSE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.ar_model import AutoReg as AR
from statsmodels.graphics.tsaplots import plot_acf

def fig_1(new_cases):
    plt.figure(1)
    plt.plot([i for i in range(0, 612)], new_cases)
    plt.xticks([0, 60-1, 120-1, 180-1, 240-1, 300-1, 360-1, 420-1, 480-1, 540-1, 600-1], ['FEB-20', 'APR-20', 'JUN-20','AUG-20', 'OCT-20', 'DEC-20', 'FEB-21', 'APR-21', 'JUN-21','AUG-21', 'OCT-21'])
    plt.yticks([0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000], ['0', '50K', '100K', '150K', '200K', '250K', '300K', '350K', '400K'])
    plt.xlabel('Month-Year')
    plt.ylabel('New Confirmed cases')
    plt.title('Number of new confirmed cases vs Month-Year')
    plt.show()

def fig_2(x):
    plt.figure(2)
    plt.title('Scatter plot b/w one time lag and current time series')
    plt.xlabel('Current time series')
    plt.ylabel('One time lag series')
    plt.xticks([0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000], ['0', '50K', '100K', '150K', '200K', '250K', '300K', '350K', '400K'])
    plt.yticks([0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000], ['0', '50K', '100K', '150K', '200K', '250K', '300K', '350K', '400K'])
    plt.scatter(x[1:], x[:len(x)-1], s = 3.5)
    plt.show()

def fig_3(cr_list):
    plt.figure(3)
    plt.title('Correlation coefficients for various lag data')
    plt.xlabel('Lag in the time series data')
    plt.ylabel('Pearson correlation coefficient')
    plt.plot([i for i in range(1, r_Range)], cr_list)
    plt.show()

def fig_4(predictions, test_x):
    plt.figure(4)
    plt.title('Scatter plot b/w Actual and Predicted values')
    plt.ylabel('Actual values')
    plt.xlabel('Predicted values')
    plt.xticks([0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000],['0', '50K', '100K', '150K', '200K', '250K', '300K', '350K', '400K'])
    plt.yticks([0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000],['0', '50K', '100K', '150K', '200K', '250K', '300K', '350K', '400K'])
    plt.scatter(predictions, test_x, s=5)
    plt.show()

def fig_5(predictions, test_x):
    plt.figure(5)
    plt.title('Line plots of Actual and Predicted data')
    plt.xlabel('Month-Year')
    plt.ylabel('Number of new cases')
    # plt.xticks([0, 60 - 1, 120 - 1, 180 - 1, 240 - 1, 300 - 1, 360 - 1, 420 - 1, 480 - 1, 540 - 1, 600 - 1],['FEB-20', 'APR-20', 'JUN-20', 'AUG-20', 'OCT-20', 'DEC-20', 'FEB-21', 'APR-21', 'JUN-21', 'AUG-21','OCT-21'])
    plt.yticks([0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000], ['0', '50K', '100K', '150K', '200K', '250K', '300K', '350K', '400K'])
    plt.plot([i for i in range(len(predictions))], predictions, label = "Prediction")
    plt.plot([i for i in range(len(test_x))], test_x, label = "Actual")
    plt.legend(loc="upper right")
    plt.show()

def fig_6(rmse):
    plt.figure(6)
    plt.title('RMSE error for various lag values')
    plt.xlabel('Lag values')
    plt.ylabel('RMSE Error(in %)')
    plt.xticks([0, 1, 2, 3, 4], [1, 5, 10, 15, 25])
    plt.bar([i for i in range(5)], rmse)
    plt.show()

def fig_7(mape):
    plt.figure(7)
    plt.title('MAPE error for various lag values')
    plt.xlabel('Lag values')
    plt.ylabel('MAPE Error(in %)')
    plt.xticks([0, 1, 2, 3, 4], [1, 5, 10, 15, 25])
    plt.bar([i for i in range(5)], mape)
    plt.show()


def autocorrelation(windows, train_x, test_x, index):
    rmse = []
    mape = []
    coefs = []
    for window in windows:

        model = AR(train_x, lags=window)
        model_fit = model.fit()  # fit/train the model
        coef = model_fit.params  # Get the coefficients of AR model
        # using these coefficients walk forward over time steps in test, one
        # step each time
        coefs.append(coef)
        history = train_x[len(train_x) - window:]
        history = [history[i] for i in range(len(history))]
        predictions = []  # List to hold the predictions, 1 step at a time
        for t in range(len(test)):
            length = len(history)
            lag = [history[i] for i in range(length - window, length)]
            yhat = coef[0]  # Initialize to w0
            for d in range(window):
                yhat += coef[d + 1] * lag[window - d - 1]  # Add other values
            obs = test_x[t]
            predictions.append(yhat)  # Append predictions to compute RMSE
            # later
            history.append(obs)  # Append actual test value to history, to be
            # used in next step   .
        if index == 1:
            fig_4(predictions, test_x)
            fig_5(predictions, test_x)


        # Caluclating and storing RMSE and MAPE errors
        s = 0
        s1 = 0
        s2 = 0

        for i in range(len(predictions)):
            s += (predictions[i] - test_x[i]) ** 2
            s1 += test_x[i]
            s2 += (abs(predictions[i] - test_x[i])) / test_x[i]

        s /= len(predictions)
        s1 /= len(test_x)
        s = s ** 0.5


        rmse.append((s / s1) * 100)

        mape.append((s2 / len(test_x)) * 100)

    return rmse, mape, coefs



# Creating a dataframe
df = pd.read_csv("daily_covid_cases.csv")
display_plots = 1
# Selecting new_cases row
new_cases = df['new_cases']

# Plotting the number of new cases against month-year
if display_plots:
    fig_1(new_cases)

# Creating a numpy array for correlation calculation
x = np.array(df['new_cases'])

# Calculating correlation coefficient for one lag time series
cr1 = np.corrcoef(x[1:], x[:len(x)-1])
print('Pearson’s correlation coefficient for one-day lag:', round(cr1[0][1], 3))
print()

# Plotting scatter plot between lag time series and current time series
if display_plots:
    fig_2(x)

# Calculating correaltion coefficient for various lag values
cr_list = []
r_Range = 7
for p in range(1, r_Range):
    cr = np.corrcoef(x[p:], x[:len(x)-p])
    cr_list.append(cr[0][1])

# Plotting Pearson correlation coefficient against various lag values
print('Pearson’s correlation coefficient for 6 days:', [round(i, 3) for i in cr_list])
print()
if display_plots:
    fig_3(cr_list)

# Autocorrelation or Correlogram plot
if display_plots:
    plt.figure(8)
    plot_acf(df['new_cases'])
    plt.show()


# Splitting the data into test and training portions
test_size = 0.35
X = df.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]


# Creating lists of number of cases in train and test sets
# train_x -> all training case values
# test_x -> all test case values

train_x = []
for i in range(len(train)):
    train_x.append(train[i][1])

test_x = []
for i in range(len(test)):
    test_x.append(test[i][1])


# Conducting AutoRegression for 5 lag values
rmse, mape, coefs = autocorrelation([5], train_x, test_x, 1)


# Printing coefficients of the AR model
print('Coefficients from trained AR model:')
print([round(i, 3) for i in coefs[0]])
print()

# Printing RMSE and MAPE values
print('Errors for lag values: 5')
print('RMSE and MAPE values are:', round(rmse[0], 3), 'and', round(mape[0], 3))
print()



# Conudcting AutoRegression for lag values: 1, 5, 10, 15, 25
rmse, mape, coefs = autocorrelation([1, 5, 10, 15, 25], train_x, test_x, 0)


# Calculating RMSE and MAPE values for lag values, 1, 5, 10, 15, 25
print('Errors for lag values: 1, 5, 10, 15, 25')
print('RMSE values are:', [round(i, 3) for i in rmse])
print('MAPE values are:', [round(i, 3) for i in mape])
print()

# Plotting bar chart for rmse and mape values for various lag values
if display_plots:
    fig_6(rmse)
    fig_7(mape)


# Calculating autocorrelation limits
acr = (2/((len(test_x)**0.5)))


# Calculating that for which lag value, autocorrelation is deemed significant
r_Range = 100
cr_list = []

for p in range(1, r_Range):
    cr = np.corrcoef(x[p:], x[:len(x)-p])
    cr_list.append(cr[0][1])

# Calculating autocorrelation for 100 values
dict = {'cr': cr_list, 'lag': [i for i in range(1, r_Range)]}
dg = pd.DataFrame(dict)

autocorr = []
for i in range(len(dg)):
  if abs(dg.loc[i, 'cr']) < 2/((len(test_x)**0.5)):
      autocorr.append(dg.loc[i, 'lag'])

# Lag values after 'limit' will be deemed insignificant for autoregression
limit = max(autocorr) + 1
print('Optimal number of lags:', limit)

# Calculating RMSE and MAPE values for limit - 1 lag values
rmse, mape, coefs = autocorrelation([limit], train_x, test_x, 0)
print('Errors for lag values as calculated using autocorrelation')
print('RMSE and MAPE values are:', round(rmse[0], 3), 'and', round(mape[0], 3))