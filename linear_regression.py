# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 22:02:07 2017
@author: Data 602 final project - Yun, Jenny, Jim

@ Title: Machine learning in finance

Team member: Zhenni Xie,

             Yun Mai,

             Tze Fung Lung


Research Interest 
We are interested in financial analysis in machine learning. In this project, we will pick the dataset into different method of machine learning as our study target. So this project will help us deep understanding about machine learning using in finance sector.


Data Collection
We will collect data in two different ways:

1) Download history stock price from Yahoo Finance, 

2) Extract data information by python
   We are aim to extract the desired data information from web source into machine learning analysis.

Data Analysis Methods

1. Data Visualization
   We will visualize the data by using Matplotlib library to conduct analysis at certain stock price change over time and make comparsion.

2. Statistical Analysis
   We will use SciPy library to compare recent data with the past to check if there is a significant change in financial data.

3. Predictive analysis
   We want to predict the future price for specific stock based on the history stock price data and we will make a prediction thought simulation mode, logistic regression training and test and so on.

Project Goals

    We will gain more experience with data analysis in Python and get familiar to various python modules, such as Numpy, Pandas, SciPy and Matplotlib. 

    And We can have more idea in using machine learning method.


Part - Linear Regression

Specific the Stock Parameter "GOOGL", "AMZN", and "XOM".

Specific the Index Parameter "SPY", "NDX", and "MNX".

Analysis from 2015-1-1 to Now, and assume 30 days predication

Scrapying the real time financial data by using "pandas_datareader.data" function. 

To quantify the linear relationship between an explanatory variable (x) and response variable (y).

We can then predict the average response for all subjects with a given value of the explanatory variable


"""

import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
import pandas_datareader.data as web

symbols = [
    'GOOGL',   # GOOGLE
    'AMZN',    # AMAZON
    'XOM'      # Exxon Mobil for comparsion with Crude oil
          ]  
print ("Ticker Symbols Specified: {0}".format(symbols))

index = [
    'SPY',    # S&P 500
    'NDX',    # nasdaq
    'MNX'     # SQUARE
          ]  
print ("Index Symbols Specified: {0}".format(index))

now = datetime.datetime.now()
now = str(now.strftime("%Y, %m, %d"))

sttDate = '2015, 1, 1'; 
print ("Time Period Specified: {0} -> {1}".format(sttDate, now))

# Real time data scraping from yahoo
def dataset(z):
        i = z
        df = web.DataReader(i, data_source = 'yahoo', start = sttDate, end = now)
        df.iloc
    
    # choose number of columns
        df = df[['Open',  'High',  'Low',  'Close', 'Volume']] 
        df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100.0
        df['PCT_change']=(df['Close'] - df['Open']) / df['Open'] * 100.0
        df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]
        forecast_col = 'Close'
        df.fillna(value=-99999, inplace=True)
    
    # Predict 30 furture days
        n=30 
        df['label'] = df[forecast_col].shift(-n)
        df=df[:-n]
        #print ("This dataset is for: {0} -> {1}".format(i, df.head()))
        return(df)

# Visualization
def visual (z):
    for i in z:
        df = dataset(i)
        print ("This dataset is for: {0} -> {1}".format(i, df.head()))
        
        df['Close'].plot(figsize=(15,6), color="green")
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()

        df['HL_PCT'].plot(figsize=(15,6), color="red")
        plt.xlabel('Date')
        plt.ylabel('High Low Percentage')
        plt.show()

        df['PCT_change'].plot(figsize=(15,6), color="blue")
        plt.xlabel('Date')
        plt.ylabel('Percent Change')
        plt.show()

# Machine Learning Analysis
def analysis (z):
    for i in z:
        df = dataset(i)
        
        # train and test 90% of data
        N_tt = int(math.ceil(0.1 * len(df)))

        # define X feature by np.array
        X = np.array(df.drop(['label'], 1))
        X = preprocessing.scale(X)
        # 10% difference
        X_lately=X[-N_tt:] 
        # define 90% of data
        X=X[:-N_tt]
        #print(X)
        print ("X is for: {0} -> {1}".format(i, X))
        # define y feature by np.array
        y = np.array(df['label'])
        # 10% difference
        y_known = y[-N_tt:]
        # define 90% of data
        y = y[:-N_tt]      
        #print(y)
        print ("y is for: {0} -> {1}".format(i, y))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        #train and test 90% of data
        for k in ['linear','poly','rbf','sigmoid']:
            clf = svm.SVR(kernel=k)
            clf.fit(X_train, y_train)
            confidence = clf.score(X_test, y_test)
            #print(k,confidence)
            print ("The {1} of {0} is: {2}".format(i, k, confidence))

        #train and test 90% of data
        clf = LinearRegression()
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        #print (accuracy)
        print ("The accuracy of Linear Regression for {0} is {1}".format(i, accuracy))


        # Predict use linear regression
        forecast_set = clf.predict(X_lately)
        diff = forecast_set - y_known 
        diff = diff[:-1] 

        # known averaged difference
        avg_diff=np.average(diff)
        #print (avg_diff)
        print ("The average difference for {0} is {1}".format(i, avg_diff))

        # Standard deviation
        sigma = math.sqrt(np.average((diff-avg_diff)**2))
        #print (sigma)
        print ("The sigma of SD for {0} is {1}".format(i, sigma))
        skewness = np.average(((diff-avg_diff)/sigma)**3)
        #print (skewness)
        print ("The skewness of SD for {0} is {1}".format(i, skewness))
        kurt=np.average(((diff-avg_diff)/sigma)**4)
        #print (kurt)
        print ("The kurt of SD for {0} is {1}".format(i, kurt))
        
        x=range(len(diff))
        plt.figure(1)
        forecastplot=plt.scatter(x,forecast_set[:-1],c='r')
        knownplot=plt.scatter(x,y_known[:-1], c='b')
        plt.legend([forecastplot,knownplot],['forecasted value', 'known value'])
        plt.xlabel('days')
        plt.ylabel('Adj.Close')

        plt.figure(2)  
        plt.scatter(x,diff)
        plt.xlabel('days')
        plt.ylabel('(forcated-known) of Adj.Close')

        plt.figure(3)  
        plt.hist(diff, bins=20)
        plt.title('Histogram of (forcated-known) values of Adj.Close')
        plt.show()

def group (z):
    df1 = pd.DataFrame()
    df = dataset(z[0])
    df1 = df["Close"]
    for i in z[1:]:
        df = dataset(i)
        df4 = df["Close"]
        df1 = pd.concat([df1, df4], axis=1)
    df1.columns = [z[0:]]
    print (df1.head())
    
    df1.plot(grid = True)
    return (df1)

def stock_return (z):
    df1 = group (z)
    stock_return = df1.apply(lambda x: x / x[0])
    stock_return.head()
    stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
    
#print("Ticker analysis: ")
#dataset (symbols)
#visual (symbols)
analysis (symbols)
stock_return(symbols) 

#print("Index analysis: ")
#dataset (index)
#visual (index)
analysis (index)

"""
Conclusion:
By using linear regression method, we compare and find out the relationship between specific stock and index. 
And it use the machine learning to make each of forcasting value, include stock - Google, Amazon, Exxon Mobil and index - Nasadq, SP500 and MNX. 
The function can add any index and stock symbols into comparion. 
Though this project, we use the function of "pandas_datareader.data" to scrape the real time data for ananlysis. 
But we found some constraints in this function "pandas_datareader.data", it is not pretty stable to collect data and need to re-run / re-connect several times if large amount data scrapying.
When using function of "quandl", it have a limited connection times per day, so we use this two into our different method for comparsion.

For the machine learning, we get the pre-loaded data at 12-08-2017 in Jypyter Notebook, the known trend of google is upward, but the forcast trend is downward and accuracy of Linear Regression for GOOGL is 0.86. 
And the known trend of amzn is upward, but the forcast trend is downward and accuracy of Linear Regression for amzn is 0.87. 
But for the XOM, the trend for upward and downward does not show definitely. 

For the SP500, the forcast trend is upward and the acuracy of Linear Regression is 0.75, 
And Nasdaw, no definite trend is shown, but the acuracy of Linear Regression is less than 10%, so the result of ananlysis are meaning less for thsi time.
It has a upward trend in the forcast value for MNX, and the accuracy of Linear Regression is 0.85.
From the result of analysis, the upward trend for index will be predicted for next transaction day.

"""