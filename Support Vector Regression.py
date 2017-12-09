# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 19:46:47 2017

@ Title: Machine learning in finance

Team member: Zhenni Xie,

             Yun Mai,

             Tze Fung Lung
             
Part - Support Vector Regression
           
"""

# Introduction: If the company profits go up, you make money, and if the company profits go down, you lose money.
# Is there a way to utalize the price in the past to predict the prices? Yes, machine learning.

# Install dependencies

# csv will help us to load the data from csv file
import csv

# numpy will help us to perform calculation
import numpy as np

# matplotlib will help us plot the model on graph
import matplotlib.pyplot as plt

# sklearn will help to build the predicted model
from sklearn.svm import SVR

# last 30 days historical data for AAPL, three models are built, and they are Linear Model and Polynomial Model to compare the results

# Create a list for dates and prices
dates = []
prices = []

# 

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skip the first row
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return


def predict_price(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1))

	svr_lin = SVR(kernel= 'linear', C= 1e3) # Linear Regression
	svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2) # Polynomical 
	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # RBF
	svr_rbf.fit(dates, prices) 
	svr_lin.fit(dates, prices)
	svr_poly.fit(dates, prices)

	plt.scatter(dates, prices, color= 'black', label= 'Data') 
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # Plot RBF
	plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plot linear
	plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plot polynomial
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.savefig('stocks.png') # Save a image named stocks


	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


get_data('aapl.csv') # Data Source: http://www.nasdaq.com/symbol/aapl/historical
predict_price = predict_price(dates, prices,30)
print(predict_price)
 
# Conclusion: Three model were utalized for three historical stock price for past 30 days.From the plot, we can see that RBF model seems to fit 
#  our data the best. 
