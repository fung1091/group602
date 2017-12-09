# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:48:31 2017

@ Title: Machine learning in finance

Team member: Zhenni Xie,

             Yun Mai,

             Tze Fung Lung

"""


# get the date and time for blotter and PL table
import pandas as pd
import numpy as np
from time import sleep

import matplotlib

import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')

from datetime import datetime

import quandl

from statsmodels.formula.api import ols, OLS
import statsmodels.api as slm

import math
from sklearn import cross_validation, svm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_regression

import plotly
import plotly.plotly as py
import plotly.graph_objs

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
1. Data
Get the historical stock price, S&P500 index, NASDAQ100 index, gold price,and crude oil price
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 

# historical stock price
stock_to_study = ['AAPL','GOOGL','MSFT','AMZN','EBAY','JPM','JNJ','GE','NVDA','XOM']# some other ,'XOM','NVDA','TSLA','TQQQ'

 
def StockData(stock_list):
    stock_history = pd.DataFrame()
    for ticker_symbol in stock_list:
        price_data = quandl.get("WIKI/"+str(ticker_symbol))
        price_data['ticker'] = ticker_symbol
        stock_history = pd.concat([stock_history,price_data])
        sleep(5)
    return stock_history

stock_history = StockData(stock_to_study)
#descendant
stock_history = stock_history[::-1]

#stock_history.head(2)
#stock_history.tail(2)
#stock_history.shape

#sp500 index data
#sp500_1 = quandl.get("CHRIS/CME_ES4", authtoken="Yi8bTkCQLyrhGVcuPw9U")
#sp500_2 = quandl.get("CHRIS/CME_ES3", authtoken="Yi8bTkCQLyrhGVcuPw9U")
sp500 = quandl.get("CHRIS/CME_ES2", authtoken="Yi8bTkCQLyrhGVcuPw9U")

#nasdaq100 index data
nasdap100 = quandl.get("NASDAQOMX/NDX", authtoken="Yi8bTkCQLyrhGVcuPw9U")

# gold price data
gold = quandl.get("WGC/GOLD_DAILY_USD", authtoken="Yi8bTkCQLyrhGVcuPw9U")

# WTI crude oil data
wtioil = quandl.get("ODA/POILWTI_USD", authtoken="Yi8bTkCQLyrhGVcuPw9U")

#manipulate the long table
#dif = stock_history.groupby('ticker').apply(lambda x: (100*(x['Adj. Close'] - x['Adj. Close'].shift(-1))/x['Adj. Close'].shift(-1)))
#dif = dif.reset_index('ticker')
#dif = dif.rename(columns = {'Adj. Close':'Daily_Return'})

#daily return of sp500
dailyR_SP = 100*(sp500['Last'] - sp500['Last'].shift(-1))/sp500['Last'].shift(-1)
dailyR_SP  = dailyR_SP.rename("Daily_Return") 
sp500 = pd.concat([sp500, dailyR_SP], axis=1)
#add surfix to column names of sp500
sp500_copy = sp500.copy()
sp500_copy.columns = sp500_copy.columns.map(lambda x: str(x) + '_sp')


#daily return of nasdap100
dailyR_NSDQ = 100*(nasdap100['Index Value'] - nasdap100['Index Value'].shift(-1))/nasdap100['Index Value'].shift(-1)
dailyR_NSDQ = dailyR_NSDQ.rename("Daily_Return") 
nasdap100 = pd.concat([nasdap100, dailyR_NSDQ], axis=1)
#add surfix to column names of sp500
nasdap100_copy = nasdap100.copy()
nasdap100_copy.columns = nasdap100_copy.columns.map(lambda x: str(x) + '_nsdq')


##daily return of gold value
dailyR_G = 100*(gold['Value'] - gold['Value'].shift(-1))/gold['Value'].shift(-1)
dailyR_G = dailyR_G.rename("Daily_Return") 
gold = pd.concat([gold, dailyR_G], axis=1)
gold['Up/Down'] = np.where(gold['Daily_Return']>0, 'up', 'down')
#add surfix to column names of sp500
gold_copy = gold.copy()
gold_copy.columns = gold_copy.columns.map(lambda x: str(x) + '_gold')


##daily return of gold value
dailyR_OL = 100*(wtioil['Value'] - wtioil['Value'].shift(-1))/wtioil['Value'].shift(-1)
dailyR_OL = dailyR_OL.rename("Daily_Return") 
wtioil = pd.concat([wtioil, dailyR_OL], axis=1)
wtioil['Up/Down'] = np.where(wtioil['Daily_Return']>0, 'up', 'down')
#add surfix to column names of sp500
wtioil_copy = wtioil.copy()
wtioil_copy.columns = wtioil_copy.columns.map(lambda x: str(x) + '_oil')

print('Stock list:', pd.DataFrame(stock_to_study))
No = input('Please select the stcok from the list(input the No.):')
No = int(No)

# get ticker #1 data
t1 = stock_history.loc[stock_history.ticker==stock_to_study[No]]
# daily return
dailyR = 100*(t1['Adj. Close'] - t1['Adj. Close'].shift(-1))/t1['Adj. Close'].shift(-1)
dailyR = dailyR.rename("Daily_Return") 
t1 = pd.concat([t1, dailyR], axis=1)
t1['Up/Down'] = np.where(t1['Daily_Return']>0, 'up', 'down')
#add surfix to column names of ticker #1 
t1.columns = t1.columns.map(lambda x: str(x) + "_" + stock_to_study[No])


#join the ticker #1 with the sp500 data
t1_sp = pd.concat([t1, sp500_copy], axis=1, join='inner')
t1_sp['StatusToSP'+ '_' + stock_to_study[No]] = np.where(t1_sp['Daily_Return' + '_' + stock_to_study[No]] > t1_sp['Daily_Return_sp'], 'Outperform', 'Underperform')

#join the ticker #1 with the nasdaq100 data
t1_nq = pd.concat([t1,nasdap100_copy], axis=1, join='inner')
t1_nq['StatusToNasdaq'+ '_' + stock_to_study[No]] = np.where(t1_nq['Daily_Return' + '_' + stock_to_study[No]] > t1_nq['Daily_Return_nsdq'], 'Outperform', 'Underperform')

#join the ticker #1 with the sp500 and the nasdaq100 data
t1_sp_nq = pd.concat([t1_sp,nasdap100_copy], axis=1, join='inner')
t1_sp_nq['StatusToNasdaq'+ '_' + stock_to_study[No]] = np.where(t1_sp_nq['Daily_Return' + '_' + stock_to_study[No]] > t1_sp_nq['Daily_Return_nsdq'], 'Outperform', 'Underperform')
t1_sp_nq['StatusToSN'+ '_' + stock_to_study[No]] = np.where((t1_sp_nq['Daily_Return' + '_' + stock_to_study[No]] > t1_sp_nq['Daily_Return_sp']) & (t1_sp_nq['Daily_Return' + '_' + stock_to_study[No]] > t1_sp_nq['Daily_Return_nsdq']), 'Outperform', 'Underperform')

#join the ticker #1 with the sp500, the nasdaq100 data, and the crude oil data
tem = pd.concat([t1_sp_nq,gold_copy], axis=1, join='inner')
t1_multi_index = pd.concat([tem, wtioil_copy], axis=1, join='inner')

#join the ticker #1 with the sp500 and the gold data
t1_sp_g = pd.concat([t1_sp,gold_copy], axis=1, join='inner')

#join the ticker #1 with the sp500, the nasdaq100 data, and the gold data
t1_sp_nq_g = pd.concat([t1_sp_nq,gold_copy], axis=1, join='inner')

#plot the stock price
trace0 = plotly.graph_objs.Candlestick(
    x=t1_sp_nq.index,
    open=t1_sp_nq['Adj. Close'+ "_" + stock_to_study[No]],
    high=t1_sp_nq['Adj. High'+ "_" + stock_to_study[No]],
    low=t1_sp_nq['Adj. Low'+ "_" + stock_to_study[No]],
    close=t1_sp_nq['Adj. Close'+ "_" + stock_to_study[No]]
)
"""
trace1 = plotly.graph_objs.Scatter(
    x = t1_sp_nq.index,
    y = t1_sp_nq['Last_sp'],
    name = 'lookback_predicted',
    line = dict(
        color = ('rgb(145.0, 191.0, 219.0)'),
        width = 4)
)
trace2 = plotly.graph_objs.Scatter(
    x = t1_sp_nq.index,
    y = t1_sp_nq['Index Value_nsdq'],
    name = 'pecentage difference',
    line = dict(
        color = ('rgb(252.0, 141.0, 89.0)'),
        width = 4)
) 
"""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
2. Use linear regression to define the features
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 

""""""""""""""""""""""""""""""""""""""""" 
2.1 single variable linear regression 
""""""""""""""""""""""""""""""""""""""""" 

# s&p500 index vs. stock price
y = t1_sp['Adj. Close' + '_' + stock_to_study[No]]
x = t1_sp ['Last_sp']
model = ols('y~x', t1_sp)
results = model.fit()
r2_sp = results.rsquared #model's r-squared value
p_sp = results.pvalues # p-values and corresponding coefficients in model

# s&p500 index vs. stock price
y = t1_sp['Adj. Close' + '_' + stock_to_study[No]]
x = t1_sp ['Volume_sp']
model = ols('y~x', t1_sp)
results = model.fit()
r2_spv = results.rsquared #model's r-squared value
p_spv = results.pvalues # p-values and corresponding coefficients in model

# s&p500 index vs. stock price
y = t1_nq['Adj. Close' + '_' + stock_to_study[No]]
x = t1_nq ['Index Value_nsdq']
model = ols('y~x', t1_nq)
results = model.fit()
r2_nq = results.rsquared #model's r-squared value
p_nq = results.pvalues # p-values and corresponding coefficients in model


# stock price up/down vs. gold value up/down
X = t1_multi_index.copy()
catg_x = 'Up/Down_gold' 
catg_y = 'Up/Down'+ '_' + stock_to_study[No]
y = t1_multi_index[catg_y].replace('up',1).replace('down',0)
y.groupby(X[catg_x]).mean() # compute percentage of performance for the Adj. Close
X['Up/Down_gold_num'] = pd.Categorical(X[catg_x]).codes # encode df.famhist as a numeric via pd.Factor
x=X['Up/Down_gold_num']
est = ols(formula="y ~ x", data=X).fit()
r2_g1 = est.rsquared
p_g1 = est.pvalues

# stock price up/down vs. crude oil value up/down
X = t1_multi_index.copy()
catg_x = 'Up/Down_oil' 
catg_y = 'Up/Down'+ '_' + stock_to_study[No]
y = t1_multi_index[catg_y].replace('up',1).replace('down',0)
y.groupby(X[catg_x]).mean() # compute percentage of performance for the Adj. Close
X['Up/Down_oil_num'] = pd.Categorical(X[catg_x]).codes # encode df.famhist as a numeric via pd.Factor
x=X['Up/Down_oil_num']
est = ols(formula="y ~ x", data=X).fit()
r2_o = est.rsquared
p_o = est.pvalues

# stock price up/down vs. gold value up/down
X = t1_sp_g.copy()
catg_x = 'Up/Down_gold' 
catg_y = 'Up/Down'+ '_' + stock_to_study[No]
y = t1_sp_g[catg_y].replace('up',1).replace('down',0)
y.groupby(X[catg_x]).mean() # compute percentage of performance for the Adj. Close
X['Up/Down_gold_num'] = pd.Categorical(X[catg_x]).codes # encode df.famhist as a numeric via pd.Factor
x=X['Up/Down_gold_num']
est = ols(formula="y ~ x", data=X).fit()
r2_g2 = est.rsquared
p_g2 = est.pvalues

# stock price up/down vs. status to benchmark, s&p500 index
X = t1_sp.copy()
catg_x = 'StatusToSP'+ '_' + stock_to_study[No]
catg_y = 'Up/Down'+ '_' + stock_to_study[No]
y = t1_sp[catg_y].replace('up',1).replace('down',0)
y.groupby(X[catg_x]).mean() # compute percentage of performance for the Adj. Close
X['StatusToSP_num'] = pd.Categorical(X[catg_x]).codes # encode df.famhist as a numeric via pd.Factor
x=X['StatusToSP_num'].replace({0:1, 1:0})
est = ols(formula="y ~ x", data=X).fit()
r2_StatusToSP = est.rsquared
p_StatusToSP = est.pvalues

# stock price up/down vs. status to benchmark, nasdaq100 index
X = t1_nq.copy()
catg_x = 'StatusToNasdaq'+ '_' + stock_to_study[No]
catg_y = 'Up/Down'+ '_' + stock_to_study[No]
y = t1_nq[catg_y].replace('up',1).replace('down',0)
y.groupby(X[catg_x]).mean() # compute percentage of performance for the Adj. Close
X['StatusToNQ_num'] = pd.Categorical(X[catg_x]).codes # encode df.famhist as a numeric via pd.Factor
x=X['StatusToNQ_num'].replace({0:1, 1:0})
est = ols(formula="y ~ x", data=X).fit()
r2_statusToNQ = est.rsquared
p_statusToNQ = est.pvalues


# stock price up/down vs. status to benchmark, nasdaq100 index
X = t1_sp_nq.copy()
catg_x = 'StatusToSN'+ '_' + stock_to_study[No]
catg_y = 'Up/Down'+ '_' + stock_to_study[No]
y = t1_sp_nq[catg_y].replace('up',1).replace('down',0)
y.groupby(X[catg_x]).mean() # compute percentage of performance for the Adj. Close
X['StatusToSN_num'] = pd.Categorical(X[catg_x]).codes # encode df.famhist as a numeric via pd.Factor
x=X['StatusToSN_num'].replace({0:1, 1:0})
est = ols(formula="y ~ x", data=X).fit()
r2_statusToSN = est.rsquared
p_statusToSN = est.pvalues

linear_test = pd.DataFrame({'features':['sp500','nasdaq100','gold_1','oil','gold_2','StatusToSP', 'statusToNQ','statusToSN'],
              'r2':[r2_sp,r2_nq,r2_g1,r2_o,r2_g2, r2_StatusToSP, r2_statusToNQ, r2_statusToSN],
              'p-value':[p_sp[1],p_nq[1],p_g1[1],p_o[1], p_g2[1], p_StatusToSP[1], p_statusToNQ[1], p_statusToSN[1]]},columns=['features','r2','p-value'])

print(linear_test)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
    From single variable regression, we know that S&P500 index, NASDAQ100 
index are good indicators with R squared as 0.76 and 0.94 respectively. 
We also know that the crude oil and gold price are not good indicators as 
the R squared are close to 0 and p-values are high. In addition, oil price 
data is updated monthly so the data set is relatively small. In the multiple 
regression analysis, we will not consider crude oil data.
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 

""""""""""""""""""""""""""""""""""""""""" 
2.2 Multiple regression 
""""""""""""""""""""""""""""""""""""""""" 

X = t1_sp_nq_g.copy()


# trun the categorical data in X to numerical data
obj_col =(['StatusToSP'+ '_' + stock_to_study[No],
          'StatusToNasdaq'+ '_' + stock_to_study[No],
          'StatusToSN'+ '_' + stock_to_study[No],
          'Up/Down'+ '_' + stock_to_study[No],
          'Up/Down_gold'])
    
for obj in obj_col:
    X[obj+'_num'] = pd.Categorical(X[obj]).codes
    X[obj+'_num'] = X[obj+'_num'].replace({0:1, 1:0})
    

a = 'Adj. Volume'+'_'+stock_to_study[No]
b = 'Daily_Return'+'_'+stock_to_study[No]
c = 'StatusToSP'+ '_' + stock_to_study[No]+'_num'
d = 'StatusToNasdaq'+ '_' + stock_to_study[No]+'_num'
e = 'StatusToSN'+ '_' + stock_to_study[No]+'_num'
f = 'Up/Down'+ '_' + stock_to_study[No]+'_num'
g = 'Up/Down_gold'+'_num'

# set the label,y1 is the close price, y2 is up/down of the close price
y1 = X['Adj. Close'+'_'+stock_to_study[No]]
y2 = X[f]



var_set1 = ([a,b,c,d,e,f,g,'Last_sp', 'Change_sp', 'Volume_sp', 'Previous Day Open Interest_sp',
            'Daily_Return_sp', 'Index Value_nsdq', 'Total Market Value_nsdq', 
            'Dividend Market Value_nsdq', 'Daily_Return_nsdq', 'Value_gold', 
            'Daily_Return_gold'])
var_set2 = ([a,b,c,d,e,f,g,'Volume_sp', 'Total Market Value_nsdq'])

var_set = [var_set1,var_set2]


## fit a OLS model with intercept on TV and Radio
drop_var = (['Adj. Close'+'_'+stock_to_study[No],
             'ticker'+'_'+stock_to_study[No] ,
             ]+obj_col)

'''''''''''''''''''''''''''
Label the Ajust close price 
'''''''''''''''''''''''''''
# var_set1 vs y1
x = X.drop(drop_var,1)
x = x[var_set1]
x = slm.add_constant(x)
est = ols('y1~ x',data=X).fit()
#est.summary()
comb1 = est.pvalues[est.pvalues<0.05].index

# var_set2 vs y1
x = X.drop(drop_var,1)
x = x[var_set2]
x = slm.add_constant(x)
est = ols('y1~ x',data=X).fit()
#est.summary()
comb2 = est.pvalues[est.pvalues<0.05].index
  
'''''''''''''''''''''''''''''
Label the up/down close price 
'''''''''''''''''''''''''''''
# var_set1 vs y2
x = X.drop(drop_var,1)
x = x[var_set1]
x = slm.add_constant(x)
est = ols('y2~ x',data=X).fit()
#est.summary()
comb3 = est.pvalues[est.pvalues<0.05].index

# var_set2 vs y2
x = X.drop(drop_var,1)
x = x[var_set2]
x = slm.add_constant(x)
est = ols('y2~ x',data=X).fit()
#est.summary()
comb4 = est.pvalues[est.pvalues<0.05].index

# the feature sets with p-values < 0.05  
pvalue_set = [comb1,comb2,comb3,comb4]

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
find the feature sets which has more varibles for y1 or y2 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#convert the index of features sets to int  
for i in range(0,4):
    try:
        pvalue_set[i] = pvalue_set[i].str.extract('(\d+)')
        pvalue_set[i] = [x for x in pvalue_set[i] if str(x) != 'nan']
        pvalue_set[i] = list(map(int, pvalue_set[i]))
    except:
        pvalue_set[i] = []
        
featureset_for_y1 = pvalue_set[0:2]
featureset_for_y2 = pvalue_set[2:4]

# for y1
featureset1_index = max(enumerate(featureset_for_y1),  key = lambda tup: len(tup[1]))[0]
feature1_index = list(max(enumerate(featureset_for_y1), key = lambda tup: len(tup[1]))[1])

# for y2
featureset2_index = max(enumerate(featureset_for_y2), key = lambda tup: len(tup[1]))[0]
feature2_index = max(enumerate(featureset_for_y2), key = lambda tup: len(tup[1]))[1]

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
3. SVM model:regression
    
   Build the Model using the features selected by multiple regression
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 

""""""""""""""""""""""""""""""""""""""""" 
3.1 Process data
"""""""""""""""""""""""""""""""""""""""""  
from sklearn.preprocessing import StandardScaler

# prepare data for buildong model
#y = np.array(y1).reshape(-1,1)

modeldata = t1_sp_nq_g.copy()  

# trun the categorical data in X to numerical data
obj_col =(['StatusToSP'+ '_' + stock_to_study[No],
          'StatusToNasdaq'+ '_' + stock_to_study[No],
          'StatusToSN'+ '_' + stock_to_study[No],
          'Up/Down'+ '_' + stock_to_study[No],
          'Up/Down_gold'])
    
for obj in obj_col:
    modeldata[obj+'_num'] = pd.Categorical(modeldata[obj]).codes
    modeldata[obj+'_num'] = modeldata[obj+'_num'].replace({0:1, 1:0})
           
modeldata.fillna(-9999,inplace=True)
forecast_label = 'Adj. Close'+'_'+stock_to_study[No]

modeldata.sort_index(axis=0, ascending=True,inplace=True)
modeldata.dropna(inplace=True)

#select feature set for predicting y1(Ajst Close)
key_set = list()
for i in range(0,(len(feature1_index))):  
   if feature1_index[i] !=0:
       key = var_set[featureset1_index][feature1_index[i]-1]
       print(key)
       key_set.append(key)
       
# assgin label and features
X = np.array(modeldata[key_set])
y = modeldata[forecast_label]

#normalize data
#X = preprocessing.scale(X) 
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
scaler.fit(y)
y = scaler.transform(y)

# split the data into train and test set
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

""""""""""""""""""""""""""""""""""""""""" 
3.2 fit the model
"""""""""""""""""""""""""""""""""""""""""
svr_lin = svm.SVR(kernel='linear',C=1e3)
svr_pol = svm.SVR(kernel='poly',C=1e3,degree=2)
svr_rbf = svm.SVR(kernel='rbf',gamma=0.1)

svr_lin.fit(X_train,y_train)
#svr_pol.fit(X_train,y_train)
#svr_rbf.fit(X_train,y_train)

#evaluate accuracy
accuracy_svrlin = svr_lin.score(X_test,y_test)
#accuracy_svrpol = svr_pol.score(X_test,y_test)
#accuracy_svrrbf = svr_rbf.score(X_test,y_test)
print('Accuracy',accuracy_svrlin)

#train_pred = svr_lin.predict(X_train)
test_pred = svr_lin.predict(X_test)

# scale back
test_pred = scaler.inverse_transform(test_pred)
y_test = scaler.inverse_transform(y_test)

prd_pec_dif = (test_pred - y_test)/y_test
xaxix_val = np.arange(1, len(y_test)+1)

#plot the percentage of difference
label = 'pecentage difference'
title_dif = str("Difference Between Predicted and Real Price: test set of "+stock_to_study[No])

fig = plt.figure
fig, ax = plt.subplots(figsize=(10,6),facecolor='w')
fig.subplots_adjust(bottom=0.2)
ax.plot(xaxix_val, prd_pec_dif, color='#0094ff',label=label)
plt.legend(loc=0,prop={'size':9},fancybox=True)
plt.ylabel('Pecentage Difference')
plt.title(title_dif)
plt.grid(linestyle='dotted')
plt.show()

#in plotly
title_dif = str("Difference Between Predicted and Real Price: test set of "+stock_to_study[No])
trace0 = plotly.graph_objs.Scatter(
    x = xaxix_val,
    y = prd_pec_dif,
    
    line = dict(
        color = ('#0000ff'),
        width = 4)
)
plotly.offline.plot({
"data": [
    plotly.graph_objs.Scatter(x=xaxix_val,
    y=prd_pec_dif)],
"layout": plotly.graph_objs.Layout(title=title_dif),
})

""""""""""""""""""""""""""""""""""""""""" 
3.3 Look back
"""""""""""""""""""""""""""""""""""""""""
modeldata.sort_index(axis=0, ascending=True,inplace=True)
lookback_period = len(y_test)
modeldata['lookback'] = modeldata[forecast_label].shift(lookback_period)
modeldata['lookback'] = modeldata['lookback'].shift(-lookback_period)

X_lookback = X[-lookback_period:]
lookback_set = svr_lin.predict(X_lookback)
# scale back
lookback_set = scaler.inverse_transform(lookback_set)
 
modeldata['lookback_pre']=np.nan
modeldata['lookback_pre'][-lookback_period:] = lookback_set

lookback_pec_dif =((modeldata[forecast_label][-lookback_period:] - 
                    modeldata['lookback_pre'][-lookback_period:])/
                    modeldata[forecast_label][-lookback_period:])

#plot the percentage of difference
label1 = 'lookback_original'
label2 = 'lookback_predicted'
label3 = 'pecentage difference'
title_lookback = str('Compare the predicted and real stock price in the past date: Look Back of '+stock_to_study[No])

fig = plt.figure
fig, ax = plt.subplots(figsize=(10,6),facecolor='w')
fig.subplots_adjust(bottom=0.2)
ax.plot(modeldata.index[-lookback_period:],modeldata[forecast_label][-lookback_period:] ,color='#0000ff',label=label1)
ax.plot(modeldata.index[-lookback_period:],modeldata['lookback_pre'][-lookback_period:] ,color='#ff1493',label=label2)
ax.plot(modeldata.index[-lookback_period:],lookback_pec_dif,color='#0094ff',label=label3)
plt.legend(loc=0,prop={'size':9},fancybox=True)
plt.ylabel('Pecentage Difference')
plt.title(title_lookback)
plt.grid(linestyle='dotted')
plt.show()

#in plotly
title_lookback = str('Compare the predicted and real stock price in the past date: Look Back of '+stock_to_study[No])
trace0 = plotly.graph_objs.Scatter(
    x = modeldata.index[-lookback_period:],
    y = modeldata[forecast_label][-lookback_period:],
    name = 'lookback_original',
    line = dict(
        color = ('#0000ff'),
        width = 4)
)
trace1 = plotly.graph_objs.Scatter(
    x = modeldata.index[-lookback_period:],
    y = modeldata['lookback_pre'][-lookback_period:],
    name = 'lookback_predicted',
    line = dict(
        color = ('rgb(145.0, 191.0, 219.0)'),
        width = 4)
)
trace2 = plotly.graph_objs.Scatter(
    x = modeldata.index[-lookback_period:],
    y = lookback_pec_dif,
    name = 'pecentage difference',
    line = dict(
        color = ('rgb(252.0, 141.0, 89.0)'),
        width = 4)
)  

data = [trace0, trace1, trace2]

layout = dict(title = title_lookback,
              yaxis = dict(title = 'Stock Price '),
              )

fig = dict(data=data, layout=layout)
plotly.offline.plot(fig, filename='stoc_lookback1')
 

""""""""""""""""""""""""""""""""""""""""" 
3.4 Predict the future price
"""""""""""""""""""""""""""""""""""""""""    

forecast_period = int(math.ceil(0.05*len(modeldata)))
modeldata['label'] = modeldata[forecast_label]

#X = X[:-forecast_period]
X_future = X[-forecast_period:]
forecast_set = svr_lin.predict(X_future)
# scale back
forecast_set = scaler.inverse_transform(forecast_set)

modeldata['Predicted price'] = np.nan
           
last_date = modeldata.iloc[-1].name
last_timestamp = last_date.timestamp()
one_day = 86400
next_timestamp = last_timestamp + one_day

for i in forecast_set:
    next_date = datetime.fromtimestamp(next_timestamp)
    next_timestamp += one_day
    modeldata.loc[next_date] = [np.nan for _ in range(len(modeldata.columns)-1)]+[i]

#plot
label1 = 'Real Price'
label2 = 'Predict Price'
title_future = str('Predict the stock price in the future: '+stock_to_study[No])            

fig = plt.figure
fig, ax = plt.subplots(figsize=(10,6),facecolor='w')
fig.subplots_adjust(bottom=0.2)
ax.plot(modeldata.index,modeldata['label'],color='#0000ff',label=label1)
ax.plot(modeldata.index,modeldata['Predicted price'], color='#ff1493', label=label2)
#dates on the x-axis
ax.xaxis_date()
ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(),rotation=60)
plt.title(title_future)
plt.legend(loc=0,prop={'size':9},fancybox=True)
plt.ylabel('Stock Price')
plt.grid(linestyle='dotted')
plt.show()   

#in plotly
title_future = str('Predict the stock price in the future: '+stock_to_study[No])            
trace0 = plotly.graph_objs.Scatter(
    x = modeldata.index,
    y = modeldata['label'],
    name = 'original',
    line = dict(
        color = ('#0000ff'),
        width = 4)
)
trace1 = plotly.graph_objs.Scatter(
    x = modeldata.index,
    y = modeldata['Predicted price'],
    name = 'predicted',
    line = dict(
        color = ('rgb(221,28,119)'),
        width = 4)
)
    
data = [trace0, trace1]

layout = dict(title = title_future,
              yaxis = dict(title = 'Stock Price '),
              )

fig = dict(data=data, layout=layout)
plotly.offline.plot(fig, filename='stock_future1')
  
    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
4. Build the Model using the features selected by sklean feature selection
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 

""""""""""""""""""""""""""""""""""""""""" 
4.1 Process data
""""""""""""""""""""""""""""""""""""""""" 
modeldata = t1_sp_nq_g.copy()  

# trun the categorical data in X to numerical data
obj_col =(['StatusToSP'+ '_' + stock_to_study[No],
          'StatusToNasdaq'+ '_' + stock_to_study[No],
          'StatusToSN'+ '_' + stock_to_study[No],
          'Up/Down'+ '_' + stock_to_study[No],
          'Up/Down_gold'])
    
for obj in obj_col:
    modeldata[obj+'_num'] = pd.Categorical(modeldata[obj]).codes
    modeldata[obj+'_num'] = modeldata[obj+'_num'].replace({0:1, 1:0})
           
modeldata.fillna(-9999,inplace=True)
forecast_label = 'Adj. Close'+'_'+stock_to_study[No]

modeldata.sort_index(axis=0, ascending=True,inplace=True)
modeldata.dropna(inplace=True)

X = np.array(modeldata[var_set1])
y = np.array(modeldata[forecast_label])

""""""""""""""""""""""""""""""""""""""""" 
4.2 Rank and select the features
"""""""""""""""""""""""""""""""""""""""""
#Univariate feature selection
X_new = SelectKBest(f_regression, k = 4).fit_transform(X, y)

#X = preprocessing.scale(X) 
scaler = StandardScaler()
scaler.fit(X_new)
X_new = scaler.transform(X_new)
scaler.fit(y)
y = scaler.transform(y)

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X_new,y,test_size=0.2)

""""""""""""""""""""""""""""""""""""""""" 
4.3 fit the model
"""""""""""""""""""""""""""""""""""""""""

svr_lin = svm.SVR(kernel='linear',C=1e3)

svr_lin.fit(X_train,y_train)

#evaluate accuracy
accuracy_svrlin = svr_lin.score(X_test,y_test)
#accuracy_svrpol = svr_pol.score(X_test,y_test)
#accuracy_svrrbf = svr_rbf.score(X_test,y_test)
print('Accuracy',accuracy_svrlin)

""""""""""""""""""""""""""""""""""""""""" 
4.4 Look back
"""""""""""""""""""""""""""""""""""""""""
modeldata_new = modeldata.copy()
modeldata_new.sort_index(axis=0, ascending=True,inplace=True)
lookback_period = len(y_test)
modeldata_new['lookback'] = modeldata_new[forecast_label].shift(lookback_period)
modeldata_new['lookback'] = modeldata_new['lookback'].shift(-lookback_period)

X_lookback = X_new[-lookback_period:]
lookback_set = svr_lin.predict(X_lookback)
# scale back
lookback_set = scaler.inverse_transform(lookback_set)
            
modeldata_new['lookback_pre']=np.nan
modeldata_new['lookback_pre'][-lookback_period:] = lookback_set

lookback_pec_dif =((modeldata_new[forecast_label][-lookback_period:] - 
                    modeldata_new['lookback_pre'][-lookback_period:])/
                    modeldata_new[forecast_label][-lookback_period:])

#plot the percentage of difference
label1 = 'lookback_original'
label2 = 'lookback_predicted'
label3 = 'pecentage difference'
title_lookback = str('Compare the predicted and real stock price in the past date: Look Back of '+stock_to_study[No])

fig = plt.figure
fig, ax = plt.subplots(figsize=(10,6),facecolor='w')
fig.subplots_adjust(bottom=0.2)
ax.plot(modeldata_new.index[-lookback_period:],modeldata_new[forecast_label][-lookback_period:] ,color='#0000ff',label=label1)
ax.plot(modeldata_new.index[-lookback_period:],modeldata_new['lookback_pre'][-lookback_period:] ,color='#ff1493',label=label2)
ax.plot(modeldata_new.index[-lookback_period:],lookback_pec_dif,color='#0094ff',label=label3)
plt.legend(loc=0,prop={'size':9},fancybox=True)
plt.ylabel('Pecentage Difference')
plt.title(title_lookback )
plt.grid(linestyle='dotted')
plt.show()

#in plotly
title_lookback = str('Compare the predicted and real stock price in the past date: Look Back of '+stock_to_study[No])
trace0 = plotly.graph_objs.Scatter(
    x = modeldata_new.index[-lookback_period:],
    y = modeldata_new[forecast_label][-lookback_period:],
    name = 'lookback_original',
    line = dict(
        color = ('#0000ff'),
        width = 4)
)
trace1 = plotly.graph_objs.Scatter(
    x = modeldata_new.index[-lookback_period:],
    y = modeldata_new['lookback_pre'][-lookback_period:],
    name = 'lookback_predicted',
    line = dict(
        color = ('rgb(145.0, 191.0, 219.0)'),
        width = 4)
)
trace2 = plotly.graph_objs.Scatter(
    x = modeldata_new.index[-lookback_period:],
    y = lookback_pec_dif,
    name = 'pecentage difference',
    line = dict(
        color = ('rgb(252.0, 141.0, 89.0)'),
        width = 4)
)  

data = [trace0, trace1, trace2]

layout = dict(title = title_lookback,
              yaxis = dict(title = 'Stock Price '),
              )

fig = dict(data=data, layout=layout)
plotly.offline.plot(fig, filename='stock_lookback2')
    
'''''''''''''''''''''''''''''
4.5 Predict the future price
''''''''''''''''''''''''''''' 

forecast_period = int(math.ceil(0.05*len(modeldata_new)))
modeldata_new['label'] = modeldata_new[forecast_label]

#X = X_new[:-forecast_period]
X_future = X_new[-forecast_period:]
forecast_set = svr_lin.predict(X_future)
# scale back
forecast_set = scaler.inverse_transform(forecast_set)

modeldata_new['Predicted price'] = np.nan
           
last_date = modeldata_new.iloc[-1].name
last_timestamp = last_date.timestamp()
one_day = 86400
next_timestamp = last_timestamp + one_day

for i in forecast_set:
    next_date = datetime.fromtimestamp(next_timestamp)
    next_timestamp += one_day
    modeldata_new.loc[next_date] = [np.nan for _ in range(len(modeldata_new.columns)-1)]+[i]

#plot
label1 = 'Real Price'
label2 = 'Predict Price'
title_future = str('Predict the stock price in the future: '+stock_to_study[No])            
            
fig = plt.figure
fig, ax = plt.subplots(figsize=(10,6),facecolor='w')
fig.subplots_adjust(bottom=0.2)
ax.plot(modeldata_new.index,modeldata_new['label'],color='#0000ff',label=label1)
ax.plot(modeldata_new.index,modeldata_new['Predicted price'], color='#ff1493', label=label2)
#dates on the x-axis
ax.xaxis_date()
ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(),rotation=60)

plt.legend(loc=0,prop={'size':9},fancybox=True)
plt.title(title_future)
plt.ylabel('Stock Price')
plt.grid(linestyle='dotted')
plt.show()    
    
#in plotly
trace0 = plotly.graph_objs.Scatter(
    x = modeldata_new.index,
    y = modeldata_new['label'],
    name = 'original',
    line = dict(
        color = ('#0000ff'),
        width = 4)
)
trace1 = plotly.graph_objs.Scatter(
    x = modeldata_new.index,
    y = modeldata_new['Predicted price'],
    name = 'predicted',
    line = dict(
        color = ('rgb(221,28,119)'),
        width = 4)
)
    
data = [trace0, trace1]

title_future = str('Predict the stock price in the future: '+stock_to_study[No]) 
      
layout = dict(title = title_future,
              yaxis = dict(title = 'Stock Price '),
              )

fig = dict(data=data, layout=layout)
plotly.offline.plot(fig, filename='stock_future2')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Conclusion: 
    The adjusted close price is the target we want to predict in this sub-aim. The 
historical data of the S&P 500 index, NASDAQ 100 index, crude oil, gold, and 
stock were used. The correlation between the stock close price and the volume, 
indices value, daily return, the performance relative to the benchmarks were 
investigated and the variables with significant correlations were selected as  
features to predict the target.
    Two pacakages, statsmodels and sklearn, were used to select the features. By 
using statsmodels to do multiple variable regression analysis, we knew what variables 
are to be selected. In case of AAPL, 7 indicators were selected: 'Adj. Volume_AAPL',
'Daily_Return_AAPL','StatusToNasdaq_AAPL_num', 'Last_sp', 'Index Value_nsdq',
'Total Market Value_nsdq', 'Value_gold'. The key parameters were 
set as 4 for sklearn's feature_selection in this case. In contrast, the results
did not explain which variables had been selected. 
    The support vector machine(SVM) wasused to build the model and predict the 
stock price. In the case of AAPL, the accuracy is 10% higher by using the 4 
features selected by sklearn's feature_selection than that resulted from using
 the 7 features selected by statsmodels' ols. 
    Overall, the trends of the SVM model predicted stock price by using either
statsmodels or sklearn to select the features are similar.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""