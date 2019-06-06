#基础库： pandas,numpy,scipy,matplotlib,statsmodels ：

from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot



#运行的时候会报错：Undefined variable from import:，不过好像不影响结果



#1.数据：

dta=[10930, 10318, 10595, 10972, 7706, 6756, 9092, 10551, 9722, 10913, 11151, 8186, 6422,
6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
10477,10148,10395,11261,8713,7299,10424,10795, 11069, 11602,11427,9095,7707,10767,
12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722,
11999,9390,13481,14795,15845,15271,14686,11054,10395]


#print(np.size(dta))
dta=np.array(dta,dtype=np.float) #这里要转下数据类型，不然运行会报错
dta=pd.Series(dta)
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001','2090')) #应该是2090，不是2100
dta.plot(figsize=(12,8))
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

fig = sm.graphics.tsa.plot_acf(dta,lags=30,ax=ax1)
fig = sm.graphics.tsa.plot_pacf(dta,lags=30,ax=ax2)
print(sm.tsa.stattools.adfuller(dta))

fig = plt.figure(figsize=(12,8))
ax1= fig.add_subplot(111)
diff1 = dta.diff(1)
diff1.plot(ax=ax1)


diff1= dta.diff(1)#我们已经知道要使用一阶差分的时间序列，之前判断差分的程序可以注释掉 //原文有错误应该是diff1= dta.diff(1)，而非dta= dta.diff(1)

diff1=diff1.dropna()
print(diff1)
print(sm.tsa.stattools.adfuller(diff1))   #adf检验序列平稳性
fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(diff1,lags=30,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(diff1,lags=30,ax=ax2)

plt.show() #在Scala IDE要输入这个命令才能显示图片

arma_mod70 = sm.tsa.ARMA(diff1,(7,0)).fit()
print(arma_mod70.aic,arma_mod70.bic,arma_mod70.hqic)
arma_mod30 = sm.tsa.ARMA(diff1,(0,1)).fit()
print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
arma_mod71 = sm.tsa.ARMA(diff1,(7,1)).fit()
print(arma_mod71.aic,arma_mod71.bic,arma_mod71.hqic)
arma_mod80 = sm.tsa.ARMA(diff1,(8,0)).fit()
print(arma_mod80.aic,arma_mod80.bic,arma_mod80.hqic)


resid = arma_mod80.resid #原文把这个变量赋值语句漏了，所以老是出错
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
plt.show()


print(sm.stats.durbin_watson(arma_mod80.resid.values))  #dw检验相关性

print(stats.normaltest(resid))      #正态性
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)
plt.show()


r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

predict_dta = arma_mod80.predict('2091', '2100', dynamic=True)
print(predict_dta)
predict_dta_restored = pd.Series([dta[89]], index=[dta.index[89]]) .append(predict_dta).cumsum()
print(predict_dta_restored)
fig, ax = plt.subplots(figsize=(12, 8))
ax = diff1.ix['2002':].plot(ax=ax)
fig = arma_mod80.plot_predict('2091', '2100', dynamic=True, ax=ax, plot_insample=False)
plt.show()

#fig, ax = plt.subplots(figsize=(12, 8))
#ax = dta.ix['2000':].plot(ax=ax)
#predict_dta_restored.index = pd.Index(sm.tsa.datetools.dates_from_range('2090','2100')) #应该是2090，不是2100
#fig = arma_mod80.plot_predict('2090', '2100', dynamic=True, ax=ax, plot_insample=False)


#predict = dta + predict_dta_restored
#print(predict)
#del predict[90]
#predict
#predict = pd.Series(predict)
#predict.index = pd.Index(sm.tsa.datetools.dates_from_range('2001','2100')) #应该是2090，不是2100
#predict.plot(figsize=(12,8))
