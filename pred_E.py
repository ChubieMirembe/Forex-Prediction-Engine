# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
# import plotly.express as px

# Importing the dataset and dropping unnecesary columns
dataset = pd.read_csv('AUDUSD60.csv', delimiter='\t', names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], 
                      parse_dates=['Date'], 
                      )
dataset = dataset.drop(['Open', 'High', 'Low', 'Volume'], axis=1)
dataset = dataset.drop(dataset.index[19969:19999])
dataset = dataset.drop(dataset.index[0:47])
dataset.reset_index(drop=True, inplace=True)
dataset = dataset.set_index('Date')

#For loop to create the weekkly splits
s = pd.Series(dataset['Close'], index=pd.date_range('2017-09-20', '2020-12-06', freq='W'))
e = pd.Series(dataset['Close'], index=pd.date_range('2017-09-24', '2020-12-12', freq='W-SAT'))
group = []

for d in range(0, len(s)):
    start = s.index[d]
    end = e.index[d]
    data = (dataset[start:end])
    data.reset_index(level=0, inplace=True)
    if len(data) < 120:
        continue
    else:
        group.append(data)
       
'''        
#For loop to add the linear regression line
for l in range(0,len(group)):
   x = np.arange(group[l].index.size)
   fit = np.polyfit(x, group[l]['Close'], deg=1)
   fit_function = np.poly1d(fit)
   plt.plot(group[l].index, fit_function(x))
   plt.plot(group[l].index, group[l]['Close'])
   plt.title('Group {}'.format(l))
   plt.show()
'''     

#Creating the histogram format
regressor = LinearRegression()
for g in range(0,len(group)):
    X = np.reshape(group[g].index, (120,-1))        # Will have problems on the last one
    regressor.fit(X, group[g]['Close'])
    yi = regressor.predict(X)
    group[g]['Distance'] = group[g]['Close'].values - yi
    plt.bar(group[g].index, group[g]['Distance'])
    plt.title('Group {}'.format(g))
    plt.show()

   
#Segmentation Algorithm 
pattern = []
minArea = 5
for i in range(0, len(group)):
    checkA = 0
    checkB = 0
    halfA = []
    halfB = []
    positiveC = 0
    negativeC = 0
    for di in range(0, len(group[i])):
            if group[i]['Distance'][di] < 0:  
                if checkA == 0:
                    if len(halfB) < minArea:
                        halfA.clear()
                        halfB.clear()
                        halfA.append(group[i].iloc[di])
                        checkA += 1
                        checkB = 0
                    else:
                        if len(halfA) < minArea:
                            halfA.clear()
                            # halfB.clear()
                            halfA.append(group[i].iloc[di])
                            checkA += 1
                            checkB = 0
                        else:
                            full = halfA + halfB
                            pattern.append(full)
                            halfA.clear()
                            halfB.clear()
                            # full.clear()
                            halfA.append(group[i].iloc[di])
                            checkA += 1 
                            checkB = 0
                else:
                    halfA.append(group[i].iloc[di])
                    checkB = 0
            
            elif group[i]['Distance'][di] > 0:
                if checkB == 0:
                    if len(halfA) < minArea:
                        halfA.clear()
                        halfB.clear()
                        halfB.append(group[i].iloc[di])
                        checkB += 1
                        checkA = 0
                    else:
                        if len(halfB) < minArea:
                            # halfA.clear()
                            halfB.clear()
                            halfB.append(group[i].iloc[di])
                            checkB += 1
                            checkA = 0
                        else:
                            full = halfB + halfA
                            pattern.append(full)
                            halfA.clear()
                            halfB.clear()
                            # full.clear()
                            halfB.append(group[i].iloc[di])
                            checkB += 1
                            checkA = 0
                else:
                    halfB.append(group[i].iloc[di])
                    checkA = 0
                

#Viewing the sgemented pattern 








ps = pd.DataFrame(pattern[0], columns = ['Date', 'Close', 'Distance'])
plt.bar(ps.index, ps['Distance'])
plt.show()  

full = halfA + halfB

ps = pd.DataFrame(pattern[0], columns = ['Date', 'Close', 'Distance'])
plt.bar(ps.index, ps['Distance'])
plt.show()  

,










#Feature Creation Algorithm


#Segmentation Algorithm 


#K-Means Algorithm                    


#ANN


#DTW                

