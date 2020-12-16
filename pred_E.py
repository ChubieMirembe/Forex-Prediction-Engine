# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Importing the dataset and dropping unnecesary columns
dataset = pd.read_csv('AUDUSD60.csv', delimiter='\t', names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], 
                      parse_dates=['Date'], 
                      index_col='Date')
#index_col='Date'
dataset = dataset.drop(['Open', 'High', 'Low', 'Volume'], axis=1)


#For loop to create the weekkly splits
s = pd.Series(dataset['Close'], index=pd.date_range('2017-09-20', '2020-12-06', freq='W'))
e = pd.Series(dataset['Close'], index=pd.date_range('2017-09-24', '2020-12-12', freq='W-SAT'))

group = []
for d in range(0, len(s)):
    start = s.index[d]
    end = e.index[d]
    data = (dataset[start:end])
    group.append(data)
        
#For loop to add the linear regression line
for l in range(1,len(group)):
   x = np.arange(group[l].index.size)
   fit = np.polyfit(x, group[l]['Close'], deg=1)
   fit_function = np.poly1d(fit)
   plt.plot(group[l].index, fit_function(x))
   plt.plot(group[l].index, group[l]['Close'])
   plt.show()
   
   dataset.plot.bar(x=(fit[1]), y = dataset['Close'])    
'''
I want to use the regression line as the straigh line and plot a bar graph of
the price movement around it
'''   
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X = np.reshape(dataset.index, (20000,-1))
regressor.fit(X, dataset['Close'])

print(regressor.coef_)


   
#Segmentation Algorithm 
pattern = []
min_area = 5
negative_c = 0
positive_c = 0
for a in range(0,len(group)):
    group = group[a]
    for b in range(0, len(group[a])):
        if group[a][b] < 0:
            pattern.append(group[a][b])
            negative_c += 1
            continue
        elif group[a][b] > 0:
            pattern.append(group[a][b])
            positive_c += 1
            continue
        else:
            if negative_c < min_area:
                pattern.append(group[a][b])
                negative_c = 0
                positive_c += 1
            elif negative_c >= min_area:
                pattern.append(group[a][b])
                negative_c = 0
                positive_c += 1
            elif positive_c < min_area:
                pattern.append(group[a][b])
                positive_c = 0
                negative_c += 1
            elif positive_c >= min_area:
                pattern.append(group[a][b])
                positive_c = 0
                negative_c += 1
                
#Feature Creation Algorithm


#Segmentation Algorithm 


#K-Means Algorithm                    


#ANN


#DTW                





fig = px.bar(dataset, x=dataset.index, y=dataset['Close'], color=('Red'))
fig.show()

dataset['return'] = dataset['Close'] - dataset['Close'].shift(1)
return_range = dataset['return'].max() - dataset['return'].min()
dataset['return'] = dataset['return'] / return_range

dataset.plot(x=dataset.index, y='return')





df = px.data.stocks(indexed=True)-1
fig = px.bar(df, x=df.index, y="GOOG")
fig.show()

from sklearn.preprocessing import maxabs_scale
sc = maxabs_scale(dataset['Close'])
dataset['Close'] = sc.transform(dataset['Close'])




'''
Segementation Algorithm!!
    1. Don't need to seperate
        a. The K-Means algorithm will split 
            i. You might want to check if the the optimum number of clusters is 2
    2. Store the trend patterns in a normalised vector format
        a. Should I do that in the section where I split them, so it just carries it over
        b. Values between -1 and +1
        
        
The patterns is normalised after being segmented

Make a list for each one, th
'''
