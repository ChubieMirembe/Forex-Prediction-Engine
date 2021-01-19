# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Importing the dataset and dropping unnecesary columns
dataset = pd.read_csv('AUDUSD60.csv', delimiter='\t', 
                      names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], 
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
    if len(data) < 120: continue
    else: group.append(data)
       
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
scaler = MinMaxScaler(feature_range = (-1, 1))

regressor = LinearRegression()
for g in range(0,len(group)):
    X = np.reshape(group[g].index, (120,-1))        
    regressor.fit(X, group[g]['Close'])
    yi = regressor.predict(X)
    group[g]['Distance'] = group[g]['Close'].values - yi   
    res = group[g]['Distance'].values.reshape(-1,1)
    group[g]['Distance'] = scaler.fit_transform(res)    
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
for pat in range(0,len(pattern)):
    ps = pd.DataFrame(pattern[pat], columns = ['Date', 'Close', 'Distance'])
    ps.reset_index(level=0, inplace=True)
    plt.bar(ps.index, ps['Distance'])
    plt.title('Pattern: {}'.format(pat))
    plt.show()     

'''
count = 0
for p in range(0, len(pattern)):
    ps = pd.DataFrame(pattern[p], columns = ['Date', 'Close', 'Distance'])
    ps.reset_index(level=0, inplace=True)
    total = ps['Distance'].abs().sum()
    if total < 5:
        count += 1
        plt.bar(ps.index, ps['Distance'])
        plt.title(count)
        plt.show()
        print(total)
    else: 
        continue
'''        


#Feature Creation Algorithm
columns = ['F1', 'F2', 'A1', 'F3', 'F4', 'A2', 'T']
features = pd.DataFrame(index = range(0, len(pattern)), columns=columns)

for p in range(0, len(pattern)):
    ps = pd.DataFrame(pattern[p], columns = ['Date', 'Close', 'Distance'])
    pH = []
    nH = []
    
    for d in range(0, len(ps)):
        if ps.iloc[d]['Distance'] > 0: pH.append(ps.iloc[d]['Distance'])
        elif ps.iloc[d]['Distance'] < 0: nH.append(ps.iloc[d]['Distance'])
    
    if ps.iloc[0]['Distance'] > 0:
        features.iloc[p]['T'] = 0
        highest_point = max(pH)
        lowest_point = min(nH)
        
        if highest_point > pH[0]: features.iloc[p]['F1'] = 1
        elif highest_point == pH[0]: features.iloc[p]['F1'] = 2
        
        if highest_point > pH[-1]: features.iloc[p]['F2'] = 3
        elif highest_point == pH[-1]: features.iloc[p]['F2'] = 2
        
        if nH[0] > lowest_point: features.iloc[p]['F3'] = 3
        elif nH[0] == lowest_point: features.iloc[p]['F3'] = 2
            
        if nH[-1] > lowest_point: features.iloc[p]['F4'] = 1
        elif nH[-1] == lowest_point: features.iloc[p]['F4'] = 2
            
        features.iloc[p]['A1'] = round((round(abs(sum(pH)), 1) / ((round(abs(sum(pH)), 1)) + (round(abs(sum(nH)), 1)))) * 10, 1)
        # round(abs(sum(pH)), 1)
        # round((round(abs(sum(pH)), 1) / ((round(abs(sum(pH)), 1)) + (round(abs(sum(nH)), 1)))) * 10, 1)
        features.iloc[p]['A2'] = round((round(abs(sum(nH)), 1) / ((round(abs(sum(pH)), 1)) + (round(abs(sum(nH)), 1)))) * 10, 1)
        # round(abs(sum(nH)), 1)
        # round((round(abs(sum(nH)), 1) / ((round(abs(sum(pH)), 1)) + (round(abs(sum(nH)), 1)))) * 10, 1)
    
    elif ps.iloc[0]['Distance'] < 0:
        features.iloc[p]['T'] = 1
        highest_point = max(pH)
        lowest_point = min(nH)
        
        if nH[0] == lowest_point: features.iloc[p]['F1'] = 2 
        elif nH[0] > lowest_point: features.iloc[p]['F1'] = 3
        
        if nH[-1] > lowest_point: features.iloc[p]['F2'] = 1 
        elif nH[-1] == lowest_point: features.iloc[p]['F2'] = 2
        
        if highest_point > pH[0]: features.iloc[p]['F3'] = 1
        elif highest_point == pH[0]: features.iloc[p]['F3'] = 2 
        
        if highest_point > pH[-1]: features.iloc[p]['F4'] = 3
        elif highest_point == pH[-1]: features.iloc[p]['F4'] = 2

        features.iloc[p]['A1'] = round((round(abs(sum(nH)), 1) / ((round(abs(sum(pH)), 1)) + (round(abs(sum(nH)), 1)))) * 10, 1)
        # round((round(abs(sum(nH)), 1) / ((round(abs(sum(pH)), 1)) + (round(abs(sum(nH)), 1)))) * 10, 1)
        # round(abs(sum(nH)), 1)
        features.iloc[p]['A2'] = round((round(abs(sum(pH)), 1) / ((round(abs(sum(pH)), 1)) + (round(abs(sum(nH)), 1)))) * 10, 1)
        # round((round(abs(sum(pH)), 1) / ((round(abs(sum(pH)), 1)) + (round(abs(sum(nH)), 1)))) * 10, 1)
        # round(abs(sum(pH)), 1)

'''        
#Removing the row withe the values less than 5 
del_row = []
for row in range(len(features)):
    summing = features.iloc[row]['A1'] + features.iloc[row]['A2']
    if summing < 5: del_row.append(row)
    else: continue
r_features = features.drop(del_row)
r_features.reset_index(level = 0, drop = True, inplace=True)

for row in range(len(r_features)):
    r_features.iloc[row]['A1'] = r_features.iloc[row]['A1'] / (r_features.iloc[row]['A1']) + (r_features.iloc[row]['A2']) 
    r_features.iloc[row]['A2'] = 10 - r_features.iloc[row]['A1']
    
    if features.iloc[row]['T'] == 0:
        r_features.iloc[row]['A1'] = r_features.iloc[row]['A1'] / (r_features.iloc[row]['A1']) + (r_features.iloc[row]['A2']) * 10
        r_features.iloc[row]['A2'] = r_features.iloc[row]['A2'] / (r_features.iloc[row]['A1']) + (r_features.iloc[row]['A2']) * 10
    else: 
        r_features.iloc[row]['A1'] = round((round(abs(sum(nH)), 1) / ((round(abs(sum(pH)), 1)) + (round(abs(sum(nH)), 1)))) * 10, 1)
        r_features.iloc[row]['A2'] = round((round(abs(sum(pH)), 1) / ((round(abs(sum(pH)), 1)) + (round(abs(sum(nH)), 1)))) * 10, 1)
'''

#ANN
x = features.iloc[:, :-1].values
y = features.iloc[:, -1].values

x = np.asarray(x).astype('float32')
y = np.asarray(y).astype('float32')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(x_train, y_train, batch_size = 32, epochs = 100)

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

#DTW                










