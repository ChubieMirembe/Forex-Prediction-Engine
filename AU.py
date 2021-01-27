# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Importing the dataset and dropping unnecesary columns
dataset = pd.read_csv('AUDUSD60(full).csv', delimiter='\t', 
                      names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ATR'],  
                      )
# parse_dates=['Date']


def TR(d,c,h,l,o,yc):
    x = h-l
    y = abs(h-yc)
    z = abs(l-yc)

    if y <= x >= z:
        TR = x
    elif x <= y >= z:
        TR = y
    elif x <= z >= y:
        TR = z
        
    # print (d, TR)
    return d, TR

x = 1
TRDates = []
TrueRanges = []

while x < len(dataset):
    TRDate, TrueRange = TR(dataset['Date'][x], dataset['Close'][x], dataset['High'][x],
                           dataset['Low'][x], dataset['Open'][x], dataset['Close'][x-1])
    
    TRDates.append(TRDate)
    TrueRanges.append(TrueRange)
    
    # print(TrueRange)
    
    x+=1


def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a

fATR = ExpMovingAverage(TrueRanges, 14)

for f in range(1, len(fATR)):
    dataset['ATR'][f] = round(fATR[f] * 10000, 2)
    
    
dataset = dataset.drop(['Open', 'High', 'Low', 'Volume'], axis=1)

dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.drop(dataset.index[86531::])
dataset = dataset.drop(dataset.index[0:118])
dataset.reset_index(drop=True, inplace=True)
dataset = dataset.set_index('Date')

s = pd.Series(dataset['Close'], index=pd.date_range('2007-01-04', '2021-01-03', freq='W'))
e = pd.Series(dataset['Close'], index=pd.date_range('2007-01-07', '2021-01-09', freq='W-SAT'))

group = []
for d in range(0, len(s)):
    start = s.index[d]
    end = e.index[d]
    data = (dataset[start:end])
    data.reset_index(level=0, inplace=True)
    if len(data) < 120: continue
    else: group.append(data)
    
dataset.reset_index(level=0, inplace=True)
    
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
    # plt.bar(group[g].index, group[g]['Distance'])
    # plt.title('Group {}'.format(g))
    # plt.show()

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
        






## Test pattern 
'''
dataset_test = pd.read_csv('AUDUSD60test.csv', delimiter='\t', 
                      names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
# parse_dates=['Date']
dataset_test = dataset_test.drop(['Open', 'High', 'Low', 'Volume'], axis=1)
dataset_test = dataset_test.drop(dataset_test.index[0:880])
dataset_test['Date'] = pd.to_datetime(dataset_test['Date'])
dataset_test.reset_index(drop=True, inplace=True)
# dataset_test = dataset_test.set_index('Date')
'''

dataset_test = pd.read_csv('AUDUSD60test(new).csv', skiprows=1,
                      names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
dataset_test = dataset_test.drop(['Open', 'High', 'Low', 'Volume'], axis=1)
# dataset_test = dataset_test.drop(dataset_test.index[0:41])
dataset_test = dataset_test.tail(120)
dataset_test['Date'] = pd.to_datetime(dataset_test['Date'])
dataset_test.reset_index(drop=True, inplace=True)

scaler = MinMaxScaler(feature_range = (-1, 1))
regressor_test = LinearRegression()
    
X = np.reshape(dataset_test.index, (120,-1))        
regressor_test.fit(X, dataset_test['Close'])
yi = regressor_test.predict(X)
dataset_test['Distance'] = dataset_test['Close'].values - yi   
res = dataset_test['Distance'].values.reshape(-1,1)
dataset_test['Distance'] = scaler.fit_transform(res)    
plt.bar(dataset_test.index, dataset_test['Distance'])
plt.show()

checkA = 0
checkB = 0
halfA = []
halfB = []
pattern_test = []
for di in range(len(dataset_test) - 1, 0, -1):
            if dataset_test['Distance'][di] < 0:  
                if checkA == 0:
                    if len(halfB) < minArea:
                        halfA.clear()
                        halfB.clear()
                        halfA.append(dataset_test.iloc[di])
                        checkA += 1
                        checkB = 0
                    else:
                        if len(halfA) < minArea:
                            halfA.clear()
                            halfA.append(dataset_test.iloc[di])
                            checkA += 1
                            checkB = 0
                        else:
                            full = halfA + halfB
                            pattern_test.append(full)
                            halfA.clear()
                            halfB.clear()
                            halfA.append(dataset_test.iloc[di])
                            checkA += 1 
                            checkB = 0
                else:
                    halfA.append(dataset_test.iloc[di])
                    checkB = 0
            
            elif dataset_test['Distance'][di] > 0:
                if checkB == 0:
                    if len(halfA) < minArea:
                        halfA.clear()
                        halfB.clear()
                        halfB.append(dataset_test.iloc[di])
                        checkB += 1
                        checkA = 0
                    else:
                        if len(halfB) < minArea:
                            halfB.clear()
                            halfB.append(dataset_test.iloc[di])
                            checkB += 1
                            checkA = 0
                        else:
                            full = halfB + halfA
                            pattern_test.append(full)
                            halfA.clear()
                            halfB.clear()
                            halfB.append(dataset_test.iloc[di])
                            checkB += 1
                            checkA = 0
                else:
                    halfB.append(dataset_test.iloc[di])
                    checkA = 0
                    
                    
                    
                    



for x in pattern_test:
    x.reverse()
                    
for pat in range(0,len(pattern_test)):
    ps = pd.DataFrame(pattern_test[pat], columns = ['Date', 'Close', 'Distance'])
    ps.reset_index(level=0, inplace=True)
    plt.bar(ps.index, ps['Distance'])
    plt.title('Pattern_test: {}'.format(pat))
    plt.show() 


column = ['F1', 'F2', 'A1', 'F3', 'F4', 'A2', 'T']
features_test = pd.DataFrame(index = range(0, len(pattern_test)), columns=column)


ps = pd.DataFrame(pattern_test[0], columns = ['Date', 'Close', 'Distance'])
pH = []
nH = []
    
for d in range(0, len(ps)):
    if ps.iloc[d]['Distance'] > 0: pH.append(ps.iloc[d]['Distance'])
    elif ps.iloc[d]['Distance'] < 0: nH.append(ps.iloc[d]['Distance'])
    
if ps.iloc[0]['Distance'] > 0:
    features_test.iloc[0]['T'] = 0
    highest_point = max(pH)
    lowest_point = min(nH)
    
    if highest_point > pH[0]: features_test.iloc[0]['F1'] = 1
    elif highest_point == pH[0]: features_test.iloc[p]['F1'] = 2
        
    if highest_point > pH[-1]: features_test.iloc[0]['F2'] = 3
    elif highest_point == pH[-1]: features_test.iloc[p]['F2'] = 2
       
    if nH[0] > lowest_point: features_test.iloc[0]['F3'] = 3
    elif nH[0] == lowest_point: features_test.iloc[p]['F3'] = 2
            
    if nH[-1] > lowest_point: features_test.iloc[0]['F4'] = 1
    elif nH[-1] == lowest_point: features_test.iloc[p]['F4'] = 2
            
    features_test.iloc[0]['A1'] = round((round(abs(sum(pH)), 1) / ((round(abs(sum(pH)), 1)) + (round(abs(sum(nH)), 1)))) * 10, 1)
    features_test.iloc[0]['A2'] = round((round(abs(sum(nH)), 1) / ((round(abs(sum(pH)), 1)) + (round(abs(sum(nH)), 1)))) * 10, 1)
    
elif ps.iloc[0]['Distance'] < 0:
    features_test.iloc[0]['T'] = 1
    highest_point = max(pH)
    lowest_point = min(nH)
        
    if nH[0] == lowest_point: features_test.iloc[0]['F1'] = 2 
    elif nH[0] > lowest_point: features_test.iloc[0]['F1'] = 3
    
    if nH[-1] > lowest_point: features_test.iloc[0]['F2'] = 1 
    elif nH[-1] == lowest_point: features_test.iloc[0]['F2'] = 2
        
    if highest_point > pH[0]: features_test.iloc[0]['F3'] = 1
    elif highest_point == pH[0]: features_test.iloc[0]['F3'] = 2 
        
    if highest_point > pH[-1]: features_test.iloc[0]['F4'] = 3
    elif highest_point == pH[-1]: features_test.iloc[0]['F4'] = 2

    features_test.iloc[0]['A1'] = round((round(abs(sum(nH)), 1) / ((round(abs(sum(pH)), 1)) + (round(abs(sum(nH)), 1)))) * 10, 1)
    features_test.iloc[0]['A2'] = round((round(abs(sum(pH)), 1) / ((round(abs(sum(pH)), 1)) + (round(abs(sum(nH)), 1)))) * 10, 1)

test_pattern = []
dist_list = []
pattern_number = []
for val in range (len(pattern_test[0])): test_pattern.append(pattern_test[0][val]['Distance'])



for t in range(len(features)): 
    known_pattern = []
    if features_test.iloc[0]['T'] == features.iloc[t]['T']:
        for val in range (len(pattern[t])):
            known_pattern.append(pattern[t][val]['Distance'])
            
        x = np.array(test_pattern)
        y = np.array(known_pattern)
        
        distance, path = fastdtw(x, y, dist=euclidean)
        dist_list.append(distance)
        pattern_number.append(t)
    
    else: continue
    
s_path = min(dist_list)
p_numb = (pattern_number[(dist_list.index(s_path))])

for x in range(len(dataset)):
    if dataset.iloc[x]['Date'] == pattern[p_numb][-1]['Date']:
        cATR = dataset.iloc[x]['ATR']
        move1 = round((dataset.iloc[x]['Close'] - dataset.iloc[x+1]['Close']) * 10000, 2)
        move2 = round((dataset.iloc[x]['Close'] - dataset.iloc[x+2]['Close']) * 10000, 2)
        move3 = round((dataset.iloc[x]['Close'] - dataset.iloc[x+3]['Close']) * 10000, 2)
        move4 = round((dataset.iloc[x]['Close'] - dataset.iloc[x+4]['Close']) * 10000, 2)
        move5 = round((dataset.iloc[x]['Close'] - dataset.iloc[x+5]['Close']) * 10000, 2)
        
        if s_path <3:
            if abs(move1) >= cATR * 2:
                if move1 < 0:
                    print('ATR is {}, Price ROSE by {} pips on the first move'.format(cATR, abs(move1)))
                else: print('ATR is {}, Price FELL by {} pips on the first move'.format(cATR, abs(move1)))
                
            elif abs(move2) >= cATR * 2:
                if move2 < 0:
                    print('ATR is {}, Price ROSE by {} pips on the second move'.format(cATR, abs(move2)))
                else: print('ATR is {}, Price FELL by {} pips on the second move'.format(cATR, abs(move2)))
            
            elif abs(move3) >= cATR * 2:
                if move3 < 0:
                    print('ATR is {}, Price ROSE by {} pips on the third move'.format(cATR, abs(move3)))
                else: print('ATR is {}, Price FELL by {} pips on the third move'.format(cATR, abs(move3)))
            
            elif abs(move4) >= cATR * 2:
                if move4 < 0:
                    print('ATR is {}, Price ROSE by {} pips on the fourth move'.format(cATR, abs(move4)))
                else: print('ATR is {}, Price FELL by {} pips on the fouth move'.format(cATR, abs(move4)))
                
            elif abs(move5) >= cATR * 2:
                if move2 < 0:
                    print('ATR is {}, Price ROSE by {} pips on the fifth move'.format(cATR, abs(move5)))
                else: print('ATR is {}, Price FELL by {} pips on the fifth move'.format(cATR, abs(move5)))
            
            else: print('NO TRADE!!')
        
        else:
            print('NO TRADE!!')
            
            
            
            
            