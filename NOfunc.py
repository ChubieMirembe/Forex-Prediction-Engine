
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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
        
    return d, TR

def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a

def creatingFFP(name, str, group_numb, pattern_numb, features_numb):
    name = pd.read_csv(str, delimiter='\t', 
                      names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ATR'],  
                      )
    x = 1
    TRDates = []
    TrueRanges = []
    
    while x < len(name):
        TRDate, TrueRange = TR(name['Date'][x], name['Close'][x], name['High'][x],
                               name['Low'][x], name['Open'][x], name['Close'][x-1])      
        TRDates.append(TRDate)
        TrueRanges.append(TrueRange)            
        x+=1
        
    fATR = ExpMovingAverage(TrueRanges, 14)
    for f in range(1, len(fATR)): name['ATR'][f] = round(fATR[f] * 10000, 2)
    
    name = name.drop(['Open', 'High', 'Low', 'Volume'], axis=1)

    s = []
    for x in range(0, len(name), 120): s.append(x)
    
    group_numb = []
    for x in s:
        start = x
        end = x+120
        data = (name[start:end])
        data.reset_index(level=0, inplace=True)
        if len(data) < 120: continue
        else: 
            data = data.drop(['index'], axis=1)
            group_numb.append(data)

    scaler = MinMaxScaler(feature_range = (-1, 1))
    regressor = LinearRegression()
    for g in range(0,len(group_numb)):
        X = np.reshape(group_numb[g].index, (120,-1))        
        regressor.fit(X, group_numb[g]['Close'])
        yi = regressor.predict(X)
        group_numb[g]['Distance'] = group_numb[g]['Close'].values - yi   
        res = group_numb[g]['Distance'].values.reshape(-1,1)
        group_numb[g]['Distance'] = scaler.fit_transform(res)    
    
    #Segmentation Algorithm 
    pattern_numb = []
    minArea = 5
    for i in range(0, len(group_numb)):
        checkA = 0
        checkB = 0
        halfA = []
        halfB = []
        for di in range(0, len(group_numb[i])):
                if group_numb[i]['Distance'][di] < 0:  
                    if checkA == 0:
                        if len(halfB) < minArea:
                            halfA.clear()
                            halfB.clear()
                            halfA.append(group_numb[i].iloc[di])
                            checkA += 1
                            checkB = 0
                        else:
                            if len(halfA) < minArea:
                                halfA.clear()
                                halfA.append(group_numb[i].iloc[di])
                                checkA += 1
                                checkB = 0
                            else:
                                full = halfA + halfB
                                pattern_numb.append(full)
                                halfA.clear()
                                halfB.clear()
                                halfA.append(group_numb[i].iloc[di])
                                checkA += 1 
                                checkB = 0
                    else:
                        halfA.append(group_numb[i].iloc[di])
                        checkB = 0
                
                elif group_numb[i]['Distance'][di] > 0:
                    if checkB == 0:
                        if len(halfA) < minArea:
                            halfA.clear()
                            halfB.clear()
                            halfB.append(group_numb[i].iloc[di])
                            checkB += 1
                            checkA = 0
                        else:
                            if len(halfB) < minArea:
                                halfB.clear()
                                halfB.append(group_numb[i].iloc[di])
                                checkB += 1
                                checkA = 0
                            else:
                                full = halfB + halfA
                                pattern_numb.append(full)
                                halfA.clear()
                                halfB.clear()
                                halfB.append(group_numb[i].iloc[di])
                                checkB += 1
                                checkA = 0
                    else:
                        halfB.append(group_numb[i].iloc[di])
                        checkA = 0
                        
    #Feature Creation Algorithm
    columns = ['F1', 'F2', 'A1', 'F3', 'F4', 'A2', 'T']
    features_numb = pd.DataFrame(index = range(0, len(pattern_numb)), columns=columns)
    
    for p in range(0, len(pattern_numb)):
        ps = pd.DataFrame(pattern_numb[p], columns = ['Date', 'Close', 'Distance'])
        pH = []
        nH = []
        
        for d in range(0, len(ps)):
            if ps.iloc[d]['Distance'] > 0: pH.append(ps.iloc[d]['Distance'])
            elif ps.iloc[d]['Distance'] < 0: nH.append(ps.iloc[d]['Distance'])
        
        if ps.iloc[0]['Distance'] > 0:
            features_numb.iloc[p]['T'] = 0
            highest_point = max(pH)
            lowest_point = min(nH)
            
            if highest_point > pH[0]: features_numb.iloc[p]['F1'] = 1
            elif highest_point == pH[0]: features_numb.iloc[p]['F1'] = 2
            
            if highest_point > pH[-1]: features_numb.iloc[p]['F2'] = 3
            elif highest_point == pH[-1]: features_numb.iloc[p]['F2'] = 2
            
            if nH[0] > lowest_point: features_numb.iloc[p]['F3'] = 3
            elif nH[0] == lowest_point: features_numb.iloc[p]['F3'] = 2
                
            if nH[-1] > lowest_point: features_numb.iloc[p]['F4'] = 1
            elif nH[-1] == lowest_point: features_numb.iloc[p]['F4'] = 2
                
            features_numb.iloc[p]['A1'] = round((round(abs(sum(pH)), 1) / ((round(abs(sum(pH)), 1)) + 
                                                                      (round(abs(sum(nH)), 1)))) * 10, 1)
            features_numb.iloc[p]['A2'] = round((round(abs(sum(nH)), 1) / ((round(abs(sum(pH)), 1)) + 
                                                                      (round(abs(sum(nH)), 1)))) * 10, 1)
        elif ps.iloc[0]['Distance'] < 0:
            features_numb.iloc[p]['T'] = 1
            highest_point = max(pH)
            lowest_point = min(nH)
            
            if nH[0] == lowest_point: features_numb.iloc[p]['F1'] = 2 
            elif nH[0] > lowest_point: features_numb.iloc[p]['F1'] = 3
            
            if nH[-1] > lowest_point: features_numb.iloc[p]['F2'] = 1 
            elif nH[-1] == lowest_point: features_numb.iloc[p]['F2'] = 2
            
            if highest_point > pH[0]: features_numb.iloc[p]['F3'] = 1
            elif highest_point == pH[0]: features_numb.iloc[p]['F3'] = 2 
            
            if highest_point > pH[-1]: features_numb.iloc[p]['F4'] = 3
            elif highest_point == pH[-1]: features_numb.iloc[p]['F4'] = 2
    
            features_numb.iloc[p]['A1'] = round((round(abs(sum(nH)), 1) / ((round(abs(sum(pH)), 1)) + 
                                                                      (round(abs(sum(nH)), 1)))) * 10, 1)
            features_numb.iloc[p]['A2'] = round((round(abs(sum(pH)), 1) / ((round(abs(sum(pH)), 1)) + 
                                                                      (round(abs(sum(nH)), 1)))) * 10, 1)
    return name, pattern_numb, features_numb
    # return features_numb
            
DF1 = creatingFFP('dataset_1', 'AUDCAD60(full).csv', 'group_1', 'pattern_1', 'features_1')
Pattern1 = DF1[1]
feature1 = DF1[2]
DF2 = creatingFFP('dataset_2', 'AUDCHF60(full).csv', 'group_2', 'pattern_2', 'features_2')
Pattern2 = DF2[1]
feature2 = DF2[2]
DF3 = creatingFFP('dataset_3', 'AUDJPY60(full).csv', 'group_3', 'pattern_3', 'features_3')
Pattern3 = DF3[1]
feature3 = DF3[2]
DF4 = creatingFFP('dataset_4', 'AUDNZD60(full).csv', 'group_4', 'pattern_4', 'features_4')
Pattern4 = DF4[1]
feature4 = DF4[2]
DF5 = creatingFFP('dataset_5', 'AUDUSD60(full).csv', 'group_5', 'pattern_5', 'features_5')
Pattern5 = DF5[1]
feature5 = DF5[2]
DF6 = creatingFFP('dataset_6', 'CADJPY60(full).csv', 'group_6', 'pattern_6', 'features_6')
Pattern6 = DF6[1]
feature6 = DF6[2]
DF7 = creatingFFP('dataset_7', 'CHFJPY60(full).csv', 'group_7', 'pattern_7', 'features_7')
Pattern7 = DF7[1]
feature7 = DF7[2]
DF8 = creatingFFP('dataset_8', 'EURAUD60(full).csv', 'group_8', 'pattern_8', 'features_8')
Pattern8 = DF8[1]
feature8 = DF8[2]
DF9 = creatingFFP('dataset_9', 'EURCAD60(full).csv', 'group_9', 'pattern_9', 'features_9')
Pattern9 = DF9[1]
feature9 = DF9[2]
DF10 = creatingFFP('dataset_10', 'EURCHF60(full).csv', 'group_10', 'pattern_10', 'features_10')
Pattern10 = DF10[1]
feature10 = DF10[2]
DF11 = creatingFFP('dataset_11', 'EURGBP60(full).csv', 'group_11', 'pattern_11', 'features_11')
Pattern11 = DF11[1]
feature11 = DF11[2]
DF12 = creatingFFP('dataset_12', 'EURJPY60(full).csv', 'group_12', 'pattern_12', 'features_12')
Pattern12 = DF12[1]
feature12 = DF12[2]
DF13 = creatingFFP('dataset_13', 'EURUSD60(full).csv', 'group_13', 'pattern_13', 'features_13')
Pattern13 = DF13[1]
feature13 = DF13[2]
DF14 = creatingFFP('dataset_14', 'GBPAUD60(full).csv', 'group_14', 'pattern_14', 'features_14')
Pattern14 = DF14[1]
feature14 = DF14[2]
DF15 = creatingFFP('dataset_15', 'GBPJPY60(full).csv', 'group_15', 'pattern_15', 'features_15')
Pattern15 = DF15[1]
feature15 = DF15[2]
DF16 = creatingFFP('dataset_16', 'GBPUSD60(full).csv', 'group_16', 'pattern_16', 'features_16')
Pattern16 = DF16[1]
feature16 = DF16[2]
DF17 = creatingFFP('dataset_17', 'NZDUSD60(full).csv', 'group_17', 'pattern_17', 'features_17')
Pattern17 = DF17[1]
feature17 = DF17[2]
DF18 = creatingFFP('dataset_18', 'USDCAD60(full).csv', 'group_18', 'pattern_18', 'features_18')
Pattern18 = DF18[1]
feature18 = DF18[2]
DF19 = creatingFFP('dataset_19', 'USDCHF60(full).csv', 'group_19', 'pattern_19', 'features_19')
Pattern19 = DF19[1]
feature19 = DF19[2]
DF20 = creatingFFP('dataset_20', 'USDJPY60(full).csv', 'group_20', 'pattern_20', 'features_20')
Pattern20 = DF20[1]
feature20 = DF20[2]

allFeatures = pd.concat([feature1, feature2, feature3, feature4, feature5, feature6, feature7,
                         feature8, feature9, feature10, feature11, feature12, feature13, feature14,
                         feature15, feature16, feature17, feature18, feature19, feature20])
allFeatures.reset_index(inplace=True, drop=True)

allPatterns = Pattern1 + Pattern2 + Pattern3 + Pattern4 + Pattern5 + Pattern6 + Pattern7 + Pattern8 
allPattern2 =  Pattern9 +  Pattern10 + Pattern11 + Pattern12 + Pattern13 + Pattern14 + Pattern15 + Pattern16 
allPattern3 = Pattern17 + Pattern18 + Pattern19 + Pattern20

AP = allPatterns + allPattern2 + allPattern3




                    
## TEST!!

dataset_test = pd.read_csv('AU.csv', skiprows=1,
                      names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
dataset_test = dataset_test.drop(['Open', 'High', 'Low', 'Volume'], axis=1)
# dataset_test = dataset_test.drop(dataset_test.index[0:41])
dataset_test = dataset_test.tail(60)
dataset_test['Date'] = pd.to_datetime(dataset_test['Date'])
dataset_test.reset_index(drop=True, inplace=True)

scaler = MinMaxScaler(feature_range = (-1, 1))
regressor_test = LinearRegression()
    
X = np.reshape(dataset_test.index, (60,-1))        
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
minArea = 5
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
                    

# Reversing the pattern
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
    elif highest_point == pH[0]: features_test.iloc[0]['F1'] = 2
        
    if highest_point > pH[-1]: features_test.iloc[0]['F2'] = 3
    elif highest_point == pH[-1]: features_test.iloc[0]['F2'] = 2
       
    if nH[0] > lowest_point: features_test.iloc[0]['F3'] = 3
    elif nH[0] == lowest_point: features_test.iloc[0]['F3'] = 2
            
    if nH[-1] > lowest_point: features_test.iloc[0]['F4'] = 1
    elif nH[-1] == lowest_point: features_test.iloc[0]['F4'] = 2
            
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



for t in range(len(allFeatures)): 
    known_pattern = []
    if features_test.iloc[0]['T'] == allFeatures.iloc[t]['T']:
        for val in range (len(AP[t])):
            known_pattern.append(AP[t][val]['Distance'])
            
        x = np.array(test_pattern)
        y = np.array(known_pattern)
        
        distance, path = fastdtw(x, y, dist=euclidean)
        dist_list.append(distance)
        pattern_number.append(t)
    
    else: continue
    
s_path = min(dist_list)
p_numb = (pattern_number[(dist_list.index(s_path))])




def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix

kp = []
for val in range (len(AP[p_numb])): kp.append(AP[p_numb][val]['Distance'])
track = dtw(test_pattern, kp)

x = np.array(test_pattern)
y = np.array(kp)
        
distance, path = fastdtw(x, y, dist=euclidean)
print(distance)

if p_numb in range(1125): dataset = DF1[0]
elif p_numb in range(1125, 2175): dataset = DF2[0]
elif p_numb in range(2175, 3230): dataset = DF3[0]
elif p_numb in range(3230, 4288): dataset = DF4[0]
elif p_numb in range(4288, 5421): dataset = DF5[0]
elif p_numb in range(5421, 6499): dataset = DF6[0]
elif p_numb in range(6499, 7634): dataset = DF7[0]
elif p_numb in range(7634, 8642): dataset = DF8[0]
elif p_numb in range(8642, 9682): dataset = DF9[0]
elif p_numb in range(9682, 10709): dataset = DF10[0]
elif p_numb in range(10709, 11825): dataset = DF11[0]
elif p_numb in range(11825, 12897): dataset = DF12[0]
elif p_numb in range(12897, 14016): dataset = DF13[0]
elif p_numb in range(14016, 15151): dataset = DF14[0]
elif p_numb in range(15151, 16266): dataset = DF15[0]
elif p_numb in range(16266, 17389): dataset = DF16[0]
elif p_numb in range(17389, 18465): dataset = DF17[0]
elif p_numb in range(18465, 19509): dataset = DF18[0]
elif p_numb in range(19509, 20601): dataset = DF19[0]
elif p_numb in range(20601, 21628): dataset = DF20[0]

    


if pattern_test[0][-1]['Date'] != dataset_test.iloc[-1]['Date']:
    print('NO TRADE!!')
else:
    for x in range(len(dataset)):
        if dataset.iloc[x]['Date'] == AP[p_numb][-1]['Date']:
            cATR = dataset.iloc[x]['ATR']
            move1 = round((dataset.iloc[x]['Close'] - dataset.iloc[x+1]['Close']) * 10000, 2)
            move2 = round((dataset.iloc[x]['Close'] - dataset.iloc[x+2]['Close']) * 10000, 2)
            move3 = round((dataset.iloc[x]['Close'] - dataset.iloc[x+3]['Close']) * 10000, 2)
            move4 = round((dataset.iloc[x]['Close'] - dataset.iloc[x+4]['Close']) * 10000, 2)
            move5 = round((dataset.iloc[x]['Close'] - dataset.iloc[x+5]['Close']) * 10000, 2)
            
            if s_path <10:
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

