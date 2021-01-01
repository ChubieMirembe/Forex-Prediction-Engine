# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px

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
    miniPattern = []
    positiveC = 0
    negativeC = 0
    for di in range(0, len(group[i])):
        if group[i]['Distance'][di] < 0:
            miniPattern.append(group[i]['Distance'][di])
            negativeC += 1
            continue
            
        elif group[i]['Distance'][di] > 0:
            miniPattern.append(group[i]['Distance'][di])
            positiveC += 1
            continue
       
        else:
            if negativeC < minArea:
                miniPattern.append(group[i]['Distance'][di])
                negativeC = 0
                positiveC += 1
            
            elif negativeC >= minArea:
                miniPattern.append(group[i]['Distance'][di])
                negativeC = 0
                positiveC += 1
                
            elif positiveC < minArea:
                miniPattern.append(group[i]['Distance'][di])
                positiveC = 0
                negativeC += 1
            
            elif positiveC >= minArea:
                miniPattern.append(group[i]['Distance'][di])
                positiveC = 0
                negativeC += 1
                
    pattern.append(miniPattern)
                

pattern = []
minArea = 5
for i in range(0, len(group)):
    zeroC = 0
    count = 0
    miniP = []
    positiveC = 0
    negativeC = 0
    for di in range(0, len(group[i])):
            if group[i]['Distance'][di] < 0:
                if positiveC in range(1, 5):
                    miniP.clear()
                    positiveC = 0
                else:
                    continue 
                
                miniP.append(group[i].iloc[di])
                negativeC += 1
            
            elif group[i]['Distance'][di] > 0:
                if negativeC in range(1,5):
                    miniP.clear()
                    negativeC = 0
                else:
                    continue
                
                miniP.append(group[i].iloc[di])
                positiveC += 1

    pattern.append(miniP)
            
ps = pd.DataFrame(pattern[1], columns = ['Date', 'Close', 'Distance'])
plt.bar(ps.index, ps['Distance'])
plt.show()  

'''
zeroC = 0
for v in range(0,len(group[10])):
    continue
'''   
    

rs =(range(1,5))
print(rs)
plt.hist((group[12]['Distance'].values), bins = 15)
plt.show
    
'''
- resulant trend vector, is the coefficient(predicted price) and the distrancce at each point from real price
    - will get nor marmalised
- So start counting the index from the beggining, everytime a pattern is found store it, and start counting again f
  from that index
- The counter adds up, and is used to seperate into different trends based on the laerger number
    

for loop to loop through every week
loop for every point in that week
start from the beginning, and go until price has tried to cross zero - check the area is above min 
    - Shouldnt i check highest point?: use a line to identify this value/point

draw the distribution line
find the highs in each cycle: if its larger than 5(index) find the crossing point as exchange points
worry about storing the first trend 
loop for the rest
 
1. Second if and else is checking whether its an uptrend or downtrend
    a. Will i need to store this info. before clustering?
2. 



I want it enter into the loop to check through the week
checking for uptrends 
if its an uptrend, then enter a condition
    - carry on checking if its still up 
        - append the values if it continues
    - if its switches before it gets to 5 steps, then come out of the condition(could be while loop)
    - if its more then start to store those values
then starts check again for another down move




rn its just looking through the pattern that start with more than 5 and collecting
- if theyre not up to 5 the list will not got the rest, why its stopping at 3
- if they're more than 5 it can carry on till the end
- Empty list if the first move is up

append thw first half, if lenght is longger than 5, then will start appending the second half
- otherwise start the other side as the first half and repeat the process

add zero cross
-to find one pattern in each week, will work on the rest later 

repeat for the other side



'''













#Feature Creation Algorithm


#Segmentation Algorithm 


#K-Means Algorithm                    


#ANN


#DTW                


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
