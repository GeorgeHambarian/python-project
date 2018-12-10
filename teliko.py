import matplotlib.pyplot as plt 						
import numpy as np										
from sklearn.linear_model import Ridge					
from sklearn.preprocessing import PolynomialFeatures 	
from sklearn.pipeline import make_pipeline 				

#open and read file
f = open("tm-timi.txt", "r")
#f = open("kapoia.txt","r")
L = f.readlines()		#square meters and price
f.close()

x,y = [],[]
for i in L:
	i=i.split(' ')
	x.append(int(i[0])); y.append(int(i[1]))	
x, y = (list(t) for t in zip(*sorted(zip(x, y))))	#sort based on price

#inputs to test
count = 0
while True:
	s = eval(input('give a list with square meters of your dream house and your dream price: '))
	if s == 0: break
	else: count+=1; x+=[s[0]]; y+=[s[1]]		#add new square meters in x and new prices in y

#make x and y matrix array for train and test
x = np.array([x]).reshape(-1,1)
y = np.array([y]).reshape(-1,1)

xTrain = x[:-count]				#square meters for train
yTrain = y[:-count]				#prices for train
xTest = x[-count:]				#count inputs, new square meters
yTest = y[-count:]				#count inputs, new prices

def worth(xTest, yTest, yPred):
	W = []; N = []	#Worth and Not worth
	for i,j,k in zip(xTest,yTest,yPred):
	    if j>k: N.append([i[0],j[0]])		#overpriced --> not worth
	    else: W.append([i[0],j[0]])			#nice price --> worth
	return (W,N)

#polynomial regression for degrees 1,2,3,4
fig, ax = plt.subplots(2,2)
fig.suptitle('Is it worth renting a house?\nblue is worth and red is not')			#title for all subplots

for m,n,degree in zip([0,1,0,1],[0,0,1,1],[1,2,3,4]):		#mxn matrix for plot, degree
    model = make_pipeline(PolynomialFeatures(degree), Ridge())					#create polynomial for each degree
    model.fit(xTrain,yTrain)													#fit the model for xTrain and yTrain
    yPred = model.predict(xTrain)												#make yPredictions for each xTest
    W,N = worth(xTrain, yTrain, yPred)											#worth or not worth for train data
    ax[m,n].plot([i[0] for i in W],[i[1] for i in W],'+',color='blue')			#plot worth data for trained
    ax[m,n].plot([i[0] for i in N],[i[1] for i in N],'+',color='red')			#plot not worth data for trained
    ax[m,n].plot(xTrain,yPred, color='black', linewidth=2)						#plot the dividing line
    ndPred = model.predict(xTest)												#new data Predictions
    W,N = worth(xTest, yTest, ndPred)											#worth or not worth for new data
    ax[m,n].plot([i[0] for i in W],[i[1] for i in W],'o',color='blue')			#plot worth data for new data
    ax[m,n].plot([i[0] for i in N],[i[1] for i in N],'o',color='red')			#plot not worth data for new data
    ax[m,n].grid(True)
    ax[m,n].set_title('polynomial regression: degree=%d'%(degree))
    ax[m,n].set_xlabel('square meters of the house')
    ax[m,n].set_ylabel('price in €')

plt.show()