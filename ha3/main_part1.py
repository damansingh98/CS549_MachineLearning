import numpy as np
import matplotlib.pyplot as plt
from scipy import rand
from getDataset import getDataSet
from sklearn.linear_model import LogisticRegression


# Starting codes

# Fill in the codes between "%PLACEHOLDER#start" and "PLACEHOLDER#end"

# step 1: generate dataset that includes both positive and negative samples,
# where each sample is described with two features.
# 250 samples in total.

[X, y] = getDataSet()  # note that y contains only 1s and 0s,

# create figure for all charts to be placed on so can be viewed together
fig = plt.figure()


def func_DisplayData(dataSamplesX, dataSamplesY, chartNum, titleMessage):
    idx1 = (dataSamplesY == 0).nonzero()  # object indices for the 1st class
    idx2 = (dataSamplesY == 1).nonzero()
    ax = fig.add_subplot(1, 3, chartNum)
    # no more variables are needed
    plt.plot(dataSamplesX[idx1, 0], dataSamplesX[idx1, 1], 'r*')
    plt.plot(dataSamplesX[idx2, 0], dataSamplesX[idx2, 1], 'b*')
    # axis tight
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_title(titleMessage)


# plotting all samples
func_DisplayData(X, y, 1, 'All samples')

# number of training samples
nTrain = 120

######################PLACEHOLDER 1#start#########################
# write you own code to randomly pick up nTrain number of samples for training and use the rest for testing.
# WARNIN: 

# length of X array
maxIndex = len(X)

# Create a random array of size maxIndex (250)
randomTrainingSamples = np.random.choice(maxIndex, maxIndex, replace = False)

# Split data ratio = 48/52 (train/test) using nTrain
pickTraining = randomTrainingSamples[:nTrain] # 48% for training

pickTesting = randomTrainingSamples[nTrain:] # 52% for testing

trainX =  X[pickTraining]   # training samples
trainY =  y[pickTraining] # labels of training samples    nTrain X 1

testX =   X[pickTesting] # testing samples               
testY =  y[pickTesting] # labels of testing samples     nTest X 1

####################PLACEHOLDER 1#end#########################

# plot the samples you have pickup for training, check to confirm that both negative
# and positive samples are included.
func_DisplayData(trainX, trainY, 2, 'training samples')
func_DisplayData(testX, testY, 3, 'testing samples')

# show all charts
plt.show()


#  step 2: train logistic regression models


######################PLACEHOLDER2 #start#########################
# in this placefolder you will need to train a logistic model using the training data: trainX, and trainY.
# please delete these coding lines and use the sample codes provided in the folder "codeLogit"

'''First LR Method: Sklearn learn training'''
# training
model = LogisticRegression(fit_intercept=True, C=1e15).fit(trainX, trainY)

# performance
perform = model.score(testX, testY)
print("Performance: ", perform)

''' Second Method: Gradient Descent training '''
from codeLogit.main_logit import *

# m = len(trainX) # number of samples
# cst = [] # store theta
# cost = [] #cost function
# for x in range(250):
#     new_theta = Gradient_Descent(trainX, trainY,theta,m,alpha)
#     theta = new_theta
#     cst.append(theta)
#     print("Cost function: ", Cost_Function(trainX, trainY,theta,m))
    


######################PLACEHOLDER2 #end #########################


# step 3: Use the model to get class labels of testing samples.

######################PLACEHOLDER3 #start#########################
# codes for making prediction, 
# with the learned model, apply the logistic model over testing samples
# hatProb is the probability of belonging to the class 1.


# WARNING: please DELETE THE FOLLOWING CODEING LINES and write your own codes for making predictions

'''Sklearn LR Method Predictions:'''

yHat = model.predict(testX).astype(int)


'''Gradient Descent Method Predictions:'''

# length = len(testX)
# yHat = []
# for i in range(length):
# 	yHat.append(round(Prediction(testX[i],theta)))



######################PLACEHOLDER 3 #end #########################


# step 4: evaluation
# compare predictions yHat and and true labels testy to calculate average error and standard deviation
testYDiff = np.abs(yHat - testY)
avgErr = np.mean(testYDiff)
stdErr = np.std(testYDiff)

print('average error: {} ({})\n'.format(avgErr, stdErr))

######### CONFUSION MATRIX ###########################

def func_calConfusionMatrix(predY, trueY):
    #There are two unique classes (0 and 1)
    confMat = np.empty([2, 2], dtype = int)*0
    length = len(trueY)
    index = 0
    # Insert values into matrix based on the index
    while(index < length):
        x = int(trueY[index])
        y = int(predY[index])
        confMat[x][y] += 1
        index += 1

    #ConfMat:
            # [A B]
            # [C D]
    
    #Using the formula for Accuracy: (A+D)/(A+B+C+D)
    accuracy = (confMat[0][0] + confMat[1][1])/(confMat[0][0] + confMat[0][1]+confMat[1][0]+confMat[1][1])
    
    #Using the formula for precision: (A/(A+C)) 
    precision = confMat[0][0]/(confMat[0][0] + confMat[1][0])

    #Using the formula for recall: (A/(A+B))
    recall = confMat[0][0] /(confMat[0][0] + confMat[0][1])

    return accuracy, precision, recall


'''   Confusion Matrix Computation  '''

results = func_calConfusionMatrix(yHat, testY)
print("Confusion Matrix Results: \n")
print("Accuracy: {:.3f}%, Precision: {:.3f}%, Recall: {:.3f}%".format(results[0]*100, results[1]*100, results[2]*100))

