import pandas as pd
import numpy as np
import math as mt
from random import randint

df = pd.read_csv("https://raw.githubusercontent.com/dbdmg/data-science-lab/master/datasets/mnist_test.csv",header=None)
# The file has:
# - first column -> the number of reference
# - other 784 columns -> fades of pixels
# then 10000 rows.

df.tail() #we check the final lines

shuffleTest = df.sample(frac=0.2) #first we use 20% of dataset for training
x_test = shuffleTest.iloc[:,1:785].values
y_test = shuffleTest.iloc[:,0].values


def FromStringToValue(species_list):
    class_values= set(species_list) 
    LUT = {}
    print("Legend as it follows:")
    print()
    for i, species in enumerate(class_values):
        LUT[species] = i
        print(f"{species} is equal to {i}")
    updated_speciesList = []
    for row in species_list:
        row = LUT[row]
        updated_speciesList.append(row)
    return updated_speciesList

y_test = FromStringToValue(y_test)
dasetTest=np.column_stack((x_test,y_test)) #merge the column to the last column of the new dataset

r = randint(0,1999) #random number from 0 to 1999 to choose the testing row randomly

rowInit = dasetTest[r]
speciesInit = int(rowInit[784])
print()
print(f"Test row is: {rowInit}")


def EuclideanDistance(r1, r2): #we define the p_i and q_i of each pair of rows
    distance = 0.0
    for i in range(len(r1)-1):
        distance += (r1[i] - r2[i])**2
    return mt.sqrt(distance)


def CosineDistance(r1,r2):
    distance = 0.0
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    for i in range(len(r1)-1):
        sum1 += (r1[i]*r2[i])
        sum2 += r1[i]**2
        sum3 += r2[i]**2
    distance = 1-abs(sum1/mt.sqrt(sum2*sum3))
    return distance


def ManhattanDistance(r1,r2):
    distance = 0.0
    for i in range(len(r1)-1):
        distance += abs((r1[i] - r2[i]))
    return distance


def ChooseTypeOfDistance (k,r1,r2):
    types = {'Euclidean': 0, 'Cosine': 1, 'Manhattan': 2}
    if k == types['Euclidean']:
        return EuclideanDistance(r1,r2)
    elif k == types['Cosine']:
        return CosineDistance(r1,r2)
    elif k == types['Manhattan']:
        return ManhattanDistance(r1,r2)


def getNeighbors(trainingList, testRow, numNeighbors):
    distances = []
    for trainingRow in trainingList:
        dist = ChooseTypeOfDistance(0,testRow, trainingRow) #first we calculate eucl dist
        distances.append((trainingRow, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    
    for i in range(numNeighbors):
        neighbors.append(distances[i][0])
    return neighbors

'''
print()
neighbors = getNeighbors(dasetTest,dasetTest[r], 100)

for neighbor in neighbors:
    print(neighbor)
    print()
'''

def PredictClassification(trainingList, testRow, numNeighbors):
    neighbors = getNeighbors(trainingList, testRow, numNeighbors)
    outputs = [el[-1] for el in neighbors]
    prediction = max(set(outputs), key=outputs.count) #same idea of sort
    return prediction


neighbors = getNeighbors(dasetTest, rowInit, 100)
prediction = PredictClassification(dasetTest, rowInit, 100)

print("Data selected:")
print(list(rowInit))
print()
print(f"Predicted {int(prediction)}")
print()
print(f"Verifying: the expected one is {speciesInit}")
print()
