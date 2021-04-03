from numpy import loadtxt#load excel or import .csv to process file from local directory
from keras.models import Sequential#in handling model applications, model from keras. sequential is a null or empty neural network which we will make from scratch
from keras.layers import Dense#to design layers depending on dense or not. dense is type of layer
from keras.models import model_from_json#helps to save models in format .json
import pandas as pd
import csv

#print("All imported")
dataset=loadtxt('heart.csv', delimiter=',',skiprows=1)#every column separated by comma
x=dataset[:,0:13]#frst 8 columns
y=dataset[:,13]#last 9th column
#print(x)
json_file=open('modelheart.json','r')
loaded_model_json=json_file.read()
json_file.close()
model=model_from_json(loaded_model_json)
model.load_weights("modelheart.h5")
print("Loaded model from disk")

predictions=model.predict_classes(x)

#x contains 8 columns from where to be predicted
for i in range(250,255):#testing input no of rows
    print('%s => %d (expected %d)'%(x[i].tolist(),predictions[i],y[i].tolist()))

    #max accuracy rate reached=88.41%
