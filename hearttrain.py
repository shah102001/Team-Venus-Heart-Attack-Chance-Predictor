'''
1. Number of times pregnant
2. Plasma glucose concentration in 2 hrs in an oral glucose
3. Diastolic Blood Pressure in mm of Hg
4. Triceps skin fold thickness in mm of Hg
5. 2 hour serum insulin (mu U/ml)
6. Body Mass Index (weight in kg/(height in m)^2)
7. Diabetes pedigree function
8. Age (years)
9. Class variable(0 or 1)
'''

from numpy import loadtxt#load excel or import .csv to process file from local directory
from keras.models import Sequential#in handling model applications, model from keras. sequential is a null or empty neural network which we will make from scratch
from keras.layers import Dense#to design layers depending on dense or not. dense is type of layer
from keras.models import model_from_json#helps to save models in format .json
#print("All imported")
dataset=loadtxt('heart.csv', delimiter=',',skiprows=1)#every column separated by comma
x=dataset[:,0:13]
y=dataset[:,13]
#print(x)
model=Sequential()#basic nn initialises an empty stack
model.add(Dense(12,input_dim=13,activation='relu'))#layer 1
model.add(Dense(13,activation='relu'))#layer 2
model.add(Dense(1,activation='sigmoid'))#layer 3 giving probability distribution
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=900,batch_size=10)#more epoch, more accuracy. gives accuracy and loss for every epoch

_, accuracy=model.evaluate(x,y)#overall accuracy
print('Accuracy:%.2f'%(accuracy*100))

model_json=model.to_json()#write to json file
with open("modelheart.json","w") as json_file:
              json_file.write(model_json)
model.save_weights("modelheart.h5")
print("Saved model to disk")
