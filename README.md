# Heart-Attack-Chance-Predictor


IDEA:
Our idea is to create a neural network model that trains itself based on a provided dataset and predicts the chances of heart attack for a particular person. The information in the dataset includes basic fields like age, gender, BMI, blood pressure and other biological factors of a person. Based on these past records and symptoms collected via a survey from various people and sources, our trained model predicts if a person has chances of a heart attack with maximum accuracy possible depending on the number of epochs.

GOALS:
•	Help the medical field in developing modern AI based systems to cure diseases.
•	Alert the person beforehand about risks and chances of heart attack in the future.
•	Use past records and symptoms to generalize the conditions for a particular disease.
•	Analyze a person’s heart condition regularly using AI.

ANYTHING ELSE:	The entire model shall be based on a python framework and using Neural Network with the training dataset created on the basis of symptoms and data provided by a large number of people.

Files:

1. heart.csv: Dataset used taken from kaggle.
2. hearttrain.py: This is the python code for trainning the model using sequential(for handling model applications) model from keras.
3. modelheart.json: The model is autosaved in a .json file for furthur testing the model.
4. modelheart.h5: loads the weights of the model.
5. hearttest.py: This is the code that predicts the output and checks the accuracy.

max. accuracy acheived was : 88.41%

How to execute?
1. save the heart.csv dataset
2. first run the hearttrain.py 
3. then run hearttest.py to get the predicted results on the test dataset and check accuracy.


