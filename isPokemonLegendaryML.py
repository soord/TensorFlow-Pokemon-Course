# pip3 install tensorflow will use python3

import tensorflow as tf #tensorflow ML models
from tensorflow import keras #package in tensorflow
import numpy as np #number and math library standard to python
import pandas as pd #csv and excel file processor in excel
import matplotlib.pyplot as plt #math plotting library in python
from sklearn import preprocessing #from science kit library in python

# need ~ before relative path in the file structure in order to read csv file
df = pd.read_csv('~/Documents/Programming/MachineLearning/Tensorflow/pokemon/pokemon_alopez247.csv')
# df.columns shows features in a dataset
# df means DataFrame and is a object containing the data you load into it

# sets dataframe to only use the useful columns to our usecase
df = df[['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style']]

# sets column as numerical data and represent as a boolean 0 or 1
df['isLegendary'] = df['isLegendary'].astype(int)

# There are a few other categories that we'll need to convert as well. Let's look at "Type_1" as an example. Pokémon have associated elements, such as water and fire. Our first intuition at converting these to numbers could be to just assign a number to each category, such as: Water = 1, Fire = 2, Grass = 3 and so on. This isn't a good idea because these numerical assignments aren't ordinal; they don't lie on a scale. By doing this, we would be implying that Water is closer to Fire than it is Grass, which doesn't really make sense.

# The solution to this is to create dummy variables. By doing this we'll be creating a new column for each possible variable. There will be a column called "Water" that would be a 1 if it was a water Pokémon, and a 0 if it wasn't. Then there will be another column called "Fire" that would be a 1 if it was a fire Pokémon, and so forth for the rest of the types. This prevents us from implying any pattern or direction among the types. Let's do that:

def dummy_creation(df, dummy_categories):
    for i in dummy_categories:
        df_dummy = pd.get_dummies(df[i])
        df = pd.concat([df,df_dummy],axis=1)
        df = df.drop(i, axis=1)
    return(df)

# creates dummy Dataframe and concats to original df for each column then for each var given it will create a separate column with each of the values and add 1 or 0 for those columns. 
df = dummy_creation(df, ['Egg_Group_1', 'Body_Style', 'Color','Type_1', 'Type_2'])

# --------------------- Splitting into training and testing sets------------------------


# we want to split the data into testing and training sets. In this case with pokemon, we want to split it by generation. So this method will put generation == 1 pokemon in the test dataset and everything else in the train dataset. Then it drops the generation data from the dataset 

# Generally you want about a 70% train and 30% test split 
def train_test_splitter(DataFrame, column):
    df_train = DataFrame.loc[df[column] != 1]
    df_test = DataFrame.loc[df[column] == 1]
    df_train = df_train.drop(column, axis=1)
    df_test = df_test.drop(column, axis=1)
    return(df_train, df_test)


# uses Generation column for the previous method
df_train, df_test = train_test_splitter(df, 'Generation')


# now we have an answer column "isLegendary" that we need to drop from the datasets. We can have the answer keys in the data when we are trying to predict it
def label_delineator(df_train, df_test, label):
    
    train_data = df_train.drop(label, axis=1).values
    train_labels = df_train[label].values
    test_data = df_test.drop(label,axis=1).values
    test_labels = df_test[label].values
    return(train_data, train_labels, test_data, test_labels)

# returns the data separated from the labels for the two datasets of training and testing
train_data, train_labels, test_data, test_labels = label_delineator(df_train, df_test, 'isLegendary')


# normalize the data so everything is on the same scale. This will use MinMaxScaler normalization
def data_normalizer(train_data, test_data):
    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    return(train_data, test_data)

# run the method
train_data, test_data = data_normalizer(train_data, test_data)

# data is set up now. 


# ----------------------Machine Learning and Tensorflow ---------------------------



# this will set up the project using tensorflow. keras is a tensorflow library and this will set up a neural network
length = train_data.shape[1]

model = keras.Sequential()
# these are the layers. The first one uses a 'ReLU' (Rectified Linear Unit)' activation function and has the number of inputs into the system. We just set this at 500 to catch all the inputs. 
model.add(keras.layers.Dense(500, activation='relu', input_shape=[length,]))

# next layer is a softMax logistical regression. It will dilineate into two groups
model.add(keras.layers.Dense(2, activation='softmax'))

# compiles the model based off the metrics we want
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Here we're just feeding three parameters to model.compile. We pick an optimizer, which determines how the model is updated as it gains information, a loss function, which measures how accurate the model is as it trains, and metrics, which specifies which information it provides so we can analyze the model.

# The optimizer we're using is the Stochastic Gradient Descent (SGD) optimization algorithm, but there are others available. For our loss we're using sparse_categorical_crossentropy. If our values were one-hot encoded, we would want to use "categorial_crossentropy" instead.

model.fit(train_data, train_labels, epochs=400)
# needs are our training data, our training labels, and the number of epochs. One epoch is when the model has iterated over every sample once. Essentially the number of epochs is equal to the number of times we want to cycle through the data. We'll start with just 1 epoch, and then show that increasing the epoch improves the results.


# run the tests and get the accuracy and loss
loss_value, accuracy_value = model.evaluate(test_data, test_labels)
print(f'Our test accuracy was {accuracy_value}')
# >>> Our test accuracy was 0.980132




# predictor that will tell the training data if it is correct or not
def predictor(test_data, test_labels, index):
    prediction = model.predict(test_data)
    if np.argmax(prediction[index]) == test_labels[index]:
        print(f'This was correctly predicted to be a \"{test_labels[index]}\"!')
    else:
        print(f'This was incorrectly predicted to be a \"{np.argmax(prediction[index])}\". It was actually a \"{test_labels[index]}\".')
        return(prediction)


predictor(test_data, test_labels, 149)