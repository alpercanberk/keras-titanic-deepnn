from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Activation, Dropout, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

# imputer for missing data
imputer = Imputer(missing_values=np.nan, strategy='mean')

# load training data from csv file in the directory
train_test_data = pd.read_csv('train.csv', index_col=0) #read data
train_test_data['is_dev'] = 0 #add an extra column to specify that this is not
#for dev. This is important because later on, we will concatenate all the data,
#and process it and we want all the data to be processed the same way.

dev_data = pd.read_csv('test.csv', index_col=0) #read "dev" data
dev_data['is_dev'] = 1 #add an extra column that this is dev data

#concatenate both dataframes
all_data = pd.concat((train_test_data, dev_data), axis=0)

#processes titles
def get_title_last_name(name):
    full_name = name.str.split(', ', n=0, expand=True)
    last_name = full_name[0]
    titles = full_name[1].str.split('.', n=0, expand=True)
    titles = titles[0]
    return(titles)

#Note: I took these processing functions from the internet and edited them a little bit

#adds a column for title
def get_titles_from_names(df):
    df['Title'] = get_title_last_name(df['Name'])
    df = df.drop(['Name'], axis=1)
    return(df)

#gets dummies for values that are not numbers such as titles and classes
def get_dummy_cats(df):
    return(pd.get_dummies(df, columns=['Title', 'Pclass', 'Sex', 'Embarked',
                                       'Cabin', 'Cabin_letter', 'Ticket', 'Fare']))
#gets cabin letter
def get_cabin_letter(df):
    df['Cabin'].fillna('Z', inplace=True)
    df['Cabin_letter'] = df['Cabin'].str[0]
    return(df)

def process_data(df):
    # preprocess titles, cabin, embarked
    df = get_titles_from_names(df)
    df['Embarked'].fillna('S', inplace=True)
    df = get_cabin_letter(df)

    # create dummies for categorial features
    df = get_dummy_cats(df)

    return(df)

#run process data
proc_data = process_data(all_data)

#split the data and get rid of is_dev column as it's not important for training
train_test_data = proc_data[proc_data['is_dev'] == 0]
dev_data = proc_data[proc_data['is_dev'] == 1]

train_test_data = train_test_data.drop(['is_dev'], axis=1)
dev_data = dev_data.drop(['is_dev'], axis=1)

#split the training data into training and test
mask = np.random.rand(len(all_data)) < 0.8
training_data, test_data = train_test_split(train_test_data, test_size=0.2)

#run the imputer to get rid of nans
imputer = imputer.fit(test_data)
test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns, index=test_data.index)

imputer = imputer.fit(training_data)
training_data = pd.DataFrame(imputer.transform(training_data), columns=training_data.columns, index=training_data.index)

#make the survived column a y vector for both train and test
X_train = training_data.drop('Survived', axis=1)
y_train = training_data['Survived']

X_test = test_data.drop('Survived', axis=1)
y_test = test_data['Survived']

dev_data = dev_data.drop('Survived', axis=1)

# creates the deep nn model, I didn't want to add more parameters as changing them affects
# the score negatively most of the time.
# all weights are initialized with He
# all biases are initialized with 0
def create_titanic_model(keep_prob):
    model = Sequential()

    keep_prob = keep_prob

    model.add(Dense(input_dim=X_train.shape[1], units=256,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(1-keep_prob))
    model.add(BatchNormalization())

    for _ in range(6):
        model.add(Dense(units=128, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dropout(1-keep_prob))
        # sweet sweet batch norm...
        model.add(BatchNormalization())


    model.add(Dense(units=64, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(1-keep_prob))

    model.add(Dense(units=32, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(1-keep_prob))

    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train.values, y_train.values, validation_data=(X_test.values,y_test.values), epochs=250, verbose=1)

    score, acc = model.evaluate(X_train.values, y_train.values)
    score, acc = model.evaluate(X_test.values, y_test.values)

    print('Train accuracy:', acc)
    print('Test accuracy:', acc)

    return model

#create the model with 0.8 keep_prob for dropout
model = create_model(0.8)

#turn the predictions into binary
prediction = (model.predict(dev_data).astype(float) > 0.5).astype(int)
#turn np.array into panda dataframe
prediction = pd.DataFrame(prediction, columns=['Survived'], index=dev_data.index)

#finally, create the csv file for submission
prediction.to_csv('survival_prediction.csv')
