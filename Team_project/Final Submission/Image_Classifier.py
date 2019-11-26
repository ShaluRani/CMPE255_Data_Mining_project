import os
from glob import glob
import pandas as pd
import numpy as np

#Reading the Dataset from the HAM10000_metadata.csv file
root = 'data'
pathCsv = os.path.join(root, 'HAM10000_metadata.csv')
df = pd.read_csv(pathCsv)
print("The head of the current DataFrame")
print(df.head())
print("\n")


#Checking for the null values in the DataFrame
print("Checking for null values")
print(df.isnull().sum())
df['age'].fillna((df['age'].mean()), inplace=True)    #Replacing null values with the mean.
print("Checking for null values after replacing")
print(df.isnull().sum())
print("\n")


#Creating the class values for each type
df['Class'] = pd.Categorical(df['dx']).codes

Path = {}
for item in glob(os.path.join(root, 'reshaped', '*.jpg')):
    filename = os.path.splitext(os.path.basename(item))[0]
    Path[filename] = item
print("Creating a column 'path' that contains path for the corresponding image")
df['path'] = df['image_id'].map(Path.get)
print(df.head())
print("\n")

print("Value types in each column")
print(df.dtypes)
print("\n")

print("Converting the image into Numpy array")
from PIL import Image
df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x)))
print(df.head())
print("\n")

print("Checking the class distribution")
df['dx'].value_counts()
print("\n")


#Making a temporary dataframe for each class for Balancing
df_nv = df[df.dx=='nv']
df_mel = df[df.dx=='mel']
df_bkl = df[df.dx=='bkl']
df_bcc = df[df.dx=='bcc']
df_akiec = df[df.dx=='akiec']
df_vasc = df[df.dx=='vasc']
df_df = df[df.dx=='df']


#Resampling to achieve balanced data
#Keeping 1500 samples for each class
from sklearn.utils import resample

df_df_upsample = resample(df_df, 
                                 replace=True,     
                                 n_samples=1500,    
                                 random_state=123)
df_nv_downsample = resample(df_nv, 
                                 replace=True,     
                                 n_samples=1500,    
                                 random_state=123)
df_mel_upsample = resample(df_mel, 
                                 replace=True,     
                                 n_samples=1500,    
                                 random_state=123)
df_bkl_upsample = resample(df_bkl, 
                                 replace=True,     
                                 n_samples=1500,    
                                 random_state=123)
df_bcc_upsample = resample(df_bcc, 
                                 replace=True,     
                                 n_samples=1500,    
                                 random_state=123)
df_akiec_upsample = resample(df_akiec, 
                                 replace=True,     
                                 n_samples=1500,    
                                 random_state=123)
df_vasc_upsample = resample(df_vasc, 
                                 replace=True,     
                                 n_samples=1500,    
                                 random_state=123)


df_resampled = pd.DataFrame(columns=['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization',
        'path', 'image'],
      dtype='object')



#Concatinating all temporary dataframes into single dataframe
df_resampled = pd.concat([df_resampled, df_df_upsample])

df_resampled = pd.concat([df_resampled, df_nv_downsample])

df_resampled = pd.concat([df_resampled, df_mel_upsample])

df_resampled = pd.concat([df_resampled, df_bkl_upsample])

df_resampled = pd.concat([df_resampled, df_bcc_upsample])

df_resampled = pd.concat([df_resampled, df_akiec_upsample])

df_resampled = pd.concat([df_resampled, df_vasc_upsample])

df = df_resampled  #Renaming dataframe

print("Printing the number of values in the dataframe")
print(df['dx'].value_counts())
print(df.shape)


#Saving the dataframe as CSV for future reference
df.to_csv('saved.csv')


temp = df['image'][1].shape
count = 0
for i in range(df.shape[0]):
    count += 1
print("Dimensions of image: {}, number: {}".format(temp, count))



#Splitting the data into trian and test
from sklearn.model_selection import train_test_split
x=df.drop(['Class'],axis=1)
y=df['Class']
xTrain, xTest, yTrain, yTest  = train_test_split(x, y, test_size=0.20, random_state=42)

#Normalization
x_train = np.asarray(xTrain['image'].tolist())
x_test = np.asarray(xTest['image'].tolist())
xTrainmean = np.mean(x_train)
xTrainstd = np.std(x_train)
xTestmean = np.mean(x_test)
xTeststd = np.std(x_test)
x_train = (x_train - xTrainmean)/xTrainstd
x_test = (x_test - xTestmean)/xTeststd


import keras
from keras.utils.np_utils import to_categorical

y_train = to_categorical(yTrain, num_classes = 7)
y_test = to_categorical(yTest, num_classes = 7)

np.save("test_data_x", x_test)
np.save("test_data_y", yTest)

#Splitting data into 90% train and 10%validation
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)
x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))
x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(75, 100, 3))


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import itertools
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

dimensions = (75, 100, 3)
Classes = 7

model = Sequential()

model.add(Conv2D(32, (3, 3),activation='relu',padding = 'Same',input_shape=dimensions))
model.add(Conv2D(32, (3, 3), activation='relu',padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.40))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(Classes, activation='softmax'))
model.summary()

#Data Augmentation
Info = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False, 
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10, 
        zoom_range = 0.1, 
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        horizontal_flip=False,
        vertical_flip=False)

Info.fit(x_train)



#Training data fitted into the model
#Using SGD_Optimizer
#Adam_Optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
SGD_optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer = SGD_optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
Lap = 75 
Volume = 10
history = model.fit_generator(Info.flow(x_train,y_train, batch_size=Volume),
                              epochs = Lap, validation_data = (x_validate,y_validate),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // Volume
                             )


loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
model.save("model_1500samples_50_keras_SGD.h5")



#Plots for the generated model

'''#1. Function to plot model's validation loss and validation accuracy
import matplotlib.pyplot as plt
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


plot_model_history(history)'''







