# Use GPU for Theano, comment to use CPU instead of GPU
# Tensorflow uses GPU by default
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

# If using tensorflow, set image dimensions order
from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")
    
import time
import keras
import matplotlib.pyplot as plt
import numpy as np
% matplotlib inline
np.random.seed(2017) 
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.optimizers import RMSprop,SGD, Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


###########################################################################################################################

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols =  x_train.shape
num_test, _, _, _ =  x_test.shape
num_classes = len(np.unique(y_train))
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_test.shape[1], 'images channels')
print(num_classes, 'classes')

###########################################################################################################################

## Visualisation des features avec leurs etiquettes
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(y_train[:]==i)[0]
    features_idx = x_train[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    im = np.transpose(features_idx[img_num,::], (1, 2, 0))
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()

###########################################################################################################################
# Initialisations, redimensionnement de la base et la conversion de la matrice en matrice binaire

batch_size = 512
epochs = 200

dimData=np.prod(x_train.shape[1:])
x_train = x_train.reshape(50000,dimData)
x_test = x_test.reshape(10000,dimData)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

############################################################################################################################
# Entrainement d'un reseau de neurones en variant le nombre de couches
#   Architecture avec 2 couches (2 hidens layers)
#   Nombre de noeuds par couche : 128
#   Optimizer = rmsprop

model = Sequential()
model.add(Dense(128, activation='relu', init ='uniform', input_shape=(3072,)))
model.add(Dropout(0.1))
model.add(Dense(128, activation='sigmoid', init ='uniform'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))

model.summary()

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

start = time.time()

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

end = time.time()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)

############################################################################################################################
#   Architecture avec 3 couches (3 hidens layers)
#   Nombre de noeuds par couche : 128
#   Optimizer = rmsprop

model1 = Sequential()
model1.add(Dense(128, activation='relu', init ='uniform', input_shape=(3072,)))
model1.add(Dropout(0.1))
model1.add(Dense(128, activation='sigmoid', init ='uniform'))
model1.add(Dropout(0.1))
model1.add(Dense(128, activation='relu', init ='uniform'))
model1.add(Dropout(0.1))
model1.add(Dense(10, activation='softmax'))

model1.summary()

opt = keras.optimizers.adam(lr=0.0001, decay=1e-6)
model1.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

start1 = time.time()

history1 = model1.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

end1 = time.time()

score = model1.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)

############################################################################################################################
# Visualisation des courbes

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
def accuracy(b, y_test, model):
    result = model.predict(b)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(y_test, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)
    
plot_model_history(history)
print("Le modéle avec une architecture à 2 couches et avec 128 neurones par couche fait %0.2f secondes de calcul"%(end - start))
print("Accuracy on test data is: %0.2f"%accuracy(x_test, y_test, model))

plot_model_history(history1)
print("Le modéle avec une architecture à 3 couches et avec 128 neurones par couche fait %0.2f secondes de calcul"%(end1 - start1))
print("Accuracy on test data is: %0.2f"%accuracy(x_test, y_test, model1))

#############################################################################################################################
