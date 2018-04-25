# Architecture avec 2 couches de 32 filtres et une couche cachée de 256 neurones

batch_size = 512
epochs = 20

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols =  x_train.shape
num_test, _, _, _ =  x_test.shape
num_classes = len(np.unique(y_train))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model5 = Sequential()

# Conv1 32 32 (3) => 30 30 (32)
model5.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model5.add(Activation('relu'))
# Conv2 15 15 (32) => 28 28 (64)
model5.add(Conv2D(32, (3, 3), padding='same'))
model5.add(Activation('relu'))
# Pool1 28 28 (64) => 14 14 (64)
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(Dropout(0.25))

model5.add(Flatten())
model5.add(Dense(256))
model5.add(Activation('relu'))
model5.add(Dropout(0.5))
model5.add(Dense(num_classes))
model5.add(Activation('softmax'))

model5.summary()

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model5.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

start5 = time.time()

history5 = model5.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True)

end5 = time.time()

# Score trained model.
scores = model5.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1]*100)

#############################################################################################################################
# Architecture avec 2 couches de 64 filtres et une couche cachée de 256 neurones

model6 = Sequential()

# Conv1 32 32 (3) => 30 30 (32)
model6.add(Conv2D(64, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model6.add(Activation('relu'))
model6.add(Conv2D(64, (3, 3), padding='same'))
model6.add(Activation('relu'))
model6.add(MaxPooling2D(pool_size=(2, 2)))
model6.add(Dropout(0.25))

model6.add(Flatten())
model6.add(Dense(256))
model6.add(Activation('relu'))
model6.add(Dropout(0.5))
model6.add(Dense(num_classes))
model6.add(Activation('softmax'))

model6.summary()

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
history6 = model6.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

start6 = time.time()

history6 = model6.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True)

end6 = time.time()

# Score trained model.
scores = model6.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#############################################################################################################################
# Visualisation des metrics

plot_model_history(history5)
print("Le modéle avec une architecture à 2 couches convolutionnelles de 32 filtres avec 256 neurones fait %0.2f secondes de calcul"%(end5 - start5))
print("Accuracy on test data is: %0.2f"%accuracy(x_test, y_test, model5))

plot_model_history(history6)
print("Le modéle avec une architecture à 2 couches convolutionnelles de 64 filtres avec 256 neurones fait %0.2f secondes de calcul"%(end6 - start6))
print("Accuracy on test data is: %0.2f"%accuracy(x_test, y_test, model6))
