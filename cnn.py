from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import idx2numpy
from keras.utils import to_categorical

x_train = 'dataset/train-images.idx3-ubyte'
y_train = 'dataset/train-labels.idx1-ubyte'
x_test = 'dataset/t10k-images.idx3-ubyte'
y_test = 'dataset/t10k-labels.idx1-ubyte'

x_train=idx2numpy.convert_from_file(x_train)
x_test=idx2numpy.convert_from_file(x_test)
y_train=idx2numpy.convert_from_file(y_train)
y_test=idx2numpy.convert_from_file(y_test)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(128, kernel_size=3, activation='relu',
                  input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          epochs=3)