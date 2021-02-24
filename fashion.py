# Machine Learning Homework 4 - Image Classification

__author__ = '**'

# General imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import os
import sys
import pandas as pd

# Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier


### Already implemented
def get_data(datafile):
	dataframe = pd.read_csv(datafile)
	dataframe = shuffle(dataframe)
	data = list(dataframe.values)
	labels, images = [], []
	for line in data:
		labels.append(line[0])
		images.append(line[1:])
	labels = np.array(labels)
	images = np.array(images).astype('float32')
	images /= 255
	return images, labels

def get_test_data(datafile):
	dataframe = pd.read_csv(datafile)
	#dataframe = shuffle(dataframe)
	data = list(dataframe.values)
	labels, images = [], []
	for line in data:
		labels.append(line[0])
		images.append(line[1:])
	labels = np.array(labels)
	images = np.array(images).astype('float32')
	images /= 255
	return images, labels

### Already implemented
def visualize_weights(trained_model, num_to_display=20, save=True, hot=True):
	layer1 = trained_model.layers[0]
	weights = layer1.get_weights()[0]

	# Feel free to change the color scheme
	colors = 'hot' if hot else 'binary'
	try:
		os.mkdir('weight_visualizations')
	except FileExistsError:
		pass
	for i in range(num_to_display):
		wi = weights[:,i].reshape(28, 28)
		plt.imshow(wi, cmap=colors, interpolation='nearest')
		if save:
			plt.savefig('./weight_visualizations/unit' + str(i) + '_weights.png')
		else:
			plt.show()


### Already implemented
def output_predictions(predictions):
	with open('predictions.txt', 'w+') as f:
		for pred in predictions:
			f.write(str(pred) + '\n')


def plot_history(history):
  train_loss_history = history.history['loss']
  val_loss_history = history.history['val_loss']
  train_acc_history = history.history['acc']
  val_acc_history = history.history['val_acc']
  # plot
  plt.plot(train_loss_history)
  plt.plot(val_loss_history)
  plt.title('Loss vs Epoch')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(['training','testing'])
  plt.show()
  plt.plot(train_acc_history)
  plt.plot(val_acc_history)
  plt.title('Accuracy vs Epoch')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend(['training','testing'])
  plt.show()

def create_mlp(args=None):
	# You can use args to pass parameter values to this method

	# Define model architecture
  model = Sequential()
  model.add(Dense(units=50, activation='relu', input_dim=28*28))
	# add more layers...
  
  #output layer
  model.add(Dense(units=10, activation='softmax'))

	# Define Optimizer
  optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

	# Compile
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model

def train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=None):
	# You can use args to pass parameter values to this method
	y_train = keras.utils.to_categorical(y_train, num_classes=10)
	model = create_mlp(args)
	history = model.fit(x_train,y_train, validation_split=0.2, epochs=10, batch_size=100)
	return model, history


def create_cnn(args=None):
  # You can use args to pass parameter values to this method
  # 28x28 images with 1 color channel
  input_shape = (28, 28, 1)
  # Define model architecture
  model = Sequential()
  model.add(Conv2D(filters=50, activation="relu", kernel_size=3, strides=2, input_shape=input_shape))
  model.add(Conv2D(filters=50, activation="relu", kernel_size=3, strides=2, input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=2, strides=2))
  # can add more layers here...
  model.add(Flatten())
  # can add more layers here...
  model.add(Dense(units=10, activation='softmax'))
  # Optimizer
  optimizer = keras.optimizers.Adam(lr=0.01)
  # Compile
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model


def train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=None):
	# You can use args to pass parameter values to this method
	x_train = x_train.reshape(-1, 28, 28, 1)
	y_train = keras.utils.to_categorical(y_train, num_classes=10)
	model = create_cnn(args)
	history = model.fit(x_train,y_train, validation_split=0.2, epochs=10, batch_size=100)
	return model, history


def train_and_select_model(train_csv):
  x_train, y_train = get_data(train_csv)
  args = {'learning_rate': 0.01}
  mlp_model, mlp_history = train_mlp(x_train, y_train, None, None, args)
  mlp_acc=mlp_history.history['val_acc']
  cnn_acc=[0]
  mlp_acc=[0]
  cnn_model, cnn_history = train_cnn(x_train, y_train, None, None, args)
  cnn_acc=cnn_history.history['val_acc']
  if mlp_acc[-1]>cnn_acc[-1]:
    best_model=mlp_model
    history=mlp_history
  else:
    best_model=cnn_model
    history=cnn_history 
  return best_model, history


if __name__ == '__main__':
  ### Switch to "development_mode = False" before you submit ###
  grading_mode = False
  if grading_mode:
    # When we grade, we'll provide the file names as command-line arguments
    if (len(sys.argv) != 3):
      print("Usage:\n\tpython3 fashion.py train_file test_file")
      exit()
    train_file, test_file = sys.argv[1], sys.argv[2]

    # train your best model
    best_model = train_and_select_model(train_file)[0]

    # use your best model to generate predictions for the test_file
    x_test,y_test=get_test_data(test_file)
    x_test=x_test.reshape(-1,28,28,1)
    predictions=best_model.predict(x_test,batch_size=100)
    output_predictions(predictions)

  # Include all of the required figures in your report. Don't generate them here.
  else:
    ### Edit the following two lines if your paths are different
    train_file = 'fashion_train.csv'
    test_file = 'fashion_test.csv'
    x_train, y_train = get_data(train_file)
    best_model, history = train_and_select_model(train_file)
    plot_history(history)
    #visualize_weights(best_model)
    x_test,y_test=get_test_data(test_file)
    x_test=x_test.reshape(-1,28,28,1)
    predictions=best_model.predict(x_test,batch_size=100)
    output_predictions(predictions)
