import tensorflow as tf 
import keras
import numpy as np 
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

class_name = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandel','Shirt','Sneaker','Bag','Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0
'''
plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(train_images[i],cmap=plt.cm.binary)
	plt.xlabel(class_name[train_labels[i]])
plt.show()
'''

model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
							keras.layers.Dense(128,activation=tf.nn.relu),
							keras.layers.Dense(10,activation=tf.nn.softmax)])
model.compile(loss = 'sparse_categorical_crossentropy',
				optimizer ='adam',
				metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs=2)

test_loss, test_accu = model.evaluate(test_images, test_labels)
print('Test accuracy:',test_accu)

prediction = model.predict(test_images)
print(prediction[0])

print(np.argmax(prediction[0]))
print(test_labels[0])

def plot_images(i,prediction_array,true_label,img):
	prediction_array, true_label, img = prediction_array[i], true_label[i], img[i]
	plt.xticks([])
	plt.yticks([])
	plt.imshow(img, cmap = plt.cm.binary)
	predicted_label = np.argmax(prediction_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel('{} {:2.0f}% ({})'.format(class_name[predicted_label],
										100*np.max(prediction_array),
										class_name[true_label],
										color = color))

def plot_value_array(i,prediction_array,true_label):
	prediction_array, true_label = prediction_array[i], true_label[i]
	plt.xticks([])
	plt.yticks([])
	thisplot = plt.bar(range(10), prediction_array, color = '#777777' )
	plt.ylim([0,1])
	predicted_label = np.argmax(prediction_array)
	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')


i = 12 
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_images(i,prediction,test_labels,test_images)
plt.subplot(1,2,2)
plot_value_array(i, prediction, test_labels) 
plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols,2*num_rows))
for i in range(num_images):
	plt.subplot(num_rows, num_cols*2, 2*i+1)
	plot_images(i,prediction,test_labels, test_images)
	plt.subplot(num_rows,num_cols*2, 2*i+2)
	plot_value_array(i,prediction,test_labels)
plt.show()

# Grab an image from the test dataset
img = test_images[0]
print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)

prediction_single = model.predict(img)
print(prediction_single)

plot_value_array(0, prediction_single, test_labels)
plt.xticks(range(10), class_name, rotation = 45)
plt.show()

print(np.argmax(prediction_single[0]))
