import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#print(tf.__version__)

f_m=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=f_m.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape) #Training Images
print(test_images.shape)  #Test Images
#PREPROCESSING BOTH TRAINING AND TESTING IMAGES IN A SAME WAY
plt.figure()
plt.imshow(train_images[3])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(36):  #Range can always be perfect square?
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
#Training the sets
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax') #Softmax is generally used in output layer of the function
])

#Compiling the model to reduce loss 

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#Train it by fit it

model.fit(train_images, train_labels, epochs=10)



test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy of the Fashion_MNIST DataSet :', test_acc)

predictions = model.predict(test_images)
print(predictions[0])
np.argmax(predictions[0])


test_labels[0]
