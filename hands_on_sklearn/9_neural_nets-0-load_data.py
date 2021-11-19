from keras.datasets import mnist
import matplotlib.pyplot as plt

""" ---------------------------------------"""
"""          Neural networks 1             """
""" ---------------------------------------"""
"""https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/"""


# Plot ad hoc mnist instances

(X_train, y_train), (X_test, y_test) = mnist.load_data()   # Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

