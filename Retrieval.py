import cv2
import numpy as np
from keras.models import Model
from keras.datasets import mnist
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
#image ko read karna hai and convert
img = cv2.imread('image.jpg', cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#dataset ko load karke convert kia
(X_test,_),(X_train,_) = mnist.load_data()
X_test = X_test.astype('float32') / 255.
X_test = np.reshape(X_test, (-1, 28, 28, 1))
 
#print('Original Dimensions : ',img.shape)
#Image resizing so that it is comparable to the given dataset 
scale_percent = 14 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
resized=cv2.bitwise_not(resized)
#display the resized image
cv2.imshow('output', resized)
cv2.waitKey(0)
    
 
#print('Resized Dimensions : ',resized.shape)
x_train=resized
x_train = x_train.astype('float32') / 255.

x_train= np.reshape(x_train, (1, 28, 28, 1))
print(x_train.shape)

#load the trained model
autoencoder = load_model('autoencoder.h5')
print('Model Loaded')
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
query = x_train

#dimension of the query image and dataset that is changed so that
#neighbouring distance can be calculated
print(query.shape,X_test.shape)

codes = encoder.predict(X_test)
query_code = encoder.predict(query.reshape(1,28,28,1))
print(codes.shape,query_code.shape)

#how many similar images you want to print
n_neigh = 7
codes = codes.reshape(-1, 4*4*8); print(codes.shape)
query_code = query_code.reshape(1, 4*4*8); print(query_code.shape)
nbrs = NearestNeighbors(n_neighbors=n_neigh).fit(codes)
distances, indices = nbrs.kneighbors(np.array(query_code))
closest_images = X_test[indices]
closest_images = closest_images.reshape(-1,28,28,1); print(closest_images.shape)
plt.imshow(query.reshape(28,28), cmap='gray')
plt.figure(figsize=(20, 6))
for i in range(n_neigh):
    # display original
    ax = plt.subplot(1, n_neigh, i+1)
    plt.imshow(closest_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()