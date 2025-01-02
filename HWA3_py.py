from skimage import io
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (14.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
import warnings
warnings.simplefilter('ignore')
# make the notebook automatically reload external python modules

#%%
# load the image and confirm skimage is imported properly.
image = io.imread('small_duck.jpg')
io.imshow(image)
io.show()
print(image.shape)
#%% md
#%%
# save the dimensions of the image and reshape it for easier manipulation
rows = image.shape[0]
cols = image.shape[1]
image = image.reshape(image.shape[0]*image.shape[1],3)


#%% md
## Random centroids (15 points)
#%%
import random


def get_random_centroids(X, k):
    np.random.seed(43)
    return random.sample(list(X), k)


C = get_random_centroids(image, 5)


#%%
def minkowski(X,centroid,p):
    return (np.sum(np.abs(X-centroid)**p, axis = 1)**(1/p))
#%%
def lp_distance(X, centroids, p=2):
    distances = []
    for c in centroids:
        c_distance= (minkowski(X,c,p))
        distances.append(c_distance)
    print('dist calculated')
    return np.array(distances)



def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_features, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.
    Outpust:
    - The calculated centroids
    - The final assignment of all RGB points to the closest centroids
    """

    centroids = get_random_centroids(X,k)
    while(True):
        clusters = []
        max_iter -=1
        distances = lp_distance(X,centroids, p)
        classes = distances.argmin(axis=0)
        for i in range(k):
            k_indices = np.where(classes == i)
            clusters.append(X[k_indices])
        new_centroids = [(np.round(np.mean(np.array(c),axis=0), decimals=3)) for c in clusters]
        if np.all([np.array_equal(a, b) for a, b in zip(centroids, new_centroids)]) or max_iter == 0 :
            print(max_iter)
            return np.array(centroids), np.array(classes)
        centroids = new_centroids
#%%
centroids, classes = kmeans(image, 48, p=1, max_iter=100)
#%%
classes = classes.reshape(rows,cols)
compressed_image = np.zeros((classes.shape[0], classes.shape[1], 3), dtype=np.uint8)
for i in range(classes.shape[0]):
    for j in range(classes.shape[1]):
            compressed_image[i,j,:] = centroids[classes[i,j],:]
io.imshow(compressed_image)
io.show()
#%% md
## PART 2:  High dimensional data (20 points)
#%%
from sklearn.cluster import KMeans

# Load data
X = np.loadtxt('data.csv',delimiter=',',dtype='int')
X.shape
#%% md

###########################################################################
# TODO: START OF YOUR CODE                                                #
###########################################################################
pass
###########################################################################
#                             END OF YOUR CODE                            #
###########################################################################