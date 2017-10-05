from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


import numpy as np
import glob
from PIL import Image
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import copy
import time
from sklearn.cluster import KMeans
import scipy
from PIL import Image
import glob


use_cuda = torch.cuda.is_available()
# use_cuda = False
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


imsize = 512 if use_cuda else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Scale(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image



original_model = models.vgg19(pretrained=True)


class AggConv_last(nn.Module):

            def __init__(self):
                super(AggConv_last, self).__init__()
                self.features = nn.Sequential(
                    *list(original_model.features.children())
                )

            def forward(self, x):
                x = self.features(x)
                return x

# Storing image names into the list filenames
filenames = []
for filename in glob.glob('data_fayum/faces_numbered/*'):
    filenames.append(filename)


num_images = len(filenames)

# Initialize the feature matrices
feature_matrices = np.empty([num_images, 512])
# print(feature_matrices.shape)

for i in range(num_images):

    # It calls the image_loader function
    image = image_loader(filenames[i])

    # Initialize a model
    new_model = AggConv_last()

    # Extract the features and transform the data type to numpy
    features = new_model.forward(image).cpu().data.numpy()

    # Initialize a feature vector
    feature_vector = np.zeros(features.shape[1])

    print('we are at image No.', i)

    # The number of feature maps is presented by features.shape[1], which is 512
    for j in range(features.shape[1]):

        # It functions like an average pooling
        feature_vector[j] = np.mean(features[0, j])

    # feature_matrices is the stack of feature vector from each map
    feature_matrices[i, :] = feature_vector

# features = features.data()

print(feature_vector.shape)
print(feature_matrices.shape)

# Create the distance matrix
dist = np.zeros((num_images, num_images))

# double loop to calculate distance. However, this step can be better.
for i in range(num_images):

    for j in range(num_images):

        # Get teh distance matrix between every two image
        dist[i, j] = scipy.spatial.distance.euclidean(feature_matrices[i, :], feature_matrices[j, :])

# Save the distance matrix to the following file
np.savetxt('distance_matrix_2.csv', dist, delimiter=',')

index = dist.argsort()

print(index)

np.savetxt('index_2.csv', index, delimiter=',')


# print(model)
# features = model.features

# -
