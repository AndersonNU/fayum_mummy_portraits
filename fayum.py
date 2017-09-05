# -*- coding: utf-8 -*-


from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import copy
import time
start_time = time.time()
######################################################################
# Cuda
# ~~~~
#
# If you have a GPU on your computer, it is preferable to run the
# algorithm on it, especially if you want to try larger networks (like
# VGG). For this, we have ``torch.cuda.is_available()`` that returns
# ``True`` if you computer has an available GPU. Then, we can use method
# ``.cuda()`` that moves allocated proccesses associated with a module
# from the CPU to the GPU. When we want to move back this module to the
# CPU (e.g. to use numpy), we use the ``.cpu()`` method. Finally,
# ``.type(dtype)`` will be use to convert a ``torch.FloatTensor`` into
# ``torch.cuda.FloatTensor`` to feed GPU processes.
#

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# dtype = torch.FloatTensor

######################################################################
# Load images
# ~~~~~~~~~~~
#
# In order to simplify the implementation, let's start by importing a
# style and a content image of the same dimentions. We then scale them to
# the desired output image size (128 or 512 in the example, depending on gpu
# availablity) and transform them into torch tensors, ready to feed
# a neural network:
#
# .. Note::
#     Here are links to download the images required to run the tutorial:
#     `picasso.jpg </_static/img/neural-style/picasso.jpg>`__ and
#     `dancing.jpg </_static/img/neural-style/dancing.jpg>`__. Download these
#     two images and add them to a directory with name ``images``


# desired size of the output image
imsize = 32 if use_cuda else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Scale(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image

from PIL import Image
import glob
image_list = []
filenames = []
# This is to obtain all the file names in the folder
for filename in glob.glob('data_fayum/faces/*.bmp'):
    # im = image_loader(filename).type(dtype)
    #im = Image.open(filename)
    filenames.append(filename)
    # image_list.append(im)

# print(image_list)
# image_list[0].save('data_fayum/output/image.bmp', 'bmp')
# print(filenames)
# style_img = []
# for image in image_list:
#     style_img.append(image_loader(image).type(dtype))


# style_img = image_list[0]

#content_img = image_loader("images/arch.jpg").type(dtype)

# assert style_img.size() == content_img.size(), \
#     "we need to import style and content images of the same size"



# We will use ``plt.imshow`` to display images. So we need to first
# reconvert them into PIL images:
#

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()



class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        # print('c is', c, ' and d is', d)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)



######################################################################
# Load the neural network
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Now, we have to import a pre-trained neural network. As in the paper, we
# are going to use a pretrained VGG network with 19 layers (VGG19).
#
# PyTorch's implementation of VGG is a module divided in two child
# ``Sequential`` modules: ``features`` (containing convolution and pooling
# layers) and ``classifier`` (containing fully connected layers). We are
# just interested by ``features``:
#

cnn = models.vgg19(pretrained=True).features

# move it to the GPU if possible:
if use_cuda:
    cnn = cnn.cuda()


######################################################################
# A ``Sequential`` module contains an ordered list of child modules. For
# instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU,
# Maxpool2d, Conv2d, ReLU...) aligned in the right order of depth. As we
# said in *Content loss* section, we wand to add our style and content
# loss modules as additive 'transparent' layers in our network, at desired
# depths. For that, we construct a new ``Sequential`` module, in wich we
# are going to add modules from ``vgg19`` and our loss modules in the
# right order:
#

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
# style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
style_layers_default = ['conv_14','conv_15','conv_16']

def get_style_model_and_matrix(cnn, style_img, style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # just in order to have an iterable access to or list of content/syle
    # losses
    #content_losses = []
    #style_losses = []
    gram_matrix = []

    model = nn.Sequential()  # the new Sequential module network
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    if use_cuda:
        model = model.cuda()
        gram = gram.cuda()

    i = 1
    # Here list is to separate a
    for layer in list(cnn):
        # print(layer)
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)
            # print(name)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                #style_loss = StyleLoss(target_feature_gram, style_weight)
                gram_matrix.append(target_feature_gram)
                ##style_losses.append(style_loss)

        # if isinstance(layer, nn.ReLU):
        #     name = "relu_" + str(i)
        #     model.add_module(name, layer)
        #
        #     if name in style_layers:
        #         # add style loss:
        #         target_feature = model(style_img).clone()
        #         target_feature_gram = gram(target_feature)
        #         #style_loss = StyleLoss(target_feature_gram, style_weight)
        #         model.add_module("style_loss_" + str(i), style_loss)
        #         style_losses.append(style_loss)

            i += 1

        # if isinstance(layer, nn.MaxPool2d):
        #     name = "pool_" + str(i)
        #     model.add_module(name, layer)  # ***

    # return model, gram_matrix
    return gram_matrix

######################################################################
# .. Note::
#    In the paper they recommend to change max pooling layers into
#    average pooling. With AlexNet, that is a small network compared to VGG19
#    used in the paper, we are not going to see any difference of quality in
#    the result. However, you can use these lines instead if you want to do
#    this substitution:
#
#    ::
#
#        # avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size,
#        #                         stride=layer.stride, padding = layer.padding)
#        # model.add_module(name,avgpool)


######################################################################
# Input image
# ~~~~~~~~~~~
#
# Again, in order to simplify the code, we take an image of the same
# dimensions than content and style images. This image can be a white
# noise, or it can also be a copy of the content-image.
#

# input_img = style_img.clone()
# if you want to use a white noise instead uncomment the below line:
# input_img = Variable(torch.randn(content_img.data.size())).type(dtype)




######################################################################
# Gradient descent
# ~~~~~~~~~~~~~~~~
#
# As Leon Gatys, the author of the algorithm, suggested
# `here <https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq>`__,
# we will use L-BFGS algorithm to run our gradient descent. Unlike
# training a network, we want to train the input image in order to
# minimise the content/style losses. We would like to simply create a
# PyTorch  L-BFGS optimizer, passing our image as the variable to optimize.
# But ``optim.LBFGS`` takes as first argument a list of PyTorch
# ``Variable`` that require gradient. Our input image is a ``Variable``
# but is not a leaf of the tree that requires computation of gradients. In
# order to show that this variable requires a gradient, a possibility is
# to construct a ``Parameter`` object from the input image. Then, we just
# give a list containing this ``Parameter`` to the optimizer's
# constructor:
#

def get_input_param_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer



######################################################################
# Finally, run the algorithm

# Here we have four inputs, the network, the, content img, style image, input image)
# input_mage is the copy of contenct image using the clone function
print('Finishing loading images....\nBegin to obtaining feature maps...')
num_images = len(image_list)
num_images = 1
vector = []
# for filename in filenames:
for i in range(len(filenames)):
# for i in range(4):

    style_img = image_loader(filenames[i]).type(dtype)
    # style_img = image_list[i]
    matrix = get_style_model_and_matrix(cnn, style_img)
    print(type(matrix))
    print('the size of matrix is:', len(matrix))
    # print('The size of the first style layer we want is', matrix[0].size())
    # For each image, vectorizing all the gram matrix and append them together
    gram_vector = torch.FloatTensor()
    gram_vector = gram_vector.cuda()
    print('the type of gram_vector is', type(gram_vector))
    for j in range(len(matrix)):
        triu_indices = torch.nonzero(matrix[j].data).transpose(0, 1)
        # gram_vector.append(matrix[j][triu_indices[0], triu_indices[1]])
        # Get the vectorized upper Gram matrix
        gram_vector_new = matrix[j][triu_indices[0], triu_indices[1]].data.view(1, -1)
        # gram_vector_new = gram_vector_new.data()
        print('the type of new gram vector is:', type(gram_vector_new))
        gram_vector = torch.cat((gram_vector, gram_vector_new), 1)
    # gram_vector = torch.cat(gram_vector, 1)
    vector.append(gram_vector)


    # triu_indices = torch.nonzero(matrix[0].data).transpose(0, 1)
    # print('the triu_indices is:', triu_indices)
    # triu_indices = matrix[0].triu().data().cpu().nonzero().transpose(0, 1)
    # vector_0 = matrix[0][triu_indices]
    # vector_0 = matrix[0][triu_indices[0], triu_indices[1]].view(1, -1)
    # print('the size of vector_0 is:', vector_0.size())
    # vector_0 = matrix[0][np.triu_indices(len(matrix[0]))]
    # vector_1 = matrix[1].numpy()[np.triu_indices(len(matrix[1]))].view(1, -1)
    # vector_2 = matrix[2].numpy()[np.triu_indices(len(matrix[2]))].view(1, -1)
    # vector.append(torch.cat((matrix[0].view(1, -1), matrix[1].view(1, -1), matrix[2].view(1, -1)), 1))
    # vector.append(torch.cat((vector_0, vector_1, vector_2), 1))



print('finishing creating feature maps...')

# matrix = get_style_model_and_matrix(cnn, style_img)

print('the type of vector[0] is', type(vector[0]))
print('the length of vector[0] is', len(vector[0]))
#
# print("Each feature map has a size of ", vector[0].size())
# new_variable = vector[0].clone()
# print(new_variable.data)
#print(vector[0].encode('utf-8'))

pdist = nn.PairwiseDistance(p=2)

# dist_1 = pdist(vector[0], vector[1])
# print('the type of two distance is ', type(dist_1))
#
# dist_2 = pdist(vector[0], vector[2])
#
# if dist_1 > dist_2:
#     print('1 is bigger')
# else:
#     print('2 is bigger')

specified_image = 0
# dist_min = 10000000000
dist = []
for i in range(50):
    dist.append(pdist(vector[specified_image], vector[i]).cpu().numpy())
    # print('finishing calculated distance', i)
    # print('the size of dist is', len(dist))
# print('the size of the dist is', len(dist[0]))
# print('dist is', dist)
# print(dist)
dist = np.asarray(dist)
dist = np.squeeze(dist)
# dist_numpy = dist.numpy()
print('converting to numpy completed...')
# print('the type of dist is', type(dist))
# print('the shape of dist is', np.squeeze(dist).shape)
index = dist.argsort()[:3]
# index = dist.argmax()
# index = np.argmin(dist_numpy)
print(index)

# image_list[index[1]].save('data_fayum/output/', 'bmp')
# print(filenames[0])
first_image = Image.open(filenames[index[0]])
# print('convert to firstimage')
first_image.save('data_fayum/output/first.bmp', 'bmp')

Image.open(filenames[index[1]]).save('data_fayum/output/second.bmp', 'bmp')
Image.open(filenames[index[2]]).save('data_fayum/output/third.bmp', 'bmp')

print("--- %s seconds ---" % (time.time() - start_time))

# plt.figure()
# imshow(output, title='Output Image')
#
# # sphinx_gallery_thumbnail_number = 4
# plt.ioff()
# plt.show()
# plt.savefig('fig.png')

