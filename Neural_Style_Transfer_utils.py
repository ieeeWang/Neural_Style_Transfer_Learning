# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:45:11 2020

@author: lwang
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:35:41 2020
AI_S4W4_solution
Neural Style Transfer
Art_Generation_with_Neural_Style_Transfer_v3a_solution
@author: lwang
"""
#%%
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import pprint

import imageio
import time
import glob # serach under a foler
# print(tf.__version__)

#%% functions
def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C, [m, n_H*n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [m, n_H*n_W, n_C])
    
    # compute the cost with tensorflow (≈1 line)
    J_content =  1/(4*n_H*n_W*n_C)* tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    ### END CODE HERE ###
    
    return J_content


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    ### START CODE HERE ### (≈1 line)
    GA = tf.matmul(A,tf.transpose(A))
    ### END CODE HERE ###
    
    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S, [n_H*n_W, n_C]), perm=[1,0])
    a_G = tf.transpose(tf.reshape(a_G, [n_H*n_W, n_C]), perm=[1,0])

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer =  (1/(2*n_H*n_W*n_C)**2)* tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    
    ### END CODE HERE ###
    
    return J_style_layer


def compute_style_cost(sess, model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    
    ### START CODE HERE ### (≈1 line)
    J =  alpha * J_content + beta * J_style
    ### END CODE HERE ###
    
    return J

def model_nn(sess, model, cache, train_step, input_image, num_iterations = 200, output_dir = "output_up/5L2_"):
    # output_dir = "output_up/5L2_"
    (J, J_content, J_style) = cache
    # Initialize global variables (you need to run the session on the initializer)
    # make global variables having their intial values:
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
    ### START CODE HERE ### (1 line)
    generated_image = sess.run(model["input"].assign(input_image))
    ### END CODE HERE ###
    Jt_list, Jc_list, Js_list = [],[],[]
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        ### START CODE HERE ### (1 line)
        sess.run(train_step)
        ### END CODE HERE ###
        
        # Compute the generated image by running the session on the current model['input']
        ### START CODE HERE ### (1 line)
        generated_image = sess.run(model["input"])
        ### END CODE HERE ###

        # Print every 20 iteration.
        i+=1
        if i%50 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
                        
            Jt_list.append(Jt)
            Jc_list.append(Jc)
            Js_list.append(Js)
            # save current generated image in the "/output" directory
            save_image(output_dir + str(i) + ".png", generated_image)
    
    # save last generated image
    # save_image('output/generated_image.jpg', generated_image)
        
    history_cost={} # Create a new dictionary with some data
    history_cost= {'total': Jt_list, 'content': Jc_list, 'style': Js_list}  
    
    
    return generated_image, history_cost


#%% main f
def NST_main(model, folder_in, folder_out, image_C, image_S, S_weit ='up', N = 200):    
    #% initial important hyper-param
    stye_coef = .4 # 40
    
    S_weight={} # dict
    S_weight['eq'] = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]
    S_weight['up']  = [
        ('conv1_1', 0.01),
        ('conv2_1', 0.1),
        ('conv3_1', 0.2),
        ('conv4_1', 0.3),
        ('conv5_1', 0.4)]
    S_weight['dn'] = [
        ('conv1_1', 0.4),
        ('conv2_1', 0.3),
        ('conv3_1', 0.2),
        ('conv4_1', 0.1),
        ('conv5_1', 0.01)]

    STYLE_LAYERS = S_weight[S_weit]

    # S_weit = 'eq' # 'eq', 'up', 'dn'
    content_cost_layer = 'conv4_2' # default = 'conv4_2'
    image_C_name = image_C.replace(".jpg","")
    save_dir = folder_out +"/"+ image_C_name + "_" +content_cost_layer+S_weit +"_" # "output_up/5L2_"


    #% laod images
    content_image = imageio.imread(os.path.join(folder_in,image_C))
    plt.figure()
    imshow(content_image)
    content_image = reshape_and_normalize_image(content_image)
    
    # style_image = imageio.imread(os.path.join(folder_in, image_S))
    style_image = imageio.imread( image_S)
    plt.figure()
    imshow(style_image)
    style_image = reshape_and_normalize_image(style_image)
    
    # initial generate image:
    generated_image = generate_noise_image(content_image)
    plt.figure()
    imshow(generated_image[0])
      
    
    #% initial model input
    # Start interactive session
    sess = tf.InteractiveSession()
    
    # Assign the content image to be the input of the VGG model.  
    sess.run(model['input'].assign(content_image))
    
    # Select the output tensor of layer conv4_2
    out = model[content_cost_layer]
    
    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)
    
    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out
    
    
    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)
    
    # Assign the input of the model to be the "style" image 
    sess.run(model['input'].assign(style_image))
    
    # Compute the style cost
    J_style = compute_style_cost(sess, model, STYLE_LAYERS)
    
    J =  total_cost(J_content, J_style,  alpha = 1, beta = stye_coef)
    
    cache = (J, J_content, J_style) # tuple, to be used by model_nn
    #% train model
    # define optimizer (1 line)
    optimizer = tf.train.AdamOptimizer(2.0)
    
    # define train_step (1 line)
    train_step = optimizer.minimize(J)
    
    # generate an artistic image. It should take about 3min on CPU for every 20 iterations but you start observing attractive results after ≈140 iterations
    _, history_cost=model_nn(sess, cache, train_step, generated_image, num_iterations = N, output_dir=save_dir)
    sess.close() # Otherwise, another active session may lead to OOM error when run many times
    
    # show generated_image
    Ct = history_cost['total']
    Cc = history_cost['content']
    Cs = history_cost['style']
    
    plt.figure(figsize=(10,6))
    plt.plot(Cc,'o-', linewidth=2, markersize=5, label='content')
    plt.plot(Cs,'o-', linewidth=2, markersize=5, label='style')
    plt.plot(Ct,'o--', linewidth=2, markersize=5, label='total')
    plt.yscale('symlog')
    plt.legend(loc='upper right', fontsize='x-large')
    plt.grid(True)
    plt.ylabel("Cost") 
    plt.xlabel("every 50 epochs") 
    
    plt.savefig(save_dir + 'cost.png')
    