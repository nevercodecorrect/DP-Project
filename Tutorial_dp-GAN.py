#!/usr/bin/env python
# coding: utf-8

# # How to  generate new data using Generative Adversarial Networks
# # in a differential private mode 
# 
#    
# #   
# #### ***by Lyudmyla Dymytrova, Lorenzo Frigerio, and Anderson Santana de Oliveira***
# August 2019

# ###    
# ## Introduction
# 
#    This tutorial shows how to generate a new (fake) dataset using Generative Adversarial Network as reported in  Frigerio L., Santana de Oliveira A., Gomez L. & Duverger P. (2019) Differentially private GANs for time series, continuous and discrete Open Data) (http://arxiv.org/abs/1901.02477). With the exception that we use out of the box, AutoML classifier for checking the quality of the generated data. 

# In[1]:


# Load modules
import os, sys


import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import seaborn as sns
import csv
import time
import pickle 

import tflib as lib
import tflib.ops.linear
import tflib.plot
import tflib.inception_score

from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
from differential_privacy.dp_sgd.dp_optimizer import sanitizer
from differential_privacy.privacy_accountant.tf import accountant

from discretedata_manager import discretedata_manager
from models import Generator, Discriminator

from scipy.stats import wasserstein_distance, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score


# In[ ]:


# Initialize H2O.api for AutoML 
import h2o
from h2o.automl import H2OAutoML
h2o.init()


# In[ ]:


# Load dataset for training
adult = pd.read_csv("./dataset/adult.csv",delimiter=',',
                    skipinitialspace = True)
adult.head()


# ### Data preprocessing
# 
# The first step is a preparation of data to a specific format for training WGAN. The input data set must be in a format of Pandas DataFrame. Data can be labelled or unlabeled. 
# - **Missing values** is an issue only for classifier and do not impact the WGAN itself, so the next step might be skipped for unlabeled data. If dataset contains missing values, by default columns with more than 50% of NaN will be removed. The number of permitted missing values in a column can be reduced by user (an argument na_percent). If the number of permitted missing values in a column is less than a given threshold, each missing value will be replaced by a string that combines ‘missing’ and the column name where this missing value occurs. This procedure allows to distinguish between NaN values that occur in different columns. 
# - So far only **categorical** features are implemented (the algorithm automatically removes all numerical features except a target column). If data labelled, a target column must be binary due to H2O AutoML does not have implemented multilabel classification yet??? A **target column** can be categorical (with two levels) or numerical. 
# - The algorithm performs a random subsampling of majority class if data are **imbalanced**. Within the preprocessing step a function ‘prepare_data’ performs a random shuffling and **splitting data** into test (33% of data) and train (67%) datasets preserving a class balance for the target column in both datasets.  
# - There is no train/test split for an **unlabelled** data. The entire input dataset is used for WGAN training.
# - **The output** contains 4 instances, i.e. train_data (data frame), test_data (data frame), variables (list of strings) and target (string) that are used at next steps to train WGAN.
# 
# ```adult = pd.read_csv("./dataset/adult.csv", delimiter=',',  skipinitialspace = True)
# prepData = prepare_data(adult, target='output')
# prepData.test_data.shape
# prepData.target
# prepData.variables```

# In[ ]:


# Data preprocessing with train/test splitting 

class prepare_data(object):
    def __init__(self, data, target = None, na_percent = 0.5):
        self.data = data
        self.target = target
        self.na_percent = na_percent
        self.train_data, self.test_data, self.variables  = self.prepare_discrete_data()
 
    def prepare_discrete_data(self):
        TextData = self.data.select_dtypes(include = ['object']).copy()
        discrete_cols = TextData.columns
        NewData = TextData[discrete_cols]
        # Check if there are NaN values 
        if NewData.isnull().values.any() == True:
            # Remove columns with more than xx% of NaN (xx% is user-defined or 0.5 by default)
            NewData.dropna(thresh = NewData.shape[0]*(1-self.na_percent), how ='all', axis = 1, inplace = True)
            # In rest of columns replace NaN by a string 'missing_column_name'
            for col, name in enumerate(NewData.columns):
                values = {name: str("missing "+ NewData.columns[col])}
                NewData.fillna(value = values, inplace = True)
        # If data is unlabelled, no preprocessing
        if self.target == None:
            train_data = NewData
            test_data = pd.DataFrame()
            print('WARNING: no test data for unlabelled dataset')
            return train_data, test_data, discrete_cols
        else:
            # If data is labelled, check a format of the target variable
            if self.data[self.target].dtype != object:
                NewData = pd.concat((TextData, self.data[self.target]), axis = 1)
                discrete_cols = NewData.columns
            pos = NewData.loc[NewData[self.target] == NewData[self.target].unique()[0]]
            neg = NewData.loc[NewData[self.target] == NewData[self.target].unique()[1]]
            # Check if classes are balanced
            if len(pos) == len(neg):
                train_data, test_data = train_test_split(NewData, test_size = 0.33, random_state = 1, shuffle = True,
                                                             stratify = NewData[self.target])
                return train_data, test_data, discrete_cols
            # If classes are unbalanced, randomly subsample majority class
            elif len(pos) < len(neg):
                neg = neg.sample(len(pos))
                data = pd.concat((pos,neg),axis=0)
                train_data, test_data = train_test_split(data, test_size = 0.33, random_state = 1, shuffle = True,
                                                             stratify = data[self.target])
                return train_data, test_data, discrete_cols
            else:
                pos = pos.sample(len(neg))
                data = pd.concat((pos,neg),axis=0)
                train_data, test_data = train_test_split(data, test_size = 0.33, random_state = 1, shuffle = True,
                                                             stratify = data[self.target])
                return train_data, test_data, discrete_cols


# In[ ]:


# ADULT DATASET:  Data preprocessing
adult_pr = prepare_data(adult, target='output')
adult_pr.test_data.shape


# In[ ]:


adult_pr.train_data.output.value_counts()


# ### Tuning WGAN hyperparameters
# 
# The next important step is a choice of hyperparameters for training WGAN. This can be done at one step using a function ‘WGAN_params’. 
# - **privacy**: boolean, True or False. The user can choose to train WGAN in a differential private or public mode. 
# - **batch_size**: the number of samples n that are processed by WGAN at once. The generator generates n fake samples (batch) that are further compared by discriminator to a randomly sampled n samples (batch) of real training data. After processing a batch, generator parameters are updating, and a new set of n samples (batch) are generated. 
# - **complexity**: the shape of output arrays produced by each layers of Neural Network models in Generator and Discriminator. The exception are the last layers in both NN models that produce one-dimensional array in case of Discriminator and m-dimensional array for Generator with m equal to the number of rows in train data. This parameter defines the complexity of Neural Network models.
# - **mode**: WGAN algorithm can be performed in a standard way (‘wgan’) or using a gradient penalty clipping (‘wgan-gp’) introduced by Gulrajani et al. (2017).
# - **gradient_coeff**: a gradient penalty coefficient that was experimentally tested by Gulrajani et al. (2017) on a variety of NN architectures and datasets. The default value is 10. According to Frigerio L. smaller values of a gradient penalty coefficient increase a speed of training for small datasets but can be compensated by increasing the number of disc_iters instead.
# - **norm_bound**: L2 norm that is used to clip a gradient of Discriminator. For more details see Abadi et al. (2016).
# - **learning_rate**: learning rate of the differential private Stochastic Gradient Descent optimizer (dp-SGD) for Discriminator. The default value is 0.001. The learning rate of the Adam optimizer for Generator is not a parameter (but can be implemented if necessary) and by default equal to 0.0001.
# - **noise**: a noise level that corresponds to a variance of the gaussian noise added to the gradient (dp-SGD) of Discriminator optimizer to achieve differential privacy. This parameter has a direct impact on privacy budget: the higher a noise level, the smaller value of epsilon.
# - **gen_iters**: the number of Generator iterations to produce one batch of fake samples. Critic_gen in Lorenzo’s version.
# - **disc_iters**: the number of Discriminator iterations performed by one iteration of Generator and using only one batch of samples.
# - **target_delta**: differential privacy parameter that defines the probability at which a certain level of privacy (epsilon) is correct. At each generator run the algorithm returns epsilon value. If target_delta is None, it is calculated as following:  1/(number of rows in train data ). The user-defined target_epsilon can be easily implemented if necessary.
# - **stop_criterion**: the differential privacy budget (epsilon, by default 32). In a differential private mode of WGAN the training is stopped as far as the epsilon value of generated data is higher than a given stopping criterion. If the stopping criterion is assigned to +Inf or WGAN is trained in a public mode, the algorithm will stop after 1000 iterations (epochs). 
# - **interval**: a period with which the WGAN outputs and accuracies of Classifiers (scores and plots) are saved.
# 
# Use print(my_params.__dict__) to see the default values: 
# 
# ```my_params = WGAN_params(noise = 3.0, target_delta = 0.00001)
# print(my_params.__dict__)```

# In[ ]:


class WGAN_params(object):
    def __init__(self, batch_size = 256, complexity = 512, mode = 'wgan-gp',
                 gradient_coeff = 10, norm_bound = 0.2, learning_rate = 1e-3, noise = 7.0,
                 gen_iters = 1, disc_iters = 5, target_delta = None, stop_criterion = 32, interval = 20):
        self.batch_size = batch_size
        self.complexity = complexity
        self.mode = mode
        self.gradient_coeff = gradient_coeff
        self.norm_bound = norm_bound
        self.learning_rate = learning_rate
        self.noise = noise
        self.gen_iters = gen_iters
        self.disc_iters = disc_iters
        self.target_delta = target_delta
        self.stop_criterion = stop_criterion
        self.interval = interval


# In[ ]:


# Set WGAN hyperparameters
my_params = WGAN_params(interval = 2)
print(my_params.__dict__)


# In[ ]:


# Save Cramer_V statistic for association between the same categorical features in generated (fake) vs. input (real) data
# A score for each epoch

def save_CramerV(list, epoch):
    path = './correlation'
    if os.path.isdir (path) == False:
        os.makedirs(path)
    else:
        pass
    with open(path + "/CramerV_"+ str(epoch)+'.txt','w') as thefile:
        for item in list:
            thefile.write("%s\n" % item)
            
# Save the matrix of correlations between different features
# Two correlation matrices side by side are returned, i.e. for real and generated data
# A matrix for each epoch
# WARNING: data should not contain missing values

def saveCorrelationMatrix(real, fake, epoch):
    path = './correlation'
    if os.path.isdir (path) == False:
        os.makedirs(path)
    else:
        pass
    real = pd.get_dummies(real).corr(method = 'pearson')
    fake = pd.get_dummies(fake).corr(method = 'pearson')
    fig, axs = plt.subplots(figsize = (15,7), ncols=2)
    sns.heatmap(real, yticklabels = False, ax = axs[0])
    axs[0].set_title('Real data')
    sns.heatmap(fake, yticklabels = False, ax = axs[1])
    axs[1].set_title('Generated data')
    fig.savefig(path +"/CorrelationMatrix_"+ str(epoch) +".png")
    plt.close(fig)

def getOccurencies(discManager,data):
    lengths = discManager.getLabelsLength()
    totIndex = 0
    columnsValues = discManager.OHdataset.columns.tolist()
    Freq = []
    Name = []
    for index, length in enumerate(lengths):
        Freq_per_Category = []
        Name_Category = []
        if(length > 1):
            samplesDisc = np.argmax(data[:, totIndex:totIndex + length], axis=1)
            for i in range(length):
                Freq_per_Category.append((samplesDisc == i).sum())
                Name_Category.append(columnsValues[i + totIndex])
        Freq += Freq_per_Category
        Name += Name_Category
        totIndex += length
    FreqData = pd.concat([pd.Series(Name), pd.Series(Freq)], axis = 1)
    FreqData.columns = ['Category','Occurrency']
    return FreqData


# A function ‘train_WGAN’ requires only two arguments: preprocessed data and a dictionary with hyperparameters of WGAN that are set at the previous step. The algorithm will create automatically a directory (using a given path) where all outputs with a given interval (by default every 20 iterations) will be stored.
# 
# ```train_WGAN(pat, my_params, './dp-WGAN-discrete_patient_occurencies')```
# 
# 
# ![alt text](./figs/image003.png "Title")
# 
# #### Accuracy of AutoML classifier for labeled data
# 
# If data labelled, AutoML Classifier (H2O.ai) is used to evaluate the quality of differential private data generated by WGAN. There is no implementation for unlabeled data yet. The quality of differential private data generated by WGAN is evaluated following Shmelkov et al. (2018): An important characteristic of GAN model is that generated data should not only be realistic, but also recognizable as coming from a given class. An optimal GAN that perfectly captures the target distribution can generate a new set of data (x_fake), which are indistinguishable from the original real training set (x_train). Assuming both these sets have the same size, a classiﬁer trained on either of them should produce roughly the same validation accuracy.
# A figure is based on Shmelkov et al. (2018):
# 
# ![alt text](./figs/image006.jpg "Title")
# 
# - **acc_train** is the accuracy of a classiﬁer trained on fake data (x_fake) and tested on a real test data (x_test). When a GAN is not perfect, acc_train will be lower than acc_valid. When acc_train is close to validation accuracy, it means that generated fake data are high quality and as diverse as the training set. acc_train is a measure of the diversity of a GAN, similar to recall, measuring the fraction of real data covered by a GAN.
# - **acc_test** is the accuracy of a classiﬁer trained on the original train data but tested on fake (generated) data. Ideally, acc_test should be close to the validation accuracy. If it signiﬁcantly higher, it means that the GAN overﬁts, and simply memorizes the training set. On the contrary, if it is signiﬁcantly lower, the GAN does not capture the target distribution well and the data quality is poor. Note that this measure does not capture the diversity of samples. acc_test accuracy is related to the precision score, quantifying how close generated data are to real data.
# - **acc_valid** is the typical validation accuracy of the classiﬁer trained on real train data (x_train) and validated on real test data (x_test).
# 
# References:
# 1.	Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. 2016. Deep Learning with Differential Privacy. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS '16). ACM, New York, NY, USA, 308-318. DOI: https://doi.org/10.1145/2976749.2978318.
# 2.	Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. 2017. Improved training of wasserstein GANs. In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17), Ulrike von Luxburg, Isabelle Guyon, Samy Bengio, Hanna Wallach, and Rob Fergus (Eds.). Curran Associates Inc., USA, 5769-5779.
# 3.	Konstantin Shmelkov, Cordelia Schmid, and Karteek Alahari. 2018. How good is my GAN? In Proceedings of European Conference on Computer Vision.

# In[3]:


# input_data is an output of prepare_data, params is paramters for WGAN training, 
# path is a string to distinguish different datasets (will be used to create a directiry name)

def train_WGAN(input_data, params, path):
    
    # To show the execution time of WGAN
    start_time = time.time()
    
    # Define number of the training iteration 
    epoch = 1000
    
    # Define training data
    data = input_data.train_data
    # One hot encoder of the input dataset
    discManager = discretedata_manager(data)
    
    # Create a directory to store results
    path = path
    if os.path.isdir(path) == False:
        os.makedirs(path)  
    else:
        pass
    print('Outputs will be saved to:'+ str(path))
    
    # Define target delta if it is not given
    if params.target_delta is None:
        target_delta = 1/data.shape[0]
    else: target_delta = params.target_delta
    
    # Build the improved WGAN architecture [Gulrajani et al. 2017]
    diminput = discManager.getLabelLengthTot()
    real_data = tf.placeholder(tf.float32, shape = [None, diminput])
    fake_data = Generator(params.batch_size, real_data, discManager, diminput, params.complexity) 
    disc_real = Discriminator(real_data, diminput, params.complexity)
    disc_fake = Discriminator(fake_data, diminput, params.complexity)
    
    # WGAN loss functions for Discriminator and Generator
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real) 
    gen_cost = -tf.reduce_mean(disc_fake)
    
    # WGAN gradient penalty
    if params.mode == 'wgan-gp':
        alpha = tf.random_uniform(shape = [params.batch_size,1], minval=0.,maxval=1.) 
        interpolates = alpha*real_data + ((1-alpha)*fake_data)
        disc_interpolates = Discriminator(interpolates, diminput, params.complexity)
        gradients = tf.gradients(disc_interpolates, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1)**2)
        disc_cost += params.gradient_coeff*gradient_penalty

    disc_params = lib.params_with_name('Discriminator')
    gen_params = lib.params_with_name('Generator')

    # Hyperparameters of the differential privacy for WGAN
    if params.mode == 'wgan-gp':
        eps = tf.placeholder(tf.float32)
        delta = tf.placeholder(tf.float32)
        
        priv_accountant = accountant.GaussianMomentsAccountant(len(input_data.train_data))
    
        clip = True
        iteration_decay = tf.placeholder(tf.float32)
        l2norm_bound = params.norm_bound  
        batches_per_lot = 1
        gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(priv_accountant, 
                                                                  [tf.multiply(l2norm_bound,iteration_decay), clip], 
                                                                  disc_params)
        iterationW = tf.placeholder(tf.float32)
        tf.summary.scalar("iter", iterationW)
    
        # Optimizer of the Discriminator (dp-SGD) using differential privacy 
        disc_train_op, sanitized_grads, clipped_grads, not_sanitized_grads, bound, a  = dp_optimizer.DPGradientDescentOptimizer(
            params.learning_rate, 
            [eps, delta], 
            sanitizer = gaussian_sanitizer, 
            sigma = tf.multiply(params.noise,iteration_decay),
            batches_per_lot = batches_per_lot).minimize(disc_cost, iteration = iterationW, var_list = disc_params)
    
#         # Optimizer of the Discriminatort (SGD) without differential privacy
#         disc_train_opSGD = tf.train.GradientDescentOptimizer(params.learning_rate).minimize(disc_cost,var_list=disc_params)
        
#         # Optimizer of the plain GAN with ADAM
#         disc_train_opPublic = tf.train.AdamOptimizer(params.learning_rate, beta1=0.5, beta2=0.9).minimize(disc_cost,
#                                                                                                           var_list=disc_params)
    
        # Optimizer of the Generator (ADAM)
        if len(gen_params) > 0:
            gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1 = 0.5, beta2 = 0.9).minimize(gen_cost,
                                                                                                         var_list = gen_params)
        else:
            gen_train_op = tf.no_op()
            
    # Print parameters of the generator and the discriminator
    print ("Generator params:")
    for var in lib.params_with_name('Generator'):
        print ("\t{}\t{}".format(var.name, var.get_shape()))
    print ("Discriminator params:")
    for var in lib.params_with_name('Discriminator'):
        print ("\t{}\t{}".format(var.name, var.get_shape()))
    
    # Generate new (fake) data
    # Calculate occurrencies of categoies for each feature in a generated data
    # Compute the Wasserstein distance on the marginal distibution of the features
    
    def generate_samples(true_dist, discManager, epoch):
        samples = np.concatenate([session.run(fake_data) for _ in range(int(true_dist.shape[0] / int(fake_data.shape[0])))])
        discreteSamples, discreteTextDataset = discManager.convertDiscreteMatrix(samples)
        FreqTable = getOccurencies(discManager, discreteSamples)
        was = 0
        for i in range(samples.shape[1]):
            was += wasserstein_distance(samples[:, i], true_dist[:, i])
        was /= samples.shape[1]
        print("Wasserstein distance: " + str(round(was, 4)))
        return discreteTextDataset, FreqTable, was

    # Compute and display privacy values (delta, epsilon) achieved by dp-WGAN
    def plotPrivacyMeasures(session):
        spent_deltas_eps = priv_accountant.get_privacy_spent(session, target_deltas = [target_delta])
        for (spent_eps, spent_delta) in spent_deltas_eps:
            print("epsilon = " + str(spent_eps) + ", delta = " + str(spent_delta))
            return spent_eps
    # Save the plot of the Wasserstein distances wrt the time
    # A single plot for each epoch
    # Progressing over time (number of epochs on a given plot is constantly updating during the training)

    def save_plot_was(was):
        dirname = os.path.join(path,'plots').replace("\\","/")
        if os.path.isdir (dirname) == False:
            os.makedirs(dirname)
        else:
            pass
        X = list(range(0, len(was)*params.interval, params.interval))
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(X, was)
        ax.set_xlabel("Number of WGAN iterations")
        ax.set_ylabel("Wasserstein distance")
        fig.savefig(os.path.join(dirname,"wasserstein_plot.png").replace("\\","/"))
        plt.close(fig)
        
    # Save the plot to compare the accuracies of AutoML Classifiers 
    # Generated (fake) data is used to train Classifier
    # Predicted values are compared to real data: training data that where used to train WGAN and test data that were left apart
    # A single plot for each epoch
    # Progressing over time (number of epochs on a given plot is constantly updating during the training)

    def save_plot_accuracy(acc_test, acc_train, acc_valid):
        dirname = os.path.join(path,'plots').replace("\\","/")
        if os.path.isdir(dirname) == False:
            os.makedirs(dirname)
        else:
            pass
        X = list(range(0, len(acc_test)*params.interval, params.interval))
        fig, ax = plt.subplots(nrows = 1, ncols = 1 )
        ax.plot(X, acc_test, label = "Test")
        ax.plot(X, acc_train, label = "Train")
        ax.plot(X, acc_valid, label = "Validation")
        ax.set_xlabel("Number of WGAN iterations")
        ax.set_ylabel("Accuracy")
        ax.legend(loc = 'lower right')
        fig.savefig(os.path.join(dirname, "acc_plot.png").replace("\\","/"))
        plt.close(fig)

    # Save accuracy's scores of Random Forest Classifier in a text file
    # All scores for different epochs are stored to a single file

    def save_accuracy(list, name_file):
        dirname = os.path.join(path,'scores').replace("\\","/")
        if os.path.isdir (dirname) == False:
            os.makedirs(dirname)
        else:
            pass
        with open(os.path.join(dirname, str(name_file)+'.txt').replace("\\","/"),'w') as thefile:
            for item in list:
                thefile.write("%s\n" % item)
            
#     # Save WGAN model
#     saver = tf.train.Saver()
    
    # Train the WGAN model
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # Get the one-hot encoding of the input data
        data = discManager.OHdataset.values
        # Get category's occurrencies in original dataset
        OriginFreqData = getOccurencies(discManager, data)  
        OriginFreqData.to_csv(os.path.join(path, "OriginalFreq.csv").replace("\\","/"), index = False)
        # Shuffle the input data
        np.random.shuffle(data)                     
        accuracies_test = []
        accuracies_train = []
        accuracies_valid = []
        dp_epsilon = []
        wasTot = []
        best_was = np.inf
        _old_gen_cost = 0
        _gen_cost = 0
        count_change = 0
        stop_criterion = 0
        # Do n epochs 
        for iteration in range(epoch):
            # Early Stopping of WGAN training when a privacy budget is exhausted:
            if stop_criterion < params.stop_criterion:
                print("\nEpoch "+str(iteration+1))
                # Train the Critic
                for idx in range(int(data.shape[0] / (params.batch_size * params.disc_iters))):
                    # print("Batch "+str(idx+1))
                    # Train the Generator
                    if iteration > 0:
                        for _ in range(params.gen_iters):
                            _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={iterationW: (float(iteration + 1))})
                    for i in range(params.disc_iters):
                        # print("Discriminator iteration "+str(i+1))

                        # Clipping decay
                        curr_iter =  (iteration % 5)
                        iter_decay = 1 - int(iteration / 10)*0.1
                        if (iter_decay <= 0.40):
                            iter_decay = 0.35

                        # Train the Discriminator
                        indeces = np.random.randint(0, data.shape[0] - 1, params.batch_size)
                        _disc_cost, _ = session.run([disc_cost, disc_train_op],
                                                    feed_dict = {real_data: data[indeces, :], iterationW: (float(curr_iter)),
                                                    iteration_decay: iter_decay})  
                # Shuffle the data after each training to change the batches
                np.random.shuffle(data)
                print("Discriminator loss: " + str(round(_disc_cost, 4)))
                # Additional dp-GAN imrpovement to save privacy every 20 iterations
                if iteration % params.interval == 0:
                    if (abs(_gen_cost) < abs(_old_gen_cost)):
                        count_change += 1
                        if (count_change > 1 and params.disc_iters > 1):
                            params.disc_iters -= 1
                            count_change = 0
                        print("change")
                    _old_gen_cost = _gen_cost
                    # Compute, display and save differential privacy (epsilon, delta) parameters
                    stop_criterion = plotPrivacyMeasures(session)
                    dp_epsilon.append(stop_criterion)
                    save_accuracy(dp_epsilon, 'epsilon')
                    # Generate a new (fake) data
                    # Get category's occurrencies in generated dataset
                    # Compute and dipslay Wasserstein distance
                    output, FakeFreqData, was = generate_samples(data, discManager, iteration)
                    # Save generated data to a text file
                    anonymized_data = pd.DataFrame(np.array(output).T, columns=discManager.variables)
                    # Convert back missing values to NA for generated data
                    output_na = anonymized_data
                    for col, name in enumerate(output_na.columns):
                        output_na.replace(str("missing "+ output_na.columns[col]), np.nan, inplace=True)
                    output_na.to_csv(os.path.join(path, "GeneratedData_epoch_"+str(iteration+1)+".csv").replace("\\","/"), 
                                     index = False)
                    # Save category's occurencies for each feature in a generated dataset
                    FakeFreqData.to_csv(os.path.join(path, "GeneratedFreq_epoch_"+ str(iteration+1)+".csv").replace("\\","/"), 
                                        index = False)
                    # Save Wasserstein distance, i.e. the plot and values
                    wasTot.append(was)
                    save_accuracy(wasTot, 'wasserstein_distance')
                    save_plot_was(wasTot)

                    # If data labelled, calculate the accuracy scores of Classifier using H2O AutoML
                    # Train and test accuracy of WGAN are calculated following Shmelkov et al. (2018) How good is my GAN?
                    if input_data.target is not None:
                        print('\nData is labelled \nCalculating accuracy scores for AutoML Classifiers ......')
                        y = input_data.target
                        # Prepare fake (generated) data
                        h2o_fake = h2o.H2OFrame(anonymized_data)
                        h2o_fake[y] = h2o_fake[y].asfactor()
                        x_fake = h2o_fake.columns
                        x_fake.remove(y)
                        # Prepaer real train data
                        h2o_train = h2o.H2OFrame(input_data.train_data)
                        h2o_train[y] = h2o_train[y].asfactor()
                        x_train = h2o_train.columns
                        x_train.remove(y)
                        # Prepare real test data
                        h2o_test = h2o.H2OFrame(input_data.test_data)
                        h2o_test[y] = h2o_test[y].asfactor()
                        # Train WGAN-train-classifier
                        aml_train = H2OAutoML(max_models = 10, seed = 1)
                        aml_train.train(x = x_fake, y = y, training_frame = h2o_fake)
                        # Train WGAN-test-classifier
                        aml_test = H2OAutoML(max_models = 10, seed = 1)
                        aml_test.train(x = x_train, y = y, training_frame = h2o_train)
                        # Calculate a typical validation accuracy
                        preds_valid = aml_test.leader.predict(h2o_test)                    
                        acc_valid = accuracy_score(h2o_test[y].as_data_frame().values, 
                                                   preds_valid['predict'].as_data_frame().values)
                        # Calculate WGAN-train accuracy
                        preds_train = aml_train.leader.predict(h2o_test)                    
                        acc_train = accuracy_score(h2o_test[y].as_data_frame().values, 
                                                   preds_train['predict'].as_data_frame().values)
                        # Calculate WGAN-test accuracy
                        preds_test = aml_test.leader.predict(h2o_fake)
                        acc_test = accuracy_score(h2o_fake[y].as_data_frame().values, 
                                                   preds_test['predict'].as_data_frame().values)
                        # Save both AutoMl Classifiers as MOJO models
                        aml_test.leader.download_mojo(path = path +"/AutoML/test_model_"+str(iteration+1)+".zip")
                        aml_train.leader.download_mojo(path = path +"/AutoML/train_model_"+str(iteration+1)+".zip")
                        # Print the results
                        print('Validation accuracy: '+ str(acc_valid))
                        print('Train accuracy: '+ str(acc_train))
                        print('Test accuracy: '+ str(acc_test))
                        accuracies_test.append(acc_test)
                        accuracies_train.append(acc_train)
                        accuracies_valid.append(acc_valid)
                        # Store the results of the classification analysis based on generated data
                        save_accuracy(accuracies_test,'accuracy_test')
                        save_accuracy(accuracies_train, 'accuracy_train')
                        save_accuracy(accuracies_valid, 'accuracy_valid')
                        save_plot_accuracy(accuracies_test, accuracies_train, accuracies_valid)
    # Print info why training is stopped
    if stop_criterion > params.stop_criterion:
        print("\nPrivacy budget is exhausted ...")
        print("WGAN training is stopped ...")
    else: print("WGAN training is stopped after" +str(epoch) +" iterations ...")
    # Return the execution time of WGAN
    execution_time = time.time() - start_time
    print("\nThe execution time: "+str(round(execution_time/60,))+" min "+str(round(execution_time%60,2))+" sec")
    return stop_criterion
                    


# In[4]:


train_WGAN(adult_pr, my_params, './adult_test_stop_criterion')


# #### Plot a side-by-side histogram of category's occurrencies in original and generated datasets

# In[ ]:


original = pd.read_csv('./adult_test_stop_criterion/OriginalFreq.csv')
fake_311 = pd.read_csv('./adult_test_stop_criterion/GeneratedFreq_epoch_311.csv')



# In[ ]:


df = original.set_index('Category').join(fake_311.set_index('Category'), 
                                    lsuffix = '_Original',rsuffix = '_Generated')
df.head()


# In[ ]:


# Plot a side-by-side histogram of category's occurrencies in original and generated datasets
ax = df.plot(kind ='bar', figsize =(30,7))
fig = ax.get_figure()
fig.savefig("./FreqCompare_epoch_87.png", bbox_inches = 'tight')
plt.close(fig)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

df.plot(kind='barh', figsize=(7,30))
plt.show()


# In[ ]:


acc_test = pd.read_csv('./adult_test_stop_criterion/scores/accuracy_test.txt', header =None)
acc_train = pd.read_csv('./adult_test_stop_criterion/scores/accuracy_train.txt', header =None)
acc_test.columns = ['acc_test']
acc_train.columns = ['acc_train']


# In[ ]:


epsilon=pd.read_csv('./adult_test_stop_criterion/scores/epsilon.txt',header =None)
epsilon.columns=['epsilon']


# In[ ]:


df = acc_test.join(acc_train)
df.head()


# In[ ]:


df = df.join(epsilon)
df.head()


# In[ ]:


df.mean()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.iloc[:,0:2].plot.box(figsize=(7,5))
plt.show()


# In[ ]:


# acc_valid = 0.79
df['diversity'] = np.abs(0.79-df['acc_train'])
df['similarity'] = np.abs(0.79-df['acc_test'])


# In[ ]:


df.head()


# In[ ]:


df.iloc[:,3:5].plot.box()

plt.show()


# In[ ]:


df.iloc[:,3:5].plot(figsize=(10,7))
plt.ylabel('Diversity/Similarity')
plt.xlabel('Epochs')
plt.show()


# Ideally, both measures,i.e. divesrity and similarity has to be = 0. The good ratio quality/utility of generated data was achieved after 50-60 epochs

# In[ ]:




