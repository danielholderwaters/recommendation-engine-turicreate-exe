#turicreate_recommendation_engine
#coded for python
#using a windows laptop, opened in ConEmu, linked to ubuntu via WSL.  windows is not supported for turicreate as of 8/22/18 

#Requires installing turicreate, which requires using IOS or LINUX.  follow the instructions here:
#https://github.com/apple/turicreate/blob/master/README.md
# in the command prompt, this is helpful for installing packages once in the virtual environment::: sudo python -m pip install matplotlib

#when in conEmu and have activated a virtual environment, paste the following into the command prompt to run:
#python venv/turicreate_recommendation_engine.py

#turicreate dependencies
# pandas, decorator, prettytable, pillow, mxnet, coremltools, numpy, requests
import pandas as pd

#%matplotlib inline
#that last line was hashed because this was written in IPython, not Sublimetext. my config doesn't require it.  windows-only requires changing sublimetext build system.  check this out: https://stackoverflow.com/questions/10831882/matplotlib-plots-not-displaying-in-sublimetext


#if you get an error with matplotlib, specifically when calling 'import matplotlib.pyplot as plt':
#ImportError: No module named _tkinter, please install the python-tk package
#this stackoverflow example solved my problem (I had to add sudo at the front of the code in order to install the package):
#https://stackoverflow.com/questions/4783810/install-tkinter-for-python
import matplotlib  
import matplotlib.pyplot as plt 
import numpy as np
import turicreate
import os
os.chdir('/mnt/c/Users/danie/Desktop/PythonCode')
print ("current working directory is:", os.getcwd())

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
encoding='latin-1')

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

#importing the test and train datasets
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

#After installing turicreate, 
#first lets import it and read the train and test dataset in our environment. 
#Since we will be using turicreate, we will need to convert the dataset in SFrames

train_data = turicreate.SFrame(ratings_train)
test_data = turicreate.SFrame(ratings_test)

#starting with a popularity model - the movies that have the overall highest scores are at the top.
popularity_model = turicreate.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

#this takes the popularity model and churns out a recommendation.
popularity_recomm = popularity_model.recommend(users=[1,2,3,4,5],k=5)
popularity_recomm.print_rows(num_rows=25)

#we will now build a collaborative filtering model. Lets train the item similarity model and make top 5 recommendations for the first 5 users.
#Training the model
item_sim_model = turicreate.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='cosine')

#Making recommendations
item_sim_recomm = item_sim_model.recommend(users=[1,2,3,4,5],k=5)
item_sim_recomm.print_rows(num_rows=25)


# Below is how matrix factorization works for predicting ratings:
# for f = 1,2,....,k:
# for rui in R :
# predict rui
# update puk and qki

#the following is a function that can predict ratings
class MF():

    # Initializing the user-movie rating matrix, no. of latent features, alpha and beta.
    def __init__(self, R, K, alpha, beta, iterations):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    # Initializing user-feature and movie-feature matrix 
    def train(self):
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # List of training samples
        self.samples = [
        (i, j, self.R[i, j])
        for i in range(self.num_users)
        for j in range(self.num_items)
        if self.R[i, j] > 0
        ]

        # Stochastic gradient descent for given number of iterations
        training_process = []
        for i in range(self.iterations):
        	np.random.shuffle(self.samples)
        	self.sgd()
        	mse = self.mse()
        	training_process.append((i, mse))
        if (i+1) % 20 == 0:
            print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    # Computing total mean squared error
    def mse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    # Ratings for user i and moive j
    def get_rating(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Full user-movie rating matrix
    def full_matrix(self):
        return mf.b + mf.b_u[:,np.newaxis] + mf.b_i[np.newaxis:,] + mf.P.dot(mf.Q.T)


#the following, R, is a user-item matrix with 0s where movies have not been reviewed
R= np.array(ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0))
print(R)
#Now let us predict all the missing ratings. Lets take K=20, alpha=0.001, beta=0.01 and iterations=100.
mf = MF(R, K=20, alpha=0.001, beta=0.01, iterations=100)
training_process = mf.train()

print("P x Q:")
print(mf.full_matrix())



print("Program is COMPLETE, mi sybiostro")
