#turicreate_recommendation_engine
#coded for python
#using a windows laptop, opened in ConEmu, linked to ubuntu via WSL.  windows is not supported for turicreate as of 8/22/18 

#Requires installing turicreate, which requires using IOS or LINUX.  follow the instructions here:
#https://github.com/apple/turicreate/blob/master/README.md

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


#importing the test and train datasets
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

train_data = turicreate.SFrame(ratings_train)
test_data = turicreate.SFrame(ratings_test)
print(test_data)
print("Program is COMPLETE, mi sybiostro")
