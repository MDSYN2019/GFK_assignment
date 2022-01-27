
# Author: Sang Young Noh

# Last Updated: 27/01/2022

# *GFK Microservice*

The repository for the GFK data analysis and microservice app. Step-by-step look 
into the development of the code is in the notebooks folder MLE_Assignment-SangYoungNoh.ipynb

The summary of the model is that we use a combination of bag-of-words model to
transform the category data to a one-hot vector. This is then used as the ML 
input to a random forest model.

Further improvements to this would be cross-validation models, and hyperparameter
tuning for the model as part of the pipeline of the model as an example. Also, the class
includes a failed attmempt at trying to make a word2vec vectorization of the columns, which,
given a longer description and a contextual description for each category, would be much more suitable
for this product.

# 

Libraries required are listed in requirments.txt. They are:

* flask
* pandas
* numpy
* sklearn
* nltk
* gensim
* seaborn 



To run the unittests, run the following command:

```
python -m unittest test.test_ML
```

The Flask API takes an example string and outputs the category by loading. This could be improved by loading in a
pickle binary sklearn library, but unfortunately I haven't had the opportunit to expand on this. As my experience with developing in Flask is rather elementary, it would take more time to develop the (base). 

To view the Flask API, run :

```
python3 app.py 
```

Once this app is run, if you go to http://localhost:105/result/ the category for 
a single string fragment should be shown. 
