"""
Last Updated: 27/01/2022
------------------------

Author: Sang Young Noh 
----------------------

Description: 

Class to train the RF model of the GFK dataset, and returns the model. 

This class also takes custom string inputs, 

"""

# Boilerplate python data library imports
import pandas as pd
import numpy as np
import nltk 
from collections import Counter

# Using the nltk library to import common stopwords in English and German,
# which will be used to modify the columns data 
from nltk.corpus import stopwords
# word2vec model
from gensim.models import Word2Vec

# sklearn libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import joblib 

# Visualization 
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('stopwords')
stopEnglish = stopwords.words('english') # List for English stopwords
stopGerman = stopwords.words('german') # List for German stopwords 

# Capitalize words using list comphrensions - as the majority of string data here are capitalized 
stopEnglish = [stopwords.upper() for stopwords in stopEnglish]
stopGerman = [stopwords.upper() for stopwords in stopGerman]


class GFKTaskMLModelGenerator(): 
    """
    """
    def __init__(self, InputCSV, N_Estimator, N_leaf_nodes, test_size, feature_column):
        
        """
        Description:
        ------------
        
        Constructor for our class
        
        InputCSV  - the path to the csv with the relevant columns 
        N_Estimator - number of estimators we want to use with the random forest model 
        leaf_nodes - number of leaf nodes we use with the random forest model 
        test_size - the proportion of the data we want to split when splitting into training and validati ondata
        feature_column - which feature column to use as the feature when training the ML model 
        
        """
        self._InputCSV = InputCSV # Path to CSV to read in the product 
        self._N_Estimator =  N_Estimator # Number of estimates for the random forest classifier 
        self._leaf_nodes =  N_leaf_nodes
        self._test_size = test_size 
        self._feature_column = feature_column       
        # Dictionary to translate the feature column 
        self._BOWDict = {
                'main_text':'main_text_BOW',
                'add_text':'add_text_BOW',
                'manufacturer':'manufacturer_BOW'
        }
        
    def CleanTextColumns(self):
        """
        Description: 
        -----------
        """
        self._ProductLabelEncoder = LabelEncoder() # LabelEnocoder - will need later to inverse label change
        df = pd.read_csv(self._InputCSV, sep= ';') # data is semicolon separated
        df = df.dropna() # remove all rows with NaNs
        
        # We first tokenize the strings inside the column to make the strings more digestable
        df['main_text_tokenized'] = df.apply(lambda row: nltk.word_tokenize(row['main_text']), axis=1)
        df['add_text_tokenized'] = df.apply(lambda row: nltk.word_tokenize(row['add_text']), axis=1)
        df['manufacturer_tokenized'] = df.apply(lambda row: nltk.word_tokenize(row['manufacturer']), axis=1)
        
        # Remove the english and german stopwords - for the column main_text
        df['main_text_tokenized_new'] = df.apply(lambda row: [word for word in row['main_text_tokenized'] if word.isalnum() and len(word) != 1 
                                                          and word not in stopEnglish and word not in stopGerman and not any(c.isdigit() for c in word)], axis =1)
        df['add_text_tokenized_new'] = df.apply(lambda row: [word for word in row['add_text_tokenized'] if word.isalnum() and len(word) != 1 
                                                         and word not in stopEnglish and word not in stopGerman and not any(c.isdigit() for c in word)], axis =1)
        df['manufacturer_tokenized_new'] = df.apply(lambda row: [word for word in row['manufacturer_tokenized'] if word.isalnum() and len(word) != 1 
                                                         and word not in stopEnglish and word not in stopGerman and not any(c.isdigit() for c in word)], axis =1)
    
        self._df_modified = df[['productgroup', 'main_text_tokenized_new','add_text_tokenized_new', 'manufacturer_tokenized_new']]
        self._df_modified['productgroup'] = self._ProductLabelEncoder.fit_transform(self._df_modified['productgroup'])
        self._df_modified = self._df_modified[self._df_modified['main_text_tokenized_new'].map(lambda d: len(d)) > 0]
        self._df_modified = self._df_modified[self._df_modified['add_text_tokenized_new'].map(lambda d: len(d)) > 0]
        self._df_modified = self._df_modified[self._df_modified['manufacturer_tokenized_new'].map(lambda d: len(d)) > 0]
        self._df_modified = self._df_modified.reindex()    
    
    def MakeWord2Vec(self):
        """
        Description:
        ------------
        
        Makes a Word2Vec list of the string vectorizations 
        
        """
        # Extract the unique tokens in each of the feature columns 
        UniqueTokensMainText = self._df_modified["main_text_tokenized_new"].explode().unique()
        UniqueTokensMainText = [str(i) for i in UniqueTokensMainText]
        UniqueTokensAddText = self._df_modified["add_text_tokenized_new"].explode().unique() 
        UniqueTokensAddText = [str(i) for i in UniqueTokensAddText]
        UniqueTokensManufacturerText = self._df_modified["manufacturer_tokenized_new"].explode().unique() 
        UniqueTokensManufacturerText = [str(i) for i in UniqueTokensManufacturerText]
        
        # Train the word2Vec models 
        word2vec_maintext = Word2Vec([UniqueTokensMainText], min_count=1) # word2vec for maintext
        word2vec_addtext = Word2Vec([UniqueTokensMainText], min_count=1) # word2vec for addtext  
        word2vec_manufacturer = Word2Vec([UniqueTokensManufacturerText], min_count=1) # word2vec for Manufacturer
        
        # Relabel the tokens as feature vectors as taken from the word2vec models 
        self._df_modified['main_text_tokenized_new'] = self._df_modified.apply(lambda row: [word2vec_maintext.wv[str(word)] for word in row['main_text_tokenized_new']], axis = 1)
        self._df_modified['add_text_tokenized_new'] = self._df_modified.apply(lambda row: [word2vec_addtext.wv[str(word)] for word in row['add_text_tokenized_new']], axis = 1)
        self._df_modified['manufacturer_tokenized_new'] = self._df_modified.apply(lambda row: [word2vec_manufacturer.wv[str(word)] for word in row['manufacturer_tokenized_new']], axis = 1)
    
    def MakeOneHot(self):
        """
        Description:
        ------------
        
        Makes a one-hot implementation of the text data columns 
        
        """
        # Extract the unique tokens in each of the feature columns 
        
        # Main Text
        self._UniqueTokensMainText = self._df_modified["main_text_tokenized_new"].explode().unique()
        self._UniqueTokensMainText = [str(i) for i in self._UniqueTokensMainText]
        
        # Add Text 
        self._UniqueTokensAddText = self._df_modified["add_text_tokenized_new"].explode().unique() 
        self._UniqueTokensAddText = [str(i) for i in self._UniqueTokensAddText]
        
        # Manufacturer Text 
        self._UniqueTokensManufacturerText = self._df_modified["manufacturer_tokenized_new"].explode().unique() 
        self._UniqueTokensManufacturerText = [str(i) for i in self._UniqueTokensManufacturerText]

        # Train the Bag of Words models 
        self._UniqueWordsCounterMainText = lambda x: Counter([y for y in x if y in self._UniqueTokensMainText])
        self._UniqueWordsCounterAddText = lambda x: Counter([y for y in x if y in self._UniqueTokensAddText])
        self._UniqueWordsCounterManufacturerText = lambda x: Counter([y for y in x if y in self._UniqueTokensManufacturerText])
        
        # Create the bag of words columns for the main_text
        self._df_modified['main_text_BOW'] = (pd.DataFrame(self._df_modified['main_text_tokenized_new'].apply(self._UniqueWordsCounterMainText).values.tolist())
               .fillna(0)
               .astype(int)
               .reindex(columns=self._UniqueTokensMainText)
               .values
               .tolist())
        
        # Create the bag of words columns for the add_text 
        self._df_modified['add_text_BOW'] = (pd.DataFrame(self._df_modified['add_text_tokenized_new'].apply(self._UniqueWordsCounterAddText).values.tolist())
               .fillna(0)
               .astype(int)
               .reindex(columns=self._UniqueTokensAddText)
               .values
               .tolist())
        
        # Create the bag of words columns for the manufacturer 
        self._df_modified['manufacturer_BOW'] = (pd.DataFrame(self._df_modified['manufacturer_tokenized_new'].apply(self._UniqueWordsCounterManufacturerText).values.tolist())
               .fillna(0)
               .astype(int)
               .reindex(columns=self._UniqueTokensManufacturerText)
               .values
               .tolist())
         
    def TrainMLModel(self):
        """
        Description
        -----------
                     
        Takes the one-hot inputs and the categories and splits the data into a training 
        and validation set, then trains the random forest model. 
     
        """
        # Split the test into a training and a validation set 
        train, val = train_test_split(self._df_modified[['productgroup', self._BOWDict[self._feature_column]]], 
                                       test_size = 0.3)
        # We take the manufacturer BOW columns as the training data
        X_train, y_train = train[[self._BOWDict[self._feature_column]]], train[['productgroup']]
        X_test, y_test = val[[self._BOWDict[self._feature_column]]], val[['productgroup']]
        X_train_new, X_test_new = [], []
        
        # Reformat the one-hot data so that it can be read by the classifier
        for entry in np.array(X_train[self._BOWDict[self._feature_column]]):
            X_train_new.append(entry)
            
        # Ditto with the test data 
        for entry in np.array(X_test[self._BOWDict[self._feature_column]]): 
            X_test_new.append(entry)
        
        # Train the Random Forest Model 
        self._rf_clf = RandomForestClassifier(n_estimators=self._N_Estimator, max_leaf_nodes=self._leaf_nodes)
        self._rf_clf.fit(X_train_new, y_train)
        
    def InputText(self, string):
        """
        Description
        -----------
        
        Takes a custom input, tokenizes it, and returns the one-hot vectorized 
        format of that one-hot vector that can be trained with the random forest model.
         
        The modification of the string has to follow the same formatting as has been done
        for the sample columns. 
        
        """
        # NLTK tokenize
        tokenized_string = nltk.word_tokenize(string)
        
        # Modify string in the same way as  CleanTextColumn function 
        modified_tokenized_string = [word for word in tokenized_string if word.isalnum() and len(word) != 1 
                                     and word not in stopEnglish and word not in stopGerman 
                                     and not any(c.isdigit() for c in word)]
        
        # From the relevant category type and counter, change the modified_tokenized string so that 
        # we get a one-hot format that can be read by the trained ML model. 
        OneHotString = []
        Tokens = None 
        # Select the correct tokens 
        if self._feature_column == 'main_text':
            Tokens = self._UniqueTokensMainText
        if self._feature_column == 'add_text':
            Tokens = self._UniqueTokensAddText
        if self._feature_column == 'manufacturer': 
            Tokens = self.UniqueTokensManufacturerText
        
        # Loop over Tokens of columns 
        for onehotentry in Tokens:
            Counter = 0 # Counter to create the one-hot vector as needed 
            # Loop over the tokenized string 
            for entry in modified_tokenized_string:
                if onehotentry == entry: # if we have a matching string, then we add a counter
                    Counter += 1 
            OneHotString.append(Counter) # append as an element to the one-hot vector 
        
        OneHotStringOutput = [entry for entry in OneHotString]
        return [OneHotString]
    
    def PredictCategory(self, string):
        """
        Description
        -----------

        Function that takes the string, and then returns the category 
        from the trained model produced from the class 
        
        """
        # Predict the category for the string input  
        Cat = self._rf_clf.predict(self.InputText(string)) # Predict the category based on the input string 
        CategoryOutput = self._ProductLabelEncoder.inverse_transform(Cat)[0] # Inverse transform numerical back to string 
        return CategoryOutput

