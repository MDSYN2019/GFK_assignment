"""
An extremely basic Flask API that returns the category 
from a 
"""

import os
from flask import Flask, request, redirect, url_for, flash,jsonify
from werkzeug.utils import secure_filename
from flask import Flask

# Import the ML model class 
from GFK.GFKAssignment import GFKTaskMLModelGenerator

# Load Flask app
app = Flask(__name__)

# Load the ML Model, tokenize and train  
RFModel = GFKTaskMLModelGenerator('data/testset_C.csv', 10, 16, 0.3, 'main_text')
RFModel.CleanTextColumns()
RFModel.MakeOneHot()
RFModel.TrainMLModel()

# Example Input for the Flask API 
ExampleInputMainText = 'LEEF IBRIDGE MOBILE SPEICHERERWEITERUNG FUER IPHONE, IPAD UND IPOD - MIT LIGHTNING UND USB, 128 GB'

@app.route('/result/', methods=['GET', 'POST'])
def ReturnCategory():
    return "For the main text '{}', we get the category {}".format(ExampleInputMainText, RFModel.PredictCategory(ExampleInputMainText))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
