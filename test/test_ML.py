"""
Last Updated: 27/01/2022 
------------------------

Author: Sang Young Noh 
-----------------------

Description:

Basic test functions to check the class functionality 
of the RF ML Model. 

Further checks can be added as the model is developed. 

Unittests for the Flask microservice can be added here as an option as well.

"""

# import GFKclass to test 
from GFK.GFKAssignment import GFKTaskMLModelGenerator
# Unittest library 
import unittest

# Trained model here
ExampleModel = GFKTaskMLModelGenerator('data/testset_C.csv', 10, 16, 0.3, 'main_text')
ExampleModel.CleanTextColumns()
ExampleModel.MakeOneHot()
ExampleModel.TrainMLModel()

# Example string input to test in the ML model
ExampleInput = 'LEEF IBRIDGE MOBILE SPEICHERERWEITERUNG FUER IPHONE, IPAD UND IPOD - MIT LIGHTNING UND USB, 128 GB'

class TestGFKMethods(unittest.TestCase):
    """
    
    Two simple methods for testing some parameters 
    in the RF ML model that we have defined

    test_gfk_ML_length - function checks the length of the one-hot generated from a sample input 
    test_gfk_string_input -  function checks whether the output from the ML model matches the output category strings 
 
    """
    def test_gfk_ML_length(self):
        """
        Checks the one hot vector length of the input string example with a known vector length
        of a working input string and checks if they are consistent  
        """
        self.assertEqual(len(ExampleModel.InputText(ExampleInput)[0]), len(ExampleModel._df_modified['main_text_BOW'][3]))

    def test_gfk_ML_string_input(self):
        """
        Check if the string input is valid, and gives out the category  
        as trained in the model 
        """
        self.assertTrue(ExampleModel.PredictCategory(ExampleInput) in ['WASHINGMACHINES', 'USB MEMORY', 'BICYCLES', 'CONTACT LENSES']) 

if __name__ == '__main__':
    unittest.main()
    
