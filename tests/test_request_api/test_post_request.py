import unittest
from mlClassifier.middleware.features_request import PostRequest

request = PostRequest()   # class containing all the useful dataframes and arrays, prior to data preprocessing.
features = request.get_features_labels()  # Array containing all the features from the .csv input file.


class TestPostRequestFormat(unittest.TestCase):

    classification_features = [
        "isPeFile",
        "isValidSignedFile",
        "fileSize",
        "filePrevalence",
        "importFunctionCount",
        "importModuleCount",
        "peAppendedSize",
    ]

    def test_classification_features_size(self):
        self.assertCountEqual(features, self.classification_features, "missing one or more features")

    def test_file_type(self):
        self.assertEqual(request.extract_filetype(), ".csv")

    def test_file_size(self):
        self.assertTrue(request.get_file_size() <= 10)  # megabytes.

