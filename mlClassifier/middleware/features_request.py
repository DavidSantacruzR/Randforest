class PostRequest:
    """
    ##puts the post request data from json data into a pandas dataframes and arrays.
    """

    def extract_features_to_dataframe(self):
        """
        ## extracts the data and returns a dataframe containing all the features for pre-modelling.
        :return: pd.Dataframe
        """
        pass

    def get_features_labels(self):
        """
        :return: array with all the field labels from the .csv input file.
        """
        pass

    def extract_filetype(self):
        pass

    def get_file_size(self):
        """
        shouldn't be more than 10 MB.
        :return:
        """
        pass
