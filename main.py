import os
from fit_predict_categorical_encoding import *

'''
Learning with dirty categorical variables.
'''

# Parameters ##################################################################
datasets = [
            # 'midwest_survey',
            # 'employee_salaries',
            # 'medical_charge',
            'traffic_violations',
            # 'road_safety',
            # 'docs_payments',
            # 'beer_reviews'
            ]
n_jobs = 26
n_splits = 100
test_size = 1./3
encoders = [
            # 'one-hot_encoding_sparse',
            # '3gram_similarity2',
            # '3gram_similarity2_1',
            # '3gram_similarity4',
            # '3gram_similarity2_2',
            # '3gram_similarity5',
            # '3grams_count_vectorizer',
            # '3grams_hot_vectorizer',
            # '3grams_tfidf_vectorizer',
            '3gram_presence_fisher_kernel',
            ]
str_preprocess = True
dimension_reductions = [['-', -1]]
# '-', 'RandomProjectionsGaussian', 'MostFrequentCategories', 'KMeans',

results_path = os.path.join('results', '2018-02-09_100splits')
# results_path = os.path.join('results', '2017-12-05_DimRed')
###############################################################################

fit_predict_categorical_encoding(datasets, n_jobs, n_splits, test_size,
                                 encoders, str_preprocess,
                                 dimension_reductions, results_path,
                                 model_path='')
