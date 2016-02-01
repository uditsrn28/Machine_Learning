__author__ = 'udit'

import graphlab
from math import log, sqrt
import numpy as np

sales = graphlab.SFrame('kc_house_data.gl/')

sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms'] * sales['bedrooms']

# In the dataset, 'floors' was defined with type string,
# so we'll convert them to float, before creating a new feature.
sales['floors'] = sales['floors'].astype(float)
sales['floors_square'] = sales['floors'] * sales['floors']

all_features = ['bedrooms', 'bedrooms_square', 'bathrooms', 'sqft_living', 'sqft_living_sqrt', 'sqft_lot',
                'sqft_lot_sqrt', 'floors', 'floors_square', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                'sqft_basement', 'yr_built', 'yr_renovated']

model_all = graphlab.linear_regression.create(sales, target='price', features=all_features, validation_set=None,
                                              l2_penalty=0., l1_penalty=1e10, verbose=False)
print "coefficients are ", model_all.get('coefficients').print_rows(num_rows=20, num_columns=3)

(training_and_validation, testing) = sales.random_split(.9, seed=1)  # initial train/test split
(training, validation) = training_and_validation.random_split(0.5, seed=1)  # split training into train and validate

l1_penalty = np.logspace(1, 7, num=13)

lowest_rss = 9999999999e+90
lowest_l1_penalty = l1_penalty[0]
l1_penalty_non_zero_coefficients = {}
for penalty in l1_penalty:
    model = graphlab.linear_regression.create(training, target='price', features=all_features, validation_set=None,
                                              l2_penalty=0., l1_penalty=penalty, verbose=False)
    l1_penalty_non_zero_coefficients[penalty] = model['coefficients']['value'].nnz()
    error = model.predict(validation) - validation['price']
    RSS = np.dot(error, error)
    if RSS < lowest_rss:
        lowest_rss = RSS
        lowest_l1_penalty = penalty

print "penalty with lowest rss is ", lowest_l1_penalty
print "lowers rss is", lowest_rss
print "count of non zero coeffients are", l1_penalty_non_zero_coefficients[lowest_l1_penalty]

# rss on test data for penalty with lowest rss

model_lowest_rss_penalty = graphlab.linear_regression.create(training, target='price', features=all_features,
                                                             validation_set=None,
                                                             l2_penalty=0., l1_penalty=penalty, verbose=False)
error_lowest_rss_penalty_testing_data = model.predict(testing) - testing['price']
RSS_lowest_rss_penalty_testing_data = np.dot(error, error)
print "RSS_lowest_rss_penalty_testing_data", RSS_lowest_rss_penalty_testing_data

max_nonzeros = 7

l1_penality_with_number_of_non_zero_coefficients = {}
for l1_penalty_values in np.logspace(8, 10, num=20):
    model_to_get_narrow_values = graphlab.linear_regression.create(training, target='price', features=all_features,
                                                                   validation_set=None, l2_penalty=0.,
                                                                   l1_penalty=l1_penalty_values, verbose=False)
    l1_penality_with_number_of_non_zero_coefficients[l1_penalty_values] = model_to_get_narrow_values['coefficients'][
        'value'].nnz()

print "l1_penality_with_number_of_non_zero_coefficients", l1_penality_with_number_of_non_zero_coefficients

l1_penalty_min = 2976351441.6313133
l1_penalty_max = 3792690190.7322536

l1_penalty_values_min_max = np.linspace(l1_penalty_min, l1_penalty_max, 20)

lowest_l1_penalty_min_max = -1
lowest_rss_min_max = 9999999e100000
for l1_penalty_min_max in l1_penalty_values_min_max:
    model_to_get_best_narrowed_value = graphlab.linear_regression.create(training, target='price',
                                                                         features=all_features,
                                                                         validation_set=None, l2_penalty=0.,
                                                                         l1_penalty=l1_penalty_min_max, verbose=False)
    error = model_to_get_best_narrowed_value.predict(validation) - validation['price']
    RSS = np.dot(error, error)
    print l1_penalty_min_max, RSS
    if RSS < lowest_rss_min_max:
        lowest_rss_min_max = RSS
        lowest_l1_penalty_min_max = l1_penalty_min_max

print "penalty with lowest rss is ", lowest_l1_penalty_min_max
print "lowers rss is", lowest_rss_min_max

# getting coefficients for l1_penalty_min_max
model_with_lowest_min_max = graphlab.linear_regression.create(training, target='price',
                                                              features=all_features,
                                                              validation_set=None, l2_penalty=0.,
                                                              l1_penalty=lowest_l1_penalty_min_max, verbose=False)
print "Coefficient for l1_penalty min max are", model_with_lowest_min_max.get('coefficients').print_rows(num_rows=20,
                                                                                                         num_columns=3)
