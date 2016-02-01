__author__ = 'udit'

import graphlab
from math import log, sqrt
import numpy as np


def get_numpy_data(data_sframe, features, output):
    # add a constant column to an SFrame
    data_sframe['constant'] = 1
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_SFrame given by the 'features' list into the SFrame 'features_sframe'
    features_sframe = data_sframe[features]
    # this will convert the features_sframe into a numpy matrix with GraphLab Create >= 1.7!!
    features_matrix = features_sframe.to_numpy()
    # assign the column of data_sframe associated with the target to the variable 'output_sarray'
    output_sarray = data_sframe[output]
    # this will convert the SArray into a numpy array:
    output_array = output_sarray.to_numpy()  # GraphLab Create>= 1.7!!
    return (features_matrix, output_array)


def predict_output(feature_matrix, weights):
    # predictions = feature_matrix.dot(weights)
    predictions = np.dot(feature_matrix, weights)
    return predictions


def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_feature_matrix = feature_matrix / norms
    return normalized_feature_matrix, norms


def calculate_ro(feature_matrix, output, prediction, weights, simple_features):
    all_ro = []
    for indx in range(len(simple_features)):
        i = indx + 1
        ro = (feature_matrix[:, i] * (output - prediction + weights[i] * feature_matrix[:, i])).sum()
        all_ro.append(ro)
    return all_ro


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = predict_output(feature_matrix, weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = (feature_matrix[:, i] * (output - prediction + weights[i] * feature_matrix[:, i])).sum()

    if i == 0:
        new_weight_i = ro_i
    elif ro_i < -l1_penalty / 2.:
        new_weight_i = ro_i + l1_penalty / 2
    elif ro_i > l1_penalty / 2.:
        new_weight_i = ro_i - l1_penalty / 2
    else:
        new_weight_i = 0.

    return new_weight_i


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    optimized = False
    while not optimized:
        weight_diff = []
        for i in range(len(initial_weights)):
            old_weight = initial_weights[i]
            initial_weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, initial_weights, l1_penalty)
            weight_diff.append(abs(old_weight - initial_weights[i]))
        if sum(weight_diff) < tolerance:
            optimized = True
    return initial_weights


def get_residual_sum_of_squares(predictions, outcome):
    # First get the predictions -- predictions
    # Then compute the residuals/errors
    errors = outcome - predictions
    # Then square and add them up
    RSS = errors.dot(errors.transpose()).sum()
    return RSS


sales = graphlab.SFrame('kc_house_data.gl/')
# In the dataset, 'floors' was defined with type string,
# so we'll convert them to int, before using it below
sales['floors'] = sales['floors'].astype(int)

features, norms = normalize_features(np.array([[3., 6., 9.], [4., 8., 12.]]))

print features
print norms

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)

simple_feature_matrix, norms = normalize_features(simple_feature_matrix)
weights = np.array([1., 4., 1.])

prediction = predict_output(simple_feature_matrix, weights)
print "prediction is", prediction

ro = calculate_ro(simple_feature_matrix, output, prediction, weights, simple_features)
print "ro is", ro

# Recall that, whenever ro[i] falls between -l1_penalty/2 and l1_penalty/2, the corresponding weight w[i] is
# sent to zero. Now suppose we were to take one step of coordinate descent on either feature 1 or feature 2.
# What range of values of l1_penalty would not set w[1] zero, but would set w[2] to zero,
# if we were to take a step in that coordinate?

l1_penaltys = [1.4e8, 1.64e8, 1.73e8, 1.9e8, 2.3e8]

for l1_penalty in l1_penaltys:
    print (-l1_penalty / 2 <= ro[0] <= l1_penalty / 2)
    print (-l1_penalty / 2 <= ro[1] <= l1_penalty / 2)

# should print 0.425558846691
import math

print lasso_coordinate_descent_step(1, np.array(
    [[3. / math.sqrt(13), 1. / math.sqrt(10)], [2. / math.sqrt(13), 3. / math.sqrt(10)]]),
                                    np.array([1., 1.]), np.array([1., 4.]), 0.1)

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
(normalized_simple_feature_matrix, simple_norms) = normalize_features(simple_feature_matrix)  # normalize features
weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)
print "weights for 2 features are", weights

# calculate rss here on normalized data set
predictions_for_2_features = predict_output(normalized_simple_feature_matrix, weights)
rss_for_2_features = get_residual_sum_of_squares(predictions_for_2_features,
                                                 output)
print "rss for weights with 2 features is", rss_for_2_features

# lasso with more features

train_data, test_data = sales.random_split(.8, seed=0)
all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront',
                'view',
                'condition',
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built',
                'yr_renovated']
my_output = 'price'
initial_weights1en = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
(simple_train_feature_matrix, train_data_output) = get_numpy_data(train_data, all_features, my_output)
(normalized_train_simple_feature_matrix, multi_feature_norms) = normalize_features(
    simple_train_feature_matrix)  # normalize features
l1_penalty = 1e7
tolerance = 1.0
weights1e7 = lasso_cyclical_coordinate_descent(normalized_train_simple_feature_matrix, train_data_output,
                                               initial_weights1en, l1_penalty, tolerance)
print "weihgts for 13 features l1_penalty 1e7", weights1e7

l1_penalty = 1e8
weights1e8 = lasso_cyclical_coordinate_descent(normalized_train_simple_feature_matrix, train_data_output,
                                               initial_weights1en, l1_penalty, tolerance)
print "weihgts for 13 features l1_penalty 1e8", weights1e8

l1_penalty = 1e4
tolerance = 5e5
weights1e4 = lasso_cyclical_coordinate_descent(normalized_train_simple_feature_matrix, train_data_output,
                                               initial_weights1en, l1_penalty, tolerance)
print "weihgts for 13 features l1_penalty 1e4", weights1e4


print weights1e8
print weights1e7
print weights1e4

# normalized_weights1e4 = weights1e4 / multi_feature_norms
# normalized_weights1e7 = weights1e7 / multi_feature_norms
# normalized_weights1e8 = weights1e8 / multi_feature_norms

# print multi_feature_norms[2]
# print weights1e8[2]
# print weights1e8[2] / multi_feature_norms[2]
# print weights1e7[2]
# print weights1e7[2] / multi_feature_norms[2]
# print weights1e4[2]
# print weights1e4[2] / multi_feature_norms[2]

# print "normalized_weights1e4", normalized_weights1e4
# print "normalized_weights1e7", normalized_weights1e7
# print "normalized_weights1e8", normalized_weights1e8
#
# (test_feature_matrix, test_output) = get_numpy_data(test_data, all_features, 'price')
#
# # rss for test unnormalized data for normalized_weights1e4
# prediction_weights1e4 = predict_output(test_feature_matrix, normalized_weights1e4)
# rss_weights1e4 = get_residual_sum_of_squares(prediction_weights1e4, test_output)
# print "rss for test unnormalized data for normalized_weights1e4", rss_weights1e4
#
# # rss for test unnormalized data for normalized_weights1e7
# prediction_weights1e7 = predict_output(test_feature_matrix, normalized_weights1e7)
# rss_weights1e7 = get_residual_sum_of_squares(prediction_weights1e7, test_output)
# print "rss for test unnormalized data for normalized_weights1e7", rss_weights1e7
#
# # rss for test unnormalized data for normalized_weights1e8
# prediction_weights1e8 = predict_output(test_feature_matrix, normalized_weights1e8)
# rss_weights1e8 = get_residual_sum_of_squares(prediction_weights1e8, test_output)
# print "rss for test unnormalized data for normalized_weights1e8", rss_weights1e8
