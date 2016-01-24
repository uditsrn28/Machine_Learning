import sframe
import numpy as np
import matplotlib.pyplot as plt

sales = sframe.SFrame('kc_house_data.gl/')


def get_numpy_data(data_sframe, features, output):
    # add a constant column to an SFrame
    data_sframe['constant'] = 1
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_SFrame given by the 'features' list into the SFrame 'features_sframe'
    features_sframe = sframe.SFrame()
    features_sframe[features] = data_sframe[features]
    # this will convert the features_sframe into a numpy matrix with GraphLab Create >= 1.7!!
    features_matrix = features_sframe.to_numpy()
    features_matrix = features_matrix
    # assign the column of data_sframe associated with the target to the variable 'output_sarray'
    output_sarray = data_sframe[output]
    # this will convert the SArray into a numpy array:
    output_array = output_sarray.to_numpy()  # GraphLab Create>= 1.7!!
    return features_matrix, output_array


def predict_output(feature_matrix, weights):
    # predictions = feature_matrix.dot(weights)
    predictions = np.dot(feature_matrix, weights)
    return predictions


def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    derivative = 2 * np.dot(errors, feature)
    if not feature_is_constant:
        # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
        derivative = derivative + 2 * l2_penalty * weight
    return derivative


# (example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
# my_weights = np.array([1., 10.])
# test_predictions = predict_output(example_features, my_weights)
# errors = test_predictions - example_output  # prediction errors
# # next two lines should print the same values
# print feature_derivative_ridge(errors, example_features[:, 1], my_weights[1], 1, False)
# print np.sum(errors * example_features[:, 1]) * 2 + 20.
# print ''
#
# # next two lines should print the same values
# print feature_derivative_ridge(errors, example_features[:, 0], my_weights[0], 1, True)
# print np.sum(errors) * 2.


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty,
                                      max_iterations=100):
    weights = np.array(initial_weights)  # make sure it's a numpy array

    # while not reached maximum number of iterations:
    iterations = 1
    while iterations <= max_iterations:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = predictions - output

        for i in xrange(len(weights)):  # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # (Remember: when i=0, you are computing the derivative of the constant!)
            if i == 0:
                is_feature_constant = True
            else:
                is_feature_constant = False
            # compute the derivative for weight[i].

            derivative = feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty,
                                                  is_feature_constant)

            weights[i] = weights[i] - step_size * derivative

            # subtract the step size times the derivative from the current weight
        iterations = iterations + 1

    return weights


def get_residual_sum_of_squares(feature_matrix, weights, outcome):
    # First get the predictions
    predictions = predict_output(feature_matrix, weights)
    # Then compute the residuals/errors
    errors = outcome - predictions
    # Then square and add them up
    rss = (errors * errors).sum()
    return rss


simple_features = ['sqft_living']
my_output = 'price'
train_data, test_data = sales.random_split(.8, seed=0)
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)
initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations = 1000
l2_penalty_small = 0.0

simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights,
                                                             step_size, l2_penalty_small)
print "simple weights with 0 penalty", simple_weights_0_penalty

l2_penalty_large = 1e11
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights,
                                                                step_size, l2_penalty_large)

print "simple weights with high penalty", simple_weights_high_penalty
# plt.plot(simple_feature_matrix, output, 'k.',
#          simple_feature_matrix, predict_output(simple_feature_matrix, simple_weights_0_penalty), 'b-',
#          simple_feature_matrix, predict_output(simple_feature_matrix, simple_weights_high_penalty), 'r-')
#
# plt.show()

rss_0_weights = get_residual_sum_of_squares(simple_test_feature_matrix, initial_weights, test_output)
rss_no_regularization_weights = get_residual_sum_of_squares(simple_test_feature_matrix, simple_weights_0_penalty,
                                                            test_output)
rss_high_regularization_weights = get_residual_sum_of_squares(simple_test_feature_matrix, simple_weights_high_penalty,
                                                              test_output)

print "rss with 0 as weights", rss_0_weights
print "rss with low regularization as weights", rss_no_regularization_weights
print "rss with high regularization as weights", rss_high_regularization_weights

model_features = ['sqft_living',
                  'sqft_living15']  # sqft_living15 is the average squarefeet for the nearest 15 neighbors.
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

initial_weights = np.array([0.0, 0.0, 0.0])
step_size = 1e-12
max_iterations = 1000
l2_penalty_small = 0.0

multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights,
                                                               step_size, l2_penalty_small, max_iterations)

print "multiple weights with no penalty", multiple_weights_0_penalty
multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights,
                                                                  step_size, l2_penalty_large, max_iterations)
print "multiple weights with high penalty", multiple_weights_high_penalty

rss_0_weights_multi_feature = get_residual_sum_of_squares(test_feature_matrix, initial_weights, test_output)
rss_no_regularization_weights_multi_feature = get_residual_sum_of_squares(test_feature_matrix,
                                                                          multiple_weights_0_penalty,
                                                                          test_output)
rss_high_regularization_weights_multi_feature = get_residual_sum_of_squares(test_feature_matrix,
                                                                            multiple_weights_high_penalty,
                                                                            test_output)

print "two features rss with 0 as weights", rss_0_weights_multi_feature
print "two features rss with low regularization as weights", rss_no_regularization_weights_multi_feature
print "two features rss with high regularization as weights", rss_high_regularization_weights_multi_feature

print "actual price for test data", test_output
prediction_no_regularization = predict_output(test_feature_matrix, multiple_weights_0_penalty)
print "prediction with no regularization ", prediction_no_regularization

prediction_high_regularization = predict_output(test_feature_matrix, multiple_weights_high_penalty)
print "prediction with high regularization ", prediction_high_regularization
