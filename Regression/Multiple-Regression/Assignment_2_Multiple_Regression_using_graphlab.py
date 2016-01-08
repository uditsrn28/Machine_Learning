__author__ = 'udit'
import graphlab
import math
import numpy as np

sales = graphlab.SFrame('kc_house_data.gl/')
train_data, test_data = sales.random_split(.8, seed=0)


def get_numpy_data(data_sframe, features, output):
    # add a constant column to an SFrame
    data_sframe['constant'] = 1
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_SFrame given by the 'features' list into the SFrame 'features_sframe'
    features_sframe = graphlab.SFrame()
    for feature in features:
        features_sframe[feature] = data_sframe[feature]
    # this will convert the features_sframe into a numpy matrix with GraphLab Create >= 1.7!!
    features_matrix = features_sframe.to_numpy()
    features_matrix = features_matrix.transpose()
    # assign the column of data_sframe associated with the target to the variable 'output_sarray'
    output_sarray = data_sframe[output]
    # this will convert the SArray into a numpy array:
    output_array = output_sarray.to_numpy()  # GraphLab Create>= 1.7!!
    return (features_matrix, output_array)


def predict_outcome(feature_matrix, weights):
    # predictions = feature_matrix.dot(weights)
    predictions = weights.dot(feature_matrix)
    return predictions


def feature_derivative(errors, feature):
    derivative = 2 * feature.dot(errors)
    return derivative


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        predictions = predict_outcome(feature_matrix, weights)
        # compute the errors as predictions - output:
        errors = predictions - output
        gradient_sum_squares = 0  # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors, feature_matrix[i])
            # add the squared derivative to the gradient magnitude
            # print "derivative is ", derivative
            gradient_sum_squares = (derivative * derivative).sum()
            # print "gradient sum square is ", gradient_sum_squares
            # update the weight based on step size and derivative:
            weights[i] = weights[i] - (step_size * derivative)
        gradient_magnitude = math.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return weights


def get_residual_sum_of_squares(predictions, data, outcome):
    # First get the predictions -- predictions
    # Then compute the residuals/errors
    actual_price = data[outcome].to_numpy()
    errors = actual_price - predictions
    # Then square and add them up
    RSS = errors.dot(errors.transpose()).sum()
    return RSS


features = ['sqft_living']
output_feature = 'price'
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

print "features are ", features
simple_feature_matrix, output = get_numpy_data(train_data, features, output_feature)
print "feature matrix is ", simple_feature_matrix
print "output is ", output
train_data, test_data = sales.random_split(.8, seed=0)

simple_weights = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)

print "Simple weights are ", simple_weights

test_simple_feature_matrix, test_output = get_numpy_data(test_data, features, output_feature)
test_predictions = predict_outcome(test_simple_feature_matrix, simple_weights)
print "predictions for test is ", test_predictions

test_rss = get_residual_sum_of_squares(test_predictions, test_data, output_feature)
print "residual sum of squares of test data is ", test_rss


multiple_features = ['sqft_living', 'sqft_living15']
multiple_features_output = 'price'
(multi_feature_matrix, multi_output) = get_numpy_data(train_data, multiple_features, multiple_features_output)
print "multi model feature matrix is ", multi_feature_matrix
print "multi model output is ", multi_output
multi_initial_weights = np.array([-100000., 1., 1.])
multi_step_size = 4e-12
multi_tolerance = 1e9

multiple_model_weights = regression_gradient_descent(multi_feature_matrix, multi_output,
                                                     multi_initial_weights, multi_step_size, multi_tolerance)

print "Multiple model weights are ", multiple_model_weights

multi_model_test_simple_feature_matrix, multi_model_test_output = get_numpy_data(test_data, multiple_features,
                                                                                 multiple_features_output)
multi_model_test_predictions = predict_outcome(multi_model_test_simple_feature_matrix, multiple_model_weights)
print "multil model predictions for test is ", multi_model_test_predictions

multi_model_test_rss = get_residual_sum_of_squares(multi_model_test_predictions, test_data, multiple_features_output)
print "residual sum of squares of test data is ", multi_model_test_rss
