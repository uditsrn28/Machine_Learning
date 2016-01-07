__author__ = 'udit'
import numpy
import pandas
import math


def get_log(x):
    return math.log(x)


def get_residual_sum_of_squares(model, data):
    # First get the predictions
    predictions = model.predict(data)
    # Then compute the residuals/errors
    errors = data['price'] - predictions
    # Then square and add them up
    rss = (errors * errors).sum()
    return rss


sales = pandas.read_csv('kc_house_data.csv')
train_data = pandas.read_csv('kc_house_train_data.csv')
test_data = pandas.read_csv('kc_house_test_data.csv')

train_data['bedrooms_squared'] = train_data['bedrooms'] * train_data['bedrooms']
train_data['bed_bath_rooms'] = train_data['bedrooms'] * train_data['bathrooms']
train_data['log_sqft_living'] = train_data['sqft_living'].apply(get_log)
train_data['lat_plus_long'] = train_data['lat'] * train_data['long']

test_data['bedrooms_squared'] = test_data['bedrooms'] * test_data['bedrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms'] * test_data['bathrooms']
test_data['log_sqft_living'] = test_data['sqft_living'].apply(get_log)
test_data['lat_plus_long'] = test_data['lat'] * test_data['long']

print "Bedrooms_squared mean is  on test data ", test_data['bedrooms_squared'].mean()
print "bed_bath_rooms mean is on test data ", test_data['bed_bath_rooms'].mean()
print "log_sqft_living mean is on test data ", test_data['log_sqft_living'].mean()
print "lat_plus_long mean is on test data ", test_data['lat_plus_long'].mean()

########################################################################################################################
################      WRITE FUNCTIONS FOR CALCULATING LINEAR REGRESSION FOR MULTIPLE VALUES           ##################
########################################################################################################################


# model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
# model_2_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms']
# model_3_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms', 'bedrooms_squared',
#                     'log_sqft_living', 'lat_plus_long']
# model_1 = graphlab.linear_regression.create(train_data, target='price',
#                                             features=model_1_features,
#                                             validation_set=None, verbose=False)
# model_2 = graphlab.linear_regression.create(train_data, target='price',
#                                             features=model_2_features, validation_set=None, verbose=False)
# model_3 = graphlab.linear_regression.create(train_data, target='price',
#                                             features=model_3_features, validation_set=None, verbose=False)

# print "coefficients of model 1 are ", model_1.get('coefficients')
# print "coefficients of model 2 are ", model_2.get('coefficients')
# print "coefficients of model 3 are ", model_3.get('coefficients')
#
# print "rss for model 1 on test data", get_residual_sum_of_squares(model_1, test_data)
# print "rss for model 2 on test data", get_residual_sum_of_squares(model_2, test_data)
# print "rss for model 3 on test data", get_residual_sum_of_squares(model_3, test_data)
#
# print "rss for model 1 on train data", get_residual_sum_of_squares(model_1, test_data)
# print "rss for model 2 on train data", get_residual_sum_of_squares(model_2, test_data)
# print "rss for model 3 on train data", get_residual_sum_of_squares(model_3, test_data)
#
#
