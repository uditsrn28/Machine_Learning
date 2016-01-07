__author__ = 'udit'
import graphlab
import math


def get_log(x):
    return math.log(x)


def get_residual_sum_of_squares(model, data, outcome):
    # First get the predictions
    predictions = model.predict(data)
    # Then compute the residuals/errors
    errors = outcome - predictions
    # Then square and add them up
    rss = (errors * errors).sum()
    return rss


sales = graphlab.SFrame('kc_house_data.gl/')
sales['bedrooms_squared'] = sales['bedrooms'] * sales['bedrooms']
sales['bed_bath_rooms'] = sales['bedrooms'] * sales['bathrooms']
sales['log_sqft_living'] = sales['sqft_living'].apply(get_log)
sales['lat_plus_long'] = sales['lat'] + sales['long']

train_data, test_data = sales.random_split(.8, seed=0)

print "Bedrooms_squared mean is  on test data ", test_data['bedrooms_squared'].mean()
print "bed_bath_rooms mean is on test data ", test_data['bed_bath_rooms'].mean()
print "log_sqft_living mean is on test data ", test_data['log_sqft_living'].mean()
print "lat_plus_long mean is on test data ", test_data['lat_plus_long'].mean()

model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
model_2_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms']
model_3_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms', 'bedrooms_squared',
                    'log_sqft_living', 'lat_plus_long']
model_1 = graphlab.linear_regression.create(train_data, target='price',
                                            features=model_1_features,
                                            validation_set=None, verbose=False)
model_2 = graphlab.linear_regression.create(train_data, target='price',
                                            features=model_2_features, validation_set=None, verbose=False)
model_3 = graphlab.linear_regression.create(train_data, target='price',
                                            features=model_3_features, validation_set=None, verbose=False)

print "coefficients of model 1 are ", model_1.get('coefficients')
print "coefficients of model 2 are ", model_2.get('coefficients')
print "coefficients of model 3 are ", model_3.get('coefficients')

print "rss for model 1 on test data", get_residual_sum_of_squares(model_1, test_data, test_data['price'])
print "rss for model 2 on test data", get_residual_sum_of_squares(model_2, test_data, test_data['price'])
print "rss for model 3 on test data", get_residual_sum_of_squares(model_3, test_data, test_data['price'])

print "rss for model 1 on train data", get_residual_sum_of_squares(model_1, train_data, train_data['price'])
print "rss for model 2 on train data", get_residual_sum_of_squares(model_2, train_data, train_data['price'])
print "rss for model 3 on train data", get_residual_sum_of_squares(model_3, train_data, train_data['price'])


print "price of 1 house in test set", test_data[0]['price']