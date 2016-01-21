__author__ = 'udit'

import graphlab
import matplotlib.pyplot as plt
import numpy as np


def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree + 1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_sframe[name] to be feature^power
            poly_sframe[name] = feature.apply(lambda x: x ** power)
    return poly_sframe


def get_model_with_degree_poly(data, degree, poly_input, outcome, l2_penalty):
    polynomial_data = polynomial_sframe(data[poly_input], degree)
    features = polynomial_data.column_names()
    polynomial_data[outcome] = data[outcome]
    regression_model = graphlab.linear_regression.create(polynomial_data, target=outcome, features=features,
                                                         l2_penalty=l2_penalty, validation_set=None, verbose=False)
    return regression_model

# 15 degree polynomial

sales = graphlab.SFrame('kc_house_data.gl/')
sales = sales.sort(['sqft_living', 'price'])
outcome = 'price'
polynomial_data = polynomial_sframe(sales['sqft_living'], 15)
features = polynomial_data.column_names()
polynomial_data[outcome] = sales[outcome]
l2_small_penalty = 1e-5
model_15 = graphlab.linear_regression.create(polynomial_data, target=outcome, features=features, l2_penalty=l2_small_penalty,
                                             validation_set=None, verbose=False)
print "coefficients of model are ", model_15.get('coefficients')

sales_12, sales_34 = sales.random_split(.5, seed=0)
set_1, set_2 = sales_12.random_split(.5, seed=0)
set_3, set_4 = sales_34.random_split(.5, seed=0)

# Fitting 15 degree polynomial on set_<1,2,3,4>


sets = [set_1, set_2, set_3, set_4]
i = 1
for set in sets:
    l2_small_penalty_1 = 1e-5
    set_data = graphlab.SFrame()
    set_data = polynomial_sframe(set["sqft_living"], 15)
    set_data_features = set_data.column_names()
    set_data['price'] = set['price']
    model = graphlab.linear_regression.create(set_data, target='price', features=set_data_features, validation_set=None,
                                              l2_penalty=l2_small_penalty_1, verbose=False)
    print "coefficients for set number ", i, model.get('coefficients').print_rows(20, 20)
    # plt.figure(i)
    # plt.plot(set_data['power_1'], set_data['price'], '.',
    #          set_data['power_1'], model.predict(set), '-')
    i = i + 1

for set in sets:
    l2_large_penalty = 1e5
    set_data = graphlab.SFrame()
    set_data = polynomial_sframe(set["sqft_living"], 15)
    set_data_features = set_data.column_names()
    set_data['price'] = set['price']
    model = graphlab.linear_regression.create(set_data, target='price', features=set_data_features, validation_set=None,
                                              l2_penalty=l2_large_penalty, verbose=False)
    print "coefficients for set number ", i, model.get('coefficients').print_rows(20, 20)
    # plt.figure(i)
    # plt.plot(set_data['power_1'], set_data['price'], '.',
    #          set_data['power_1'], model.predict(set), '-')
    i = i + 1

# plt.show()

(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=1)

n = len(train_valid_shuffled)
k = 10  # 10-fold cross-validation

for i in xrange(k):
    start = (n * i) / k
    end = (n * (i + 1)) / k - 1
    print i, (start, end)

start = (n * 3) / k
end = ((n * (3 + 1)) / k - 1) + 1

validation4 = train_valid_shuffled[start: end]
print int(round(validation4['price'].mean(), 0))
train4 = train_valid_shuffled[0: start].append(train_valid_shuffled[end + 1:n])
print int(round(train4['price'].mean(), 0))


def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    n = len(data)
    validation_errors = []
    for i in xrange(k):
        start = (n * i) / k
        end = (n * (i + 1)) / k - 1
        validation_set = data[start: end + 1]
        trainning_set = data[0:start].append(data[end + 1:n])
        model = graphlab.linear_regression.create(trainning_set, target=output_name, features=features_list,
                                                  l2_penalty=l2_penalty, validation_set=None, verbose=False)
        validation_error = model.predict(validation_set) - validation_set[output_name]
        RSS = sum(validation_error * validation_error)
        validation_errors.append(RSS)

    return sum(validation_errors) / len(validation_errors)


l2_penality_list = np.logspace(1, 7, num=13)

lowest_penalty = l2_penality_list[0]
lowest_validation_error = 99999999999999

outcome = 'price'
polynomial_data_train_shuffled = polynomial_sframe(train_valid_shuffled['sqft_living'], 15)
features_train_shuffled = polynomial_data_train_shuffled.column_names()
polynomial_data_train_shuffled[outcome] = train_valid_shuffled[outcome]

for l2_penality in l2_penality_list:
    validation_error = k_fold_cross_validation(10, l2_penality, polynomial_data_train_shuffled, outcome, features)
    if validation_error < lowest_validation_error:
        lowest_penalty = l2_penality
        lowest_validation_error = validation_error
    plt.scatter(l2_penality, validation_error)

print "lowest validation error is ", lowest_validation_error, "lowest penalty is ", lowest_penalty
plt.xscale('log')
plt.yscale('log')
plt.show()

model_with_lowest_penalty = graphlab.linear_regression.create(train_valid, target='price', features=['sqft_living'],
                                                              l2_penalty=lowest_penalty, validation_set=None,
                                                              verbose=False)

prediction_error = model_with_lowest_penalty.predict(test) - test['price']
rss = (prediction_error * prediction_error).sum()
print rss
