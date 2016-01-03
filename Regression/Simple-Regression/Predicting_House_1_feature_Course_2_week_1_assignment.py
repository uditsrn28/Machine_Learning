__author__ = 'udit'

import graphlab

sales = graphlab.SFrame('kc_house_data.gl/')
train_data, test_data = sales.random_split(.8, seed=0)


def simple_linear_regression(input_feature, output):
    # compute the sum of input_feature and output
    input_feature_sum = input_feature.sum()
    output_sum = output.sum()

    # compute the product of the output and the input_feature and its sum
    output_input_feature_product = (input_feature * output)
    output_input_feature_product_sum = output_input_feature_product.sum()

    # compute the squared value of the input_feature and its sum
    input_feature_square = input_feature * input_feature
    input_feature_square_sum = input_feature_square.sum()

    # use the formula for the slope
    # slope = ((( sigma(output*input) - ((sigma(output) * sigma(input) / len(input))) /
    #            (sigma(square(input)) - (sigma(input)*sigma(input) / len(input)))
    slope = ((output_input_feature_product_sum - (input_feature_sum * output_sum / len(input_feature))) / (
        input_feature_square_sum - ((input_feature_sum * input_feature_sum) / len(input_feature))))

    # use the formula for the intercept
    intercept = ((output_sum / len(input_feature)) - (slope * (input_feature_sum / len(input_feature))))

    return (intercept, slope)


input_feature = train_data['sqft_living']
output = train_data['price']

intercept, slope = simple_linear_regression(input_feature, output)
print "intercept of linear model with sqft_living as model is ", intercept
print "slope of linear model with sqft_living as model is ", slope


def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    predicted_values = intercept + (slope * (input_feature))

    return predicted_values

# predicted value for 2650 sq. ft. house
predicted_value_2650 = get_regression_predictions(2650, intercept, slope)
print "Predicted Valye for house with square feet 2650 is " , predicted_value_2650

def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    predicted_values = input_feature + (slope * (input_feature))
    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    rss = output - predicted_values
    # square the residuals and add them up
    rss_square = (rss * rss)
    rss_square_sum = rss_square.sum()
    return (rss_square_sum)

def inverse_regression_predictions(output, intercept, slope):
    estimated_input = ((output - intercept) / slope)
    return (estimated_input)


sqft_8000000= inverse_regression_predictions(800000, intercept, slope)
print "Sqft of house with price 800000 is ", sqft_8000000

input_bedrooms = train_data['bedrooms']
intercept_bedroom, slope_bedroom = simple_linear_regression(input_bedrooms, output)
print "intercept of linear model with feature as bedrooms" , intercept_bedroom
print "slope of linear model with feature as bedrooms" , slope_bedroom

rss_sqft = get_residual_sum_of_squares(input_feature,output,intercept,slope)
rss_bedrooms = get_residual_sum_of_squares(input_bedrooms,output,intercept_bedroom,slope_bedroom)

print "residual sum of squares of linear model with features as sqft_living is " , rss_sqft
print "residual sum of squares of linear model with features as bedrooms is " , rss_bedrooms