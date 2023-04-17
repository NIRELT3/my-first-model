import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn import datasets, preprocessing
from sklearn import linear_model
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

def print_hi(name):
    print(f'Hi, {name}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # load the Boston House Prices dataset
    boston_dataset = load_boston()

    # create a pandas DataFrame from the dataset
    boston_df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

    # add the target variable (MEDV) to the DataFrame
    boston_df['MEDV'] = boston_dataset.target

    # create plotbox panels
    # fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
    # index = 0
    # axs = axs.flatten()
    # for k, v in boston_df.items():
    #     sns.boxplot(y=k, data=boston_df, ax=axs[index])
    #     index += 1
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    #
    # plt.figure(figsize=(20, 10))
    # sns.heatmap(boston_df.corr().abs(), annot=True)
    # # save the DataFrame to a CSV file
    # #boston_df.to_csv('boston_house_prices1.csv', index=False)
    # #print(boston_df.describe())
    #
    # # Let's scale the columns before plotting them against MEDV
    # min_max_scaler = preprocessing.MinMaxScaler()
    # column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
    # x = boston_df.loc[:, column_sels]
    # y = boston_df['MEDV']
    # x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
    # fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
    # index = 0
    # axs = axs.flatten()
    # for i, k in enumerate(column_sels):
    #     sns.regplot(y=y, x=x[k], ax=axs[i])
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    # plt.show()

    # Load the Boston Housing Prices dataset
    boston = load_boston()
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    boston_df['PRICE'] = boston.target

    # Split the data into training and testing sets
    X = boston_df.drop('PRICE', axis=1)
    y = boston_df['PRICE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Fit a linear regression model to the training data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Use the trained model to make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate the mean squared error of the model
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error: ", mse)
    print("the linear model is: ", model.coef_)

    # Fit a Ridge regression model to the training data
    ridge = Ridge(alpha=0.5)
    ridge.fit(X_train, y_train)

    # Use the trained model to make predictions on the testing data
    y_pred = ridge.predict(X_test)

    # Calculate the mean squared error of the model
    mse = mean_squared_error(y_test, y_pred)
    print("the linear model is: ", ridge.coef_)
    print("Mean Squared Error: ", mse)

