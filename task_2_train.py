import pandas as pd
import pickle
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train pipeline')
    parser.add_argument('--data_name', help='Train file name', required=True)
    parser.add_argument('--model_name', help='Model file name', required=True)
    args = vars(parser.parse_args())

    df = pd.read_csv(f'data/{args["data_name"]}.csv')
    # Used only 6 feature, check reason eda file conclusion.
    x_train, x_test, y_train, y_test = train_test_split(
        df[['6']], df['target'], test_size=0.2)

    # Build polynomial feature
    x_train **= 2
    x_test **= 2

    # Initialize and fit model
    # Also can be included fit_intercept=False,
    # depending on result on hidden test.
    # In test on train data this gives a little higher error.
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    rmse = mean_squared_error(y_test, predictions, squared=False).round(3)

    # Train on all available data
    model.fit(df[['6']] ** 2, df['target'])

    pickle.dump(model, open(f'models/{args["model_name"]}', 'wb'))

    print(f'\nModel trained and saved successfully! RMSE on test: {rmse}.')
