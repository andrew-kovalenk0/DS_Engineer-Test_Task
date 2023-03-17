import pandas as pd
import pickle
import argparse

if __name__ == "__main__":
    # Console argument parser
    parser = argparse.ArgumentParser(description='Train pipeline')
    parser.add_argument('--data_name', help='Data to predict file name',
                        required=True)
    parser.add_argument('--result_name', help='Result file name',
                        required=True)
    parser.add_argument('--model_name', help='Model file name', required=True)
    args = vars(parser.parse_args())

    df = pd.read_csv(f'data/{args["data_name"]}.csv')

    # Used only 6 feature, check reason eda file conclusion.
    x_test = df[['6']]

    # Build polynomial feature
    x_test **= 2

    # Load model
    model = pickle.load(open(f'models/{args["model_name"]}', 'rb'))

    predictions = pd.DataFrame(model.predict(x_test), columns=['predictions'])

    predictions.to_csv(f'data/{args["result_name"]}.csv', index=False)

    print('\nDone!')
