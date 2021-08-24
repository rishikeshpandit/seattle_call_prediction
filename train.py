import pandas as pd
from xgboost import XGBRegressor
import configparser
import classdef as cd
import pickle

import numpy as np
from sklearn.metrics import mean_squared_error as msq




def train(df):
    config = configparser.ConfigParser()
    # Check if config file exists
    config.read("options.ini")
    end_date = config.get('datetime', 'end_date') #This will be end date of main dataframe
    weather = config.getboolean('boolvals', 'weather')
    

    #For separating train and test dataframes before and after this date: Default (enddate -1Y )
    split_date = (pd.Timestamp(end_date) - 
                pd.Timedelta("365D", format='%Y-%m-%d')
                ).strftime('%Y-%m-%d') 
    
    
     #Creating Train and Test Dataframes
    train, test = cd.DataPrep.train_test_split_df(df, split_date) 

    #Preparing Model Dataframe
    X_train, Y_train = cd.DataPrep.create_features(train, label='counts', weather=weather)
    X_test, Y_test = cd.DataPrep.create_features(test, label='counts', weather=weather)

    #Setting Model Params
    model = XGBRegressor(n_estimators=1000)

    #Model Fit
    model.fit(
        X_train, Y_train, 
        eval_set=[(X_train, Y_train), (X_test, Y_test)], 
        early_stopping_rounds=50)

    test_predict =  model.predict(X_test)
    train_predict =  model.predict(X_train)

    train_score = np.sqrt(msq(Y_train.values, train_predict))
    test_score = np.sqrt(msq(Y_test.values, test_predict))
    
    print(f"Train RMSE Score: {train_score:.2f}")
    print(f"Train RMSE Score: {test_score:.2f}")

    

    xdf = X_test.copy()
    xdf["hour_res"] = xdf.index
    xdf.to_csv("to_predict_df.csv", index=False)

    ydf = pd.DataFrame(Y_test).copy()
    ydf["hour_res"] = ydf.index
    ydf.to_csv("counts_df.csv", index=False)

    # save model to file
    pickle.dump(model, open("pima.pickle.dat", "wb"))

    return model

if __name__ == "__main__":
    print("Running Train Alone")
else:
    print("Importing XGBRegressor Training Module")
