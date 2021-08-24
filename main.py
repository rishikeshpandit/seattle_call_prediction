#! /usr/bin/env python3

import numpy as np
import pandas as pd
from pandas import read_csv
import pickle
import os
import configparser
#import prepdata
import dataprep
#import trainfile
import train
from sklearn.metrics import mean_squared_error as msq
import logging
logging.basicConfig(filename='run.log', format="%(levelname)s:%(message)s", encoding='utf-8', level=logging.INFO)
from xgboost import plot_importance
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 300})




def main():

    xdf = read_csv("to_predict_df.csv")
    xdf = xdf.set_index("hour_res")
    xdf.index = pd.to_datetime(xdf.index)

    ydf = read_csv("counts_df.csv")
    ydf = ydf.set_index("hour_res")
    ydf.index = pd.to_datetime(ydf.index)
    
    predict = model.predict(xdf)       #predicting counts using model file

    test_score = np.sqrt(msq(ydf.counts.values, predict))
    print(f"Prediction RMSE Score: {test_score:.2f}") #Printing Prediction RMSE to command_line



    # Plot feature importance
    fig, ax = plt.subplots(1)
    fig.set_figheight(6)
    fig.set_figwidth(16)
    ax = plot_importance(model, height=0.9)
    plt.savefig("plots/feature_importance.png")


    df = ydf.copy()
    preddf = pd.DataFrame({"counts": df.counts, "prediction": predict})
    preddf['hour_res'] = preddf.index
    preddf.to_csv("to_plot_df.csv", index=False)

   

if __name__ == "__main__":
    
    config = configparser.ConfigParser()
    # Check if config file exists
    config.read("options.ini")
    weather = config.getboolean('boolvals', 'weather')

    if os.path.isfile("pima.pickle.dat" and "to_predict_df.csv" and "counts_df.csv"):
        logging.info("****************  Loading Pre-trained Model! ******************")
        print("Pre-trained Model file exists! loading model from the file")
        # load model from file
        model = pickle.load(open("pima.pickle.dat", "rb"))

    elif os.path.isfile("to_train_df.csv"):
        logging.info("****************  Training Model From Input CSV ******************")
        print("Found to_train_df.csv, Make sure it is in the required format!")
        df = read_csv("to_train_df.csv")
        df = df.set_index("hour_res")
        df.index = pd.to_datetime(df.index)
        model = train.train(df)

    else:
        logging.info("****************  Preparing Data, Training Model and Predicting From Scratch ******************")
        print("Not Found previous files to load, Training Model from scratch")
        
        cdf = read_csv("seattle_sos.csv")
        if weather == True:
            wdf = read_csv("seattle_weather.csv")
            df = dataprep.dataprep(cdf, wdf)
        else:
            df = dataprep.dataprep(cdf)
        print("Data prep Successful, Stored Dataframe to to_train_df.csv")
        model = train.train(df)
        print("Model Training Successful, Stored Dataframe to to_predict_df.csv")

    main()
