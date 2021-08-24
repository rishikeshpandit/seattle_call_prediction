#! /usr/bin/env python3

import pandas as pd

import configparser
import classdef as cd

def dataprep(df1, df2=None):
# Prepare Data
    config = configparser.ConfigParser()
    # Check if config file exists
    config.read("options.ini")
    start_date = config.get('datetime', 'start_date') #This is the actual start date of main dataframe
    end_date = config.get('datetime', 'end_date') #This will be end date of main dataframe
    weather = config.getboolean('boolvals', 'weather') # If true, weather info will be joint to call timeseries dataframe for training

    # -1Y for Data Prep (To add backtracing Seasonal Data)
    kick_start = (pd.Timestamp(start_date) - 
                pd.Timedelta("365D", format='%Y-%m-%d')
                ).strftime('%Y-%m-%d') 


    # Trimming, Normalizing and Preprocessing the Dataframe
    df1_clean = cd.DataPrep.prepare_call_data(
        df1, trim=True, 
        startdate=kick_start, 
        enddate=end_date
    )


    # Counting Call Volume for hourly and daily resolution for cleaned dataframe
    H = cd.DataPrep.count_hourly_calls(df1_clean) #hourly counts
    #D = cd.DataPrep.count_daily_calls(cdf_clean)  #daily counts

    # Filling missing hours using linear interpoaltion
    H = cd.DataPrep.fill_missing_hours(H)

    if weather==True:
        # Cleaning and Preparing Dataframe
        df2_clean = cd.DataPrep.prep_weather(
            df2, trim=True, 
            startdate=kick_start, 
            enddate=end_date
        )

        # Checking for Duplicate Timestamps in Cleaned Frame (Also deletes DST duplicates)
        w = cd.DataPrep.check_duplicate_timestamps(df2_clean)
        # Correcting for missing DST hours
        W = cd.DataPrep.fill_missing_hours(w)


        # Preparing Training Data:
        joint = cd.DataPrep.join_weather_to_calls(H, W) #DatetimeIndex of both dataframes should be identical

        main_df = cd.DataPrep.add_seasonality(joint) # Adding Seasonality Indicators to joint weather-call dataframe
    
    else:

        main_df = cd.DataPrep.add_seasonality(H) # Adding seasonality indicators to only call dataframe


    df = main_df[main_df.index >= start_date].copy()

    main_df['hour_res'] = main_df.index
    main_df.to_csv("to_train_df.csv", index=False)

    return df


if __name__ == "__main__":
    print("Running Prep Alone")
else:
    print("Preparing Main Dataframe for Training")