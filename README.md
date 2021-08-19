## Machine Learning Assessment: Seattle Call Prediction Problem
**Main Goal:** To predict call volume (in counts) per hour of the day for each day in last 1 year (based on previous call data)

### Plan
The given problem is an example of timeseries data analysis for which one must take into account some important points before selecting the model for finding the solution and planning the prediction approach:
 - A model that remembers the variables historical movement through time would give out the best solution. This means that one cannot split the to-be-modelled data in random training and testing samples as each step contains the periodic history
 - ML methods, e.g. saving best accurate model and using it forever, that are traditionally used for classification problems cannot be used for Timeseries Data as the variable is time-dependent and a non-updated model with latest data may not work in this case
From the above points one can use following approaches to predict the call volume:

***Possible Approaches***
1. FB Prophet or SARIMA
3. SVM or Random Forests
3. LSTM
4. Regression, Boosted Trees Regression (Suggested)

For this program, I have chosen the suggested approach of using Gradient Boosting Regression technique for which I have used `xgboost` library.

## Requirements
The program is a `python 3.x` code and requires following python libraries to be existing or pre-installed before running the program
 - `numpy`
 - `matplotlib`
 - `sklearn`
 - `pandas`
 - `xgboost`
 - `sodapy`
 - `pickle-mixin`

You may install them using `pip install <library>` or `pip3 install <library>`
 
### Enclosed Data:

1. **Data Files**:
    - `seattle_weather.csv`: Weather Data for Seattle
    - `download_call_data.py`: A script to download `seattle_sos.csv`
    - `seattle_sos.csv`: 911 call data to Fire Department Service of the Seattle City - Downloaded from: https://data.seattle.gov/Public-Safety/Seattle-Real-Time-Fire-911-Calls/kzjm-xkqj

2. **Program Files**:
    - `main.py`: main executable program file which outputs final prediction dataframe `to_plot_df.csv` with actual counts and predicted counts by the model; also gives out `feature_importance.png` indicating which features were split out the most during the training procedure (very important for improving the accuracy of the model)
    - `options.ini`: config file used to define start and end dates for training dataframe and prediction
    - `classdef.py`: contains DataPrep class and miscellaneous functions to prepare data
    - `dataprep.py`: main module to prepare data, takes input two dataframes `seattle_sos.csv` and `seattle_weather.csv`; outputs joint dataframe `to_train_df.csv` to be used for training the model
    - `train.py`: training module that takes in one to-be-trained dataframe `to_train_df.csv`, splits the dataframe into train-test dataframes and creates features and labels for model to-be-trained. Trains the model using boosted decision trees regression method i.e. `XGBRegressor` from `xgboost` library; Spits out `model` after saving model to `pima.pickle.dat` and dataframe to-be-predicted to `to_predict_df.csv`
    - `plot.py`: imports output dataframe from main.py `to_plot_df.csv` containing predicted and actual counts created by `main.py`; gives out `prediction_year.png`, `prediction_month.png` and `daily_weekly_comparison.png` (backtesting) with prediction vs accuracy values; plots are self-explanatory
    - `cleanup.sh` cleans up all output files except data input to program

3. **Extras**: 
    -`investigation.ipynb`: contains all data visualisation plots of raw data, that can be useful to know more about the dataset and helps to derive the best solution for the problem. **NOTE:** Execute cells in order for a successful run until the end

## Setting-up Parameters
To use this program, a priliminary analysis is needed. One must carefully investigate the Input Data, Datatypes, Weather Indicators, Timezone etc. I have summarised my full data visualisation plots in the notebook file

### Checking the Input Dataframes
Input Dataframe to Model must be csv files with following columns:
- index: `["Date"]` : in hour resolution
- columns: `["Counts"]`, `["temp_max"]`, `["wind_speed"]`, `["weather_id"]`, `["D_before"]`, `["W_before"]`, `["M_before"]`, `["Y_before"]`

## How to Run the program
- First download Seattle Call Data and store in `seattle_sos.csv` using script `download_call_data.py`. You may directly run the executable. **Note: Make sure that you have all the libraries installed before executing any of the program files**
- Simply run the executable `main.py` with the command `./main.py` or `python3 main.py`
- After successful execution, run `plot.py` for creating final plot files
