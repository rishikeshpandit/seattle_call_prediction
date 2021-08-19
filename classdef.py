import numpy as np
import pandas as pd


class DataPrep:

    def __init__(self, dataframe):
        self.my_dataframe =  dataframe

    def prep_weather(self, trim=False, startdate=None, enddate=None):
        self.dt_iso = pd.to_datetime(self.dt_iso, format="%Y-%m-%d %H:%M:%S %z UTC")    #convert to datetime format 
        self.rename(columns={"dt_iso": "utc_time"}, inplace=True)                     
        self.insert(1, "local_time", self["utc_time"].dt.tz_convert('US/Pacific'))      #convert UTC to Seattle TZ
        self.set_index('local_time', inplace=True)                                    #Set DT index 
        self.sort_index(inplace=True)
        self.index = self.index.tz_localize(None)                                       #Remove TZ stamp from local time
        self = self.filter(["temp_max", "wind_speed", "weather_id"])                    #Select imp features needed further
        if trim == True:
            return self[(self.index >= startdate) & (self.index < enddate)]               #trim to datetime period if required
        else:
            return self

    # To Preprocess Original call dataframe and extract needed features
    def prepare_call_data(self, trim=False, startdate=None, enddate=None):
        self.rename(columns={"datetime":"local_time"}, inplace=True)
        self.local_time = pd.to_datetime(self.local_time, infer_datetime_format=True)               #infer datetime for speeding up parsing
        self.loc[:, "day"] = pd.to_datetime(self.local_time.dt.strftime("%Y-%m-%d"))                #add day resolution column
        self.loc[:, "hour"] = pd.to_datetime(self.local_time.dt.strftime("%Y-%m-%d %H:00:00"))      #add hour resolution column
        self.set_index('local_time', inplace=True)                                                #set DT index
        self.sort_index(inplace=True)
        if trim == True:
            startdate = pd.Timestamp(startdate)
            enddate = pd.Timestamp(enddate)
            return self.filter(["day", "hour"])[(self.index >= startdate) & (self.index < enddate)]   #trim to DT period if required
        else:
            return self.filter(["day", "hour"])


    def check_duplicate_timestamps(self):
        if self.index.duplicated().any():
            print(f"Found {self[self.index.duplicated()].shape[0]} duplicates, keeping latest row!")
            return self[~self.index.duplicated(keep='last')]
        else: 
            print(f"No duplicates found in dataframe! Proceed...")
            return self


    def count_hourly_calls(self):
        h = self.hour.value_counts()
        h.sort_index(inplace=True)
        h.index.name = 'hour'
        return pd.DataFrame(h).rename(columns={"hour":"counts"})

    def count_daily_calls(self):
        d = self.day.value_counts()
        d.sort_index(inplace=True)
        d.index.name = 'day'
        return pd.DataFrame(d).rename(columns={"day":"counts"})

    def fill_missing_hours(self):
        N_hours = pd.date_range(start=self.index.min(), end=self.index.max(), freq='H')
        self_reindexed = self.reindex(N_hours)
        n_hours = N_hours.difference(self.index).values.shape[0]
        if n_hours > 0:
            print(f"Missing {n_hours} hours of data. Interpolating!")
            self = self_reindexed.interpolate(method = 'linear')
            return self
        else:
            print(f"No missing hours in data! Proceed...")
            return self


    def join_weather_to_calls(self, dataframe):
        if ((self.shape[0] == dataframe.shape[0]) and 
        (self.index.min() == dataframe.index.min()) and 
        (self.index.max() == dataframe.index.max())):
            self = self.join(dataframe[["temp_max", "wind_speed", "weather_id"]], how="right")
            print("DateTimeIndices are Exact Match!")
            return self
        else:
            print("Provide Dataframes with EXACT Matching DateTimeIndices")
            
            
    def add_seasonality(self):
        
        self["D_before"] = self.counts.rolling(24).sum()
        self["W_before"] = self.counts.rolling(7*24).sum()
        self["M_before"] = self.counts.rolling(30*24).sum()
        self["Y_before"] = self.counts.rolling(365*24).sum()
        
        return self
        
            
    def create_features(self, label, weather=False):
        self['date'] = self.index
        self['hour'] = self['date'].dt.hour
        self['dayofweek'] = self['date'].dt.dayofweek
        self['dayofyear'] = self['date'].dt.dayofyear
        
        X = self[["D_before", "W_before", "M_before", "Y_before", 
                'hour', 'dayofweek', 'dayofyear']].copy()
        
        y = self[label].copy()
        
        if weather:
            X_weather =  self[["D_before", "W_before", "M_before", "Y_before", 
                'hour', 'dayofweek', 'dayofyear', 
                            'temp_max', 'wind_speed', 'weather_id']].copy()
        
            return X_weather, y
        
        return X, y


    def train_test_split_df(self, split_date):
        
        train = self.loc[self.index < split_date].copy()
        test = self.loc[self.index >= split_date].copy()
        
        return train, test

