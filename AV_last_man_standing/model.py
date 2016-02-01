#Contest URL - http://datahack.analyticsvidhya.com/contest/last-man-standing
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas_profiling

train = pd.read_csv('Train_Fyxd0t8.csv')
test = pd.read_csv('Test_C1XBIYq.csv')

#feature Engineering
def diff_week(df):
    df['diff_weeks'] = df['Number_Doses_Week'] - df['Number_Weeks_Used']
    df = df.drop(['Number_Doses_Week'], axis =1 )
    df = df.drop(['Number_Weeks_Used'], axis =1)
    return df

def crop_type_sq(df):
    df['Crop_Type'] = df['Crop_Type'] * df['Crop_Type']
    return df

def season_sq(df):
    df['Season'] = df['Season'] * df['Season'] * df['Season']
    return df

def rescale_crop(df):
    df['Crop_Type'] = df['Crop_Type'] + 1
    return df

def rescale_soil(df):
    df['Soil_Type'] = df['Soil_Type'] + 1
    return df

def soil_crop_pesticide_combo(df):
    df['Crop_Type'] = df['Crop_Type'] + 0
    df['Soil_Type'] = df['Soil_Type'] + 2
    df['Pesticide_Use_Category'] = df['Pesticide_Use_Category'] + 4
    #for generating unique number for each combination.
    df['scp_combo'] = df['Soil_Type'] * df['Season'] * df['Pesticide_Use_Category']
    df = df.drop( 'Crop_Type' , axis = 1)
    df = df.drop( 'Soil_Type', axis = 1)
    df = df.drop( 'Pesticide_Use_Category', axis = 1)
    return df

#train = diff_week(train)
#train = crop_type_sq(train)
#train = season_sq(train)
train = rescale_crop(train)
train = rescale_soil(train)
train = soil_crop_pesticide_combo(train)

#test = diff_week(test)
#test = crop_type_sq(test)
#test = season_sq(test)
test = rescale_crop(test)
test = rescale_soil(test)
test = soil_crop_pesticide_combo(test)


Y = train['Crop_Damage']
train = train.drop('Crop_Damage', axis = 1)
train = train.drop('ID', axis = 1)
Ids = test['ID']
test = test.drop('ID', axis = 1)


clf = xgb.XGBClassifier(objective='reg:logistic', nthread=4, seed=0, max_depth=7, learning_rate=0.005, n_estimators=1000)
#, max_depth=7, learning_rate=0.005, n_estimators=1000, gamma = 2
clf.fit(train, Y)

clf.score(train, Y)

test_y = clf.predict(test)
test_ans = pd.DataFrame({'ID':Ids, 'Crop_Damage': test_y})
order = ['ID', 'Crop_Damage']
test_ans[order].to_csv('out.csv',index= False)