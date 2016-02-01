#Contest URL - https://www.hackerearth.com/machine-learning-india-hacks-2016/machine-learning/will-bill-solve-it/

import pandas as pd
import matplotlib
from pandas import DataFrame, Series
import numpy as np
import time
import os
from tqdm import *
from xgboost.sklearn import XGBClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
 
np.random.seed(0)

def ac_or_not(row):
    if row['solved_status'] == 'SO' and row['result'] =='AC':
        return 1
    else:
        return 0

def tag_or_not(row, tag):
    if row['tag1'] == tag or row['tag2'] == tag or row['tag3'] == tag or row['tag4'] == tag or row['tag5'] == tag:
        return 1
    else:
        return 0
    
def skilled_or_not(row, skill):
    for lang in row['skills'].split("|"):
        if lang == skill:
            return 1
    return 0

def process_skills(df):
    df['skills'] = df['skills'].replace('Java \(openjdk 1\.7\.0_09\)', 'Java', regex=True)
    df['skills'] = df['skills'].replace('C\+\+ \(g\+\+ 4\.8\.1\)', 'C++', regex=True)
    df['skills'] = df['skills'].replace('JavaScript\(Node\.js\)', 'JavaScript', regex=True)
    df['skills'] = df['skills'].replace('JavaScript\(Rhino\)', 'JavaScript', regex=True)
    df['skills'] = df['skills'].replace('Python 3', 'Python', regex=True)
    return df  

def process_tags(df):
    df = df.replace('Ad-hoc', 'adhoc', regex=True)
    df = df.replace('Ad-Hoc', 'adhoc', regex=True)
    df = df.replace('Basic-Programming', 'Basic Programming', regex=True)
    df = df.replace('Data-Structures', 'Data Structures', regex=True)
    df = df.replace('Priority-Queue', 'Priority Queue', regex=True)
    df = df.replace('cake-walk', 'Very Easy', regex=True)
    return df
    
all_unique_tags = set()
all_unique_skills = set()

def set_unique_skills(df):
    unique_skills_str = set(pd.unique(df['skills']))
    for skills_str in unique_skills_str:
        skills_list = skills_str.split("|")
        for skill in skills_list:
            all_unique_skills.add(skill)

def set_unique_tag_values(df):
    unique_tag = set(pd.unique(df['tag1']))
    unique_tag = unique_tag.union(pd.unique(df['tag2']))
    unique_tag = unique_tag.union(pd.unique(df['tag3']))
    unique_tag = unique_tag.union(pd.unique(df['tag4']))
    unique_tag = unique_tag.union(pd.unique(df['tag5']))
    unique_tag = [x for x in unique_tag if str(x) != 'nan']
    global all_unique_tags
    all_unique_tags = all_unique_tags.union(unique_tag)

def extract_skills_feature(df):
    for skill in tqdm(all_unique_skills):
        df[skill] = 0
        df[skill] = df.apply(lambda row: skilled_or_not(row, skill), axis = 1)
    del df['skills']
    return df

def extract_tags_feature(df):
    for tag in tqdm(all_unique_tags):
        df[tag] = 0
        df[tag] = df.apply(lambda row: tag_or_not(row, tag), axis = 1) 
    del df['tag1']
    del df['tag2']
    del df['tag3']
    del df['tag4']
    del df['tag5']
    return df

submissions = pd.read_csv("train/submissions.csv")
submissions['AC']  = submissions.apply(lambda row: ac_or_not(row), axis = 1) 

submissions = submissions.drop_duplicates(['user_id', 'problem_id', 'AC'])
#if there are both AC = 0,1 for a (user, problem) taken only AC=1, discard AC=0. 
clean_submissions = submissions.groupby(['user_id', 'problem_id']).AC.transform(max)
clean_submissions = submissions[submissions.AC == clean_submissions]

users = pd.read_csv("train/users.csv")
problems = pd.read_csv("train/problems.csv")
user_n_submissions = pd.merge(clean_submissions, users, on='user_id')
merged_data = pd.merge(user_n_submissions, problems, on='problem_id')

def skills_count_func(row):
    skillsstr = row['skills']
    if str(skillsstr) != 'nan':
        skills = skillsstr.split("|")
        return len(skills)
    else:
        return 0

def tag_count_func(row):
    count = 0
    if str(row['tag1']) != 'nan':
        count+=1
    if str(row['tag2']) != 'nan':
        count+=1
    if str(row['tag3']) != 'nan':
        count+=1
    if str(row['tag4']) != 'nan':
        count+=1
    if str(row['tag5']) != 'nan':
        count+=1
    return count
    
merged_data['skills_count'] = merged_data.apply(lambda row: skills_count_func(row), axis = 1) 
merged_data['tag_count'] =  merged_data.apply(lambda row: tag_count_func(row), axis = 1) 


#delete featured which are not available for test data
del merged_data['solved_status']
del merged_data['result']
del merged_data['language_used']
del merged_data['execution_time']
merged_data.to_csv("train/merged_train.csv", index=False)
merged_data = process_skills(merged_data)
set_unique_skills(merged_data)
merged_data = process_tags(merged_data)
set_unique_tag_values(merged_data)


#label encoding for user_type and level
le_usertype = preprocessing.LabelEncoder()
le_usertype.fit(merged_data['user_type'])
merged_data['user_type'] = le_usertype.transform(merged_data['user_type'])
le_level = preprocessing.LabelEncoder()
le_level.fit(merged_data['level'])
merged_data['level'] = le_level.transform(merged_data['level'])
##
set_unique_skills(merged_data)
merged_data = extract_skills_feature(merged_data)
set_unique_tag_values(merged_data)
merged_data = extract_tags_feature(merged_data)

Y = merged_data['AC']
del merged_data['AC']
X = merged_data
Y = Y.reshape(421975,1)

test_users = pd.read_csv("test/users.csv")
test_problems = pd.read_csv("test/problems.csv")
test_submissions = pd.read_csv("test/test.csv")
test_merged = pd.merge(test_submissions, test_users, on='user_id')
test_merged = pd.merge(test_merged, test_problems, on='problem_id')
test_merged.to_csv("test/test_merged.csv", index=False)
test_merged['user_type'] = le_usertype.transform(test_merged['user_type'])
#remove noise 'O' in levels
test_merged['level'] = test_merged['level'].replace('O', np.nan)
test_merged['level'] = le_level.transform(test_merged['level'])
ids = test_merged['Id']
del test_merged['Id']
test_merged['skills_count'] = test_merged.apply(lambda row: skills_count_func(row), axis = 1) 
test_merged['tag_count'] =  test_merged.apply(lambda row: tag_count_func(row), axis = 1) 
test_merged = process_skills(test_merged)
test_merged = extract_skills_feature(test_merged)
test_merged = process_tags(test_merged)
test_merged = extract_tags_feature(test_merged)
#===============
#Model Here
'''
top_features
['solved_count_y', 'solved_count_x', 'error_count', 'attempts', 'accuracy', 'problem_id', 'user_id', 'C++', 'rating',
 'C', 'level', 'tag_count', 'Java', 'skills_count', 'user_type', 'Python', 'Algorithms', 'Math', 'Number Theory',
 'Binary Search', 'Dynamic Programming', 'Implementation', 'PHP', 'adhoc', 'C#', 'Pascal', 'Very Easy', 'Trees',
 'Sorting', 'Bitmask', 'Perl', 'Brute Force', 'Binary Search Tree', 'Scala', 'BFS', 'Ruby', 'Text', 'Bit manipulation',
 'HashMap', 'Data Structures', 'Go', 'Basic Programming', 'Heap', 'Hashing', 'JavaScript', 'Greedy', 'Objective-C',
 'Graph Theory', 'Stack', 'String Algorithms', 'Bellman Ford', 'Simple-math', 'Modular arithmetic', 'Floyd Warshall',
 'Rust', 'Game Theory', 'Shortest-path', 'Geometry', 'Simulation', 'Divide And Conquer', 'Probability']
'''
top_features = np.array(['solved_count_y', 'solved_count_x', 'error_count', 'attempts', 'accuracy', 'problem_id', 'user_id', 'C++', 'rating',
 'C', 'level', 'tag_count', 'Java', 'skills_count', 'user_type', 'Python', 'Algorithms', 'Math', 'Number Theory',
 'Binary Search', 'Dynamic Programming', 'Implementation', 'PHP', 'adhoc', 'C#', 'Pascal', 'Very Easy', 'Trees',
 'Sorting', 'Bitmask', 'Perl', 'Brute Force', 'Binary Search Tree', 'Scala', 'BFS', 'Ruby', 'Text', 'Bit manipulation',
 'HashMap', 'Data Structures', 'Go', 'Basic Programming', 'Heap', 'Hashing', 'JavaScript', 'Greedy', 'Objective-C',
 'Graph Theory', 'Stack', 'String Algorithms', 'Bellman Ford', 'Simple-math', 'Modular arithmetic', 'Floyd Warshall',
 'Rust', 'Game Theory', 'Shortest-path', 'Geometry', 'Simulation', 'Divide And Conquer', 'Probability'])

#y_test = xgb.predict(test_merged)
clf = xgb.XGBClassifier(objective='reg:logistic', nthread=4, seed=0, max_depth=6, learning_rate=0.005,
                             n_estimators=1000, gamma = 2)  
clf.fit(X[top_features],Y))

clf2 = xgb.XGBClassifier(objective='reg:logistic', nthread=4, seed=0, max_depth=6, learning_rate=0.004,
                             n_estimators=1200, gamma = 10)  
clf2.fit(X[top_features],Y)


#==================
pred_test_y = clf.predict(test_merged[top_features])
test_y_clf2 = clf2.predict(test_merged[top_features])

#ensembling, adjust weights here
ensembled_y = pred_test_y*.7 + test_y_clf2*.3
test_y = np.array([(pred > .5).astype(int) for pred in ensembled_y ])

test_ans = pd.DataFrame({'Id':ids, 'solved_status': test_y})
test_ans.to_csv('logistic.csv', index=False)
