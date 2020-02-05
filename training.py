import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
import pickle

df = pd.read_csv('loans-23k.csv', skipinitialspace=True)
df.head()

df.drop(columns=['id', 'member_id'], inplace=True)
x = df.drop(columns = ['grade'])
y = df['grade']
numeric_features = ['annual_inc', 'loan_amnt', 'int_rate']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_features = ['home_ownership','purpose', 'term']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

rf= RandomForestClassifier(n_estimators = 1600, n_jobs = -1,random_state =50, min_samples_leaf = 2, criterion='entropy')
xgb = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1)
model = VotingClassifier(estimators=[('rf', rf), ('kn', xgb)], voting='hard')

clf = Pipeline(steps=[('preprocessor', preprocessor),
                     ('classifier', model)])
clf.fit(x_train,y_train)
#print(clf.score(x_test, y_test))

pickle.dump(clf,open('model.pkl', 'wb'))
