from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class PercentageProcessing(BaseEstimator, TransformerMixin):
    def __init__(self, feature_list):
        print('__init__ called...')
        self.feature_list = feature_list
    
    def fit(self, X, y=None):
        print('fit() called...')
        return self
    
    def transform(self, X, y=None):
        print('transform() called...')               
        X_ = X.copy()
#         print(self.feature_list)
        for feature in self.feature_list:
            X_.iloc[:,feature] = pd.to_numeric(X_.iloc[:,feature].str.replace('%', '', regex = True)).apply(lambda x: self.percentage_cat(x))
        return X_
    
    def percentage_cat(self, percentage_cat):
        if percentage_cat == 100:
            return '100'
        elif percentage_cat >= 95 and percentage_cat <100:
            return 'over_95'
        elif percentage_cat >= 90 and percentage_cat <95:
            return 'over_90'
        elif percentage_cat >= 70 and percentage_cat <90:
            return 'over_70'
        else:
            return 'below_70'

class ItemCount(BaseEstimator, TransformerMixin):
    def __init__(self, feature_list):
        print('__init__ called...')
        self.feature_list = feature_list
        
    def fit(self, X, y=None):
        print('fit() called...')
        return self
    
    def transform(self, X, y=None):
        print('transform() called...')
        X_ = X.copy()
        
        for feature in self.feature_list:
#             print(X_.columns[feature])
            X_.iloc[:,feature] = X_.iloc[:,feature].apply(lambda x: len(x.replace('[','').replace(']','').split(',')))
        
        return X_

class BathroomText(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        print('__init__ called...')
        self.feature = feature
        
    def fit(self, X, y=None):
        print('fit() called...')
        return self
    
    def transform(self, X, y=None):
        print('transform() called...')
        X_ = X.copy()
        
        X_['bathroom_count'] = X_.iloc[:,self.feature].apply(lambda x: x.replace('Shared half-bath', '0.5 shared').replace('Half-bath', '0.5 bath').str.split(' '))
        X_['bathroom_count'] = X_['bathroom_count'].str[0].astype('float')
#         X_['bathroom_shared_flag'] = X_.iloc[:,self.feature].apply(lambda x: self.shared(x))
#         X_['bathroom_shared_flag'] = self.shared(X_)
        X_ = X_.drop(X_.columns[self.feature], axis=1)
        return X_
 
 