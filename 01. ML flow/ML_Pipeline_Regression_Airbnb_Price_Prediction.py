#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# Pipeline related libraries
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


os.chdir("..")


# In[3]:


input_path = os.path.join(os.getcwd(),'02. Input') 


# In[4]:


# os.path.join(input_path,'airbnb_listings.csv')


# In[5]:


data = pd.read_csv(os.path.join(input_path,'airbnb_listings.csv'))


# In[6]:


data.shape


# In[7]:


data.head(3)


# In[8]:


data.columns


# In[9]:


keep_columns = [
       'host_response_time', 'host_response_rate', 'host_acceptance_rate','host_is_superhost', 
       #'host_listings_count', 'host_total_listings_count', 
       'host_verifications',
       'host_has_profile_pic', 'host_identity_verified', 'neighbourhood',
       # 'latitude', 'longitude', 
       'property_type', 'room_type', 'accommodates',
       'bathrooms_text', 'bedrooms', 'beds', 'amenities',
       #'minimum_nights', 'maximum_nights', 'minimum_minimum_nights',
       #'maximum_minimum_nights', 'minimum_maximum_nights',
       #'maximum_maximum_nights', 'minimum_nights_avg_ntm',
       #'maximum_nights_avg_ntm', 'calendar_updated', 
       #'has_availability','availability_30', 
       #'availability_60', 'availability_90', 'availability_365', 'calendar_last_scraped', 'number_of_reviews',
       'number_of_reviews_ltm', 'number_of_reviews_l30d', 
       #'first_review', 'last_review', 
       'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 
       #'license', 
       'instant_bookable', 'calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms', 
       #'reviews_per_month', 
       'price']


# In[10]:


data = data[keep_columns]


# In[11]:


data['neighbourhood'] = data['neighbourhood'].fillna('')


# In[12]:


data = data[data['neighbourhood'].str.lower().str.contains('bristol')]


# In[13]:


data['neighbourhood'].value_counts()


# In[300]:


X = data.drop(columns=['neighbourhood', 'price'], axis=1)


# In[301]:


X.columns


# In[366]:


y = data['price'].str.replace('$','').str.replace(',','').astype('float').astype('int')


# In[ ]:





# In[14]:


data.info()


# #### 01.a Exploring columns with Object data type

# In[15]:


obj_cols = data.select_dtypes(include=['object']).columns


# In[16]:


obj_data = data[obj_cols]


# In[17]:


obj_cols


# In[18]:


obj_data['host_response_time'].isna().sum(), obj_data['host_response_time'].value_counts()


# In[19]:


obj_data['host_response_rate'].isna().sum(), obj_data['host_response_rate'].value_counts()


# In[20]:


obj_data['host_acceptance_rate'].isna().sum(), obj_data['host_acceptance_rate'].value_counts()


# In[21]:


obj_data['host_is_superhost'].isna().sum(), obj_data['host_is_superhost'].value_counts()


# In[22]:


obj_data['host_verifications'].isna().sum(), obj_data['host_verifications'].value_counts()


# In[23]:


obj_data['host_has_profile_pic'].isna().sum(), obj_data['host_has_profile_pic'].value_counts()


# In[24]:


obj_data['host_identity_verified'].isna().sum(), obj_data['host_identity_verified'].value_counts()


# In[25]:


obj_data['neighbourhood'].isna().sum(), obj_data['neighbourhood'].value_counts()


# In[26]:


obj_data['property_type'].isna().sum(), obj_data['property_type'].value_counts()


# In[27]:


obj_data['room_type'].isna().sum(), obj_data['room_type'].value_counts()


# In[28]:


obj_data['bathrooms_text'].isna().sum(), obj_data['bathrooms_text'].value_counts()


# In[31]:


obj_data['instant_bookable'].isna().sum(), obj_data['instant_bookable'].value_counts()


# In[ ]:





# #### 01.b Exploring columns with Numerical data type

# In[32]:


numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns


# In[33]:


numerical_cols


# In[34]:


numerical_data = data[numerical_cols]


# In[35]:


numerical_data.describe()


# In[36]:


numerical_data.shape


# In[ ]:





# ### 2. Cleaning the Data

# #### 2.a Cleaning numerical data

# In[37]:


for col in numerical_cols:
    if numerical_data[col].isna().sum():
        if numerical_data[col].dtype == 'int64':
            numerical_data[col] = numerical_data[col].fillna(numerical_data[col].median())
        elif numerical_data[col].dtype == 'float64':
            numerical_data[col] = numerical_data[col].fillna(numerical_data[col].mean())


# In[38]:


numerical_data['beds'] = numerical_data['beds'].astype('int64')
numerical_data['bedrooms'] = numerical_data['bedrooms'].astype('int64')


# In[39]:


numerical_data = numerical_data.reset_index(drop=True)


# In[40]:


numerical_data.info()


# In[ ]:





# #### 2.b Cleaning object type data

# In[42]:


impute = SimpleImputer(strategy='most_frequent')


# In[43]:


for col in obj_cols:
    if obj_data[col].isna().sum():
        print(col)
        obj_data[col] = impute.fit_transform(obj_data[[col]])


# In[ ]:





# ### 3. Creating Analytical Data Set (ADS)

# In[45]:


ads = pd.DataFrame()


# In[46]:


ohe = OneHotEncoder()


# In[47]:


# host_response_time
host_response_time = pd.DataFrame(ohe.fit_transform(obj_data[['host_response_time']]).toarray(),columns=ohe.get_feature_names_out())
ads = pd.concat([ads, host_response_time], axis = 1)


# In[48]:


# host_response_rate
def response_cat(response_rate):
    if response_rate == 100:
        return '100'
    elif response_rate >= 95 and response_rate <100:
        return 'over_95'
    elif response_rate >= 90 and response_rate <95:
        return 'over_90'
    elif response_rate >= 70 and response_rate <90:
        return 'over_70'
    else:
        return 'below_70'


# In[50]:


obj_data['host_response_rate'] = obj_data['host_response_rate'].str.replace('%','').astype('int')
obj_data['host_response_rate_cat'] = obj_data['host_response_rate'].apply(response_cat)
obj_data['host_response_rate_cat'].value_counts()

host_response_rate_cat = pd.DataFrame(ohe.fit_transform(obj_data[['host_response_rate_cat']]).toarray(),columns=ohe.get_feature_names_out())
ads = pd.concat([ads, host_response_rate_cat], axis = 1)


# In[51]:


# host_acceptance_rate
def acceptance_cat(acceptance_rate):
    if acceptance_rate == 100:
        return '100'
    elif acceptance_rate >= 95 and acceptance_rate <100:
        return 'over_95'
    elif acceptance_rate >= 90 and acceptance_rate <95:
        return 'over_90'
    elif acceptance_rate >= 70 and acceptance_rate <90:
        return 'over_70'
    else:
        return 'below_70'

obj_data['host_acceptance_rate'] = obj_data['host_acceptance_rate'].str.replace('%','').astype('int')
obj_data['host_acceptance_rate_cat'] = obj_data['host_acceptance_rate'].apply(acceptance_cat)
obj_data['host_acceptance_rate_cat'].value_counts()

host_acceptance_rate_cat = pd.DataFrame(ohe.fit_transform(obj_data[['host_acceptance_rate_cat']]).toarray(),columns=ohe.get_feature_names_out())
ads = pd.concat([ads, host_acceptance_rate_cat], axis = 1)


# In[52]:


# host_is_superhost

host_is_superhost = pd.DataFrame(ohe.fit_transform(obj_data[['host_is_superhost']]).toarray(),columns=ohe.get_feature_names_out())
ads = pd.concat([ads, host_is_superhost], axis = 1)


# In[54]:


# host_verifications

host_verifications = obj_data['host_verifications'].apply(lambda x: len(x.replace('[','').replace(']','').split(',')))
host_verifications = host_verifications.reset_index(drop = True)
ads = pd.concat([ads, host_verifications], axis = 1)


# In[55]:


# host_has_profile_pic

host_has_profile_pic = pd.DataFrame(ohe.fit_transform(obj_data[['host_has_profile_pic']]).toarray(),columns=ohe.get_feature_names_out())
ads = pd.concat([ads, host_has_profile_pic], axis = 1)


# In[56]:


# host_identity_verified

host_identity_verified = pd.DataFrame(ohe.fit_transform(obj_data[['host_identity_verified']]).toarray(),columns=ohe.get_feature_names_out())
ads = pd.concat([ads, host_identity_verified], axis = 1)


# In[57]:


# property_type

property_type = pd.DataFrame(ohe.fit_transform(obj_data[['property_type']]).toarray(),columns=ohe.get_feature_names_out())
ads = pd.concat([ads, property_type], axis = 1)


# In[58]:


# room_type

room_type = pd.DataFrame(ohe.fit_transform(obj_data[['room_type']]).toarray(),columns=ohe.get_feature_names_out())
ads = pd.concat([ads, room_type], axis = 1)


# In[59]:


bathroom_count = obj_data['bathrooms_text'].apply(lambda x: x.replace('Shared half-bath', '0.5 shared').replace('Half-bath', '0.5 bath').split(' ')[0]).astype('float')
bathroom_count = bathroom_count.reset_index(drop = True)
bathroom_count = bathroom_count.rename('bathroom_count')


# In[60]:


obj_data['bathrooms_text'].apply(lambda x: x.replace('Shared half-bath', '0.5 shared').replace('Half-bath', '0.5 bath').split(' ')[0])


# In[61]:


ads = pd.concat([ads, bathroom_count], axis = 1)


# In[62]:


obj_data['bathroom_shared_flag'] = obj_data['bathrooms_text'].apply(lambda x: 't' if 'shared' in x else 'f')


# In[63]:


bathroom_shared = pd.DataFrame(ohe.fit_transform(obj_data[['bathroom_shared_flag']]).toarray(),columns=ohe.get_feature_names_out())
ads = pd.concat([ads, bathroom_shared], axis=1)


# In[64]:


# amenities

amenities = obj_data['amenities'].apply(lambda x: len(x.replace('[','').replace(']','').split(',')))
amenities = amenities.reset_index(drop = True)
ads = pd.concat([ads,amenities], axis = 1)


# In[65]:


# instant_bookable

instant_bookable = pd.DataFrame(ohe.fit_transform(obj_data[['instant_bookable']]).toarray(), columns=ohe.get_feature_names_out())
ads = pd.concat([ads,instant_bookable], axis = 1)


# In[ ]:





# In[66]:


ads = pd.concat([ads, numerical_data], axis=1)


# In[67]:


price = obj_data['price'].str.replace('$','').str.replace(',','').astype('float').astype('int').reset_index(drop=True)
ads = pd.concat([ads, price], axis = 1)


# In[68]:


ads.head()


# In[69]:


ads.isna().sum().sum()


# In[70]:


ads.columns


# ### Training the model

# In[71]:


X = ads.loc[:, ads.columns!='price']
y = ads.loc[:, 'price']


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[73]:


pipe1 = Pipeline(steps=[('linear regression', LinearRegression())])


# In[74]:


model = pipe1.fit(X_train,y_train)


# In[75]:


pred = model.predict(X_test)


# In[76]:


r2_score(y_test, pred), mean_absolute_percentage_error(y_test,pred)


# In[77]:


lr = LinearRegression()


# In[78]:


model = lr.fit(X_train, y_train)


# In[79]:


pred = model.predict(X_test)


# In[80]:


r2 = r2_score(y_test, pred)
mape = mean_absolute_percentage_error(y_test, pred)


# In[81]:


r2, mape


# In[82]:


class TestTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('__init__ called...')
    
    def fit(self, X, y=None):
        print('fit called...')
        return self
    
    def transform(self, X, y=None):
        print('transformer called...')
        return X


# In[83]:


# Pipeline 2

pipe2 = Pipeline(steps=[('transform',TestTransformer()), 
                        ('linear regression', LinearRegression())])


# In[84]:


m2 = pipe2.fit(X_train, y_train)


# In[85]:


pred = m2.predict(X_test)


# In[86]:


r2_score(y_test, pred), mean_absolute_percentage_error(y_test, pred)


# In[87]:


# Sub-set of columns to consider for model training
# Separate columns based on data type
# Perform transformations on columns
# Collate all data back to create ads
# Define the estimator in the pipeline


# In[90]:


from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector


# In[252]:


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


# In[315]:


data.columns[10]


# In[322]:


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


# In[323]:


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
    
#     def shared(self, df):
#         ls = []
#         for i in range(df.shape[0]):
#             print(df.iloc[i,self.feature])
#             if 'shared' in npdf.iloc[i,self.feature]:
#                 ls.append('t')
#             else:
#                 ls.append('f')
#         return ls


# In[324]:


data.info()


# In[325]:


X.info()


# In[373]:


preprocessing = Pipeline(steps = [('PercentageProcessing', PercentageProcessing([1,2]))
                    , ('ItemCount', ItemCount([4,13]))
                    , ('BathroomText', BathroomText([10]))])


# In[374]:


data_tf = pd.DataFrame(preprocessing.fit_transform(X))
data_tf.shape


# In[375]:


data_tf.head()


# In[377]:


data_tf.iloc[0:5, 0:14]


# In[379]:


data_tf.info()


# In[341]:


imputation = ColumnTransformer(transformers=[('FloatTransform', SimpleImputer(strategy='mean'), [10, 11, 15, 16, 17, 18, 19, 20, 21, 27])
                                    ,('IntTransform', SimpleImputer(strategy='median'), [4, 9, 12, 13, 14, 23, 24, 25, 26])
                                    ,('ObjImpute', SimpleImputer(strategy='most_frequent'), [0,1,2,3,5,6,7,8,22])]
                      , verbose_feature_names_out=True
                     )


# In[ ]:





# In[348]:


# ,('imputation', imputation)
# ,
                               


# In[380]:


data_pipeline = Pipeline(steps=[('preprocessing', preprocessing)
                               ,('imputation', imputation)])


# In[386]:


inputed_data = pd.DataFrame(data_pipeline.fit_transform(X))


# In[387]:


inputed_data.iloc[0:5,0:20]


# In[388]:


inputed_data.iloc[0:5,20:40]


# In[389]:


ohe = ColumnTransformer(transformers=[('ohe', OneHotEncoder(handle_unknown='ignore'), [19,20,21,22,23,24,25,26,27])]
                       ,remainder='passthrough')


# In[390]:


data_pipeline = Pipeline(steps=[('preprocessing', preprocessing)
                               ,('imputation', imputation)
                               ,('ohe', ohe)])


# In[412]:


import pickle
pickle.dump(data_pipeline, open('data_pipeline.pkl','wb'))


# In[394]:


processed_data = pd.DataFrame(data_pipeline.fit_transform(X))
processed_data.shape


# In[397]:


processed_data.iloc[0:5, 0:20]


# In[398]:


processed_data.iloc[0:5, 20:40]


# In[399]:


processed_data.iloc[0:5, 40:60]


# In[400]:


processed_data.iloc[0:5, 60:80]


# In[402]:


ml_pipeline = Pipeline(steps=[('data_pipeline', data_pipeline)
                             ,('linear_regression', LinearRegression())])


# In[403]:


display(ml_pipeline)


# In[405]:


m1 = ml_pipeline.fit(X,y)


# In[408]:


pred = m1.predict(X)


# In[410]:


r2_score(y, pred), mean_absolute_percentage_error(y, pred)


# In[270]:


processed_data.info()


# In[ ]:


# drop 7
[0, 1, 2, 3, 5, 6, 8, 9, 23]


# In[ ]:




