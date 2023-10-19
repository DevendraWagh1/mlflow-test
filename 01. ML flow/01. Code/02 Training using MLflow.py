#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from urllib.parse import urlparse
import logging
import warnings
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from ML_Pipeline_Regression_Airbnb import PercentageProcessing, ItemCount, BathroomText


# In[2]:


parent = 'E:\\00_Learning\\01. Courses\\05. ML Pipeline'
input = 'E:\\00_Learning\\01. Courses\\05. ML Pipeline\\02. Input'


# In[3]:


def eval_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return r2, mape

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    ml_pipeline = pickle.load(open(os.path.join(parent,'ml_pipeline_airbnb.pkl'), 'rb'))
    data = pd.read_csv(os.path.join(input,'airbnb_listings.csv'))
    data = data[data['neighbourhood'].fillna('').str.lower().str.contains('bristol')]
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
    data = data[keep_columns]
    X = data.drop(columns=['neighbourhood', 'price'], axis=1)
    y = data['price'].str.replace('$','').str.replace(',','').astype('float').astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    
    with mlflow.start_run():
        m1 = ml_pipeline.fit(X_train, y_train)
        pred = m1.predict(X_test)
        r2, mape = eval_metrics(y_test, pred)
        
        print(f"R Square: {r2}")
        print(f"MAPE    : {mape}")
        mlflow.log_metric('R2', r2)
        mlflow.log_metric("MAPE", mape)
        
        predictions = m1.predict(X_train)
        signature = infer_signature(X_train, predictions)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                m1, "model", registered_model_name="LinearRegressionAirbnbModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(m1, "model", signature=signature)


# In[ ]:




