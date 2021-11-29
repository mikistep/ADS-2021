# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

from osmnx.geometries import _create_gdf
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import time
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas
import geopandas

from . import assess

tags = [
        {"key": "historic",
         "value": True,
         "name": "historic",},
        {"key": "leisure",
         "value": True,
         "name": "leisure",},
        {"key": "tourism",
         "value": True,
         "name": "tourism",},
        {"key": "healthcare",
         "value": True,
         "name": "healthcare",},
        {"key": "office",
         "value": True,
         "name": "office",},
        {"key": "public_transport",
         "value": True,
         "name": "public_transport",},
        {"key": "landuse",
         "value": ["commercial", "construction", "industrial"],
         "name": "landuse",},
        {"key": "man_made",
         "value": True,
         "name": "man_made",},
        {"key": "amenity",
         "value": ["pub", "restaurant", "bar", "cafe", "fast_food", "food_court"],
         "name": "food places",},
        {"key": "amenity",
         "value": ["school", "college", "kindergarten", "language_school", "university", "library"],
         "name": "education",},
        {"key": "amenity",
         "value": ["post_box", "post_office", "post_depot"],
         "name": "postal points",},
        {"key": "building",
         "value": ["civic", "government", "public", "transportation"],
         "name": "public buildings",}
]


def df_to_design(df, tags, date):
    feature_columns = [
        df[tag["name"]].to_numpy(dtype=float).reshape(-1, 1) for tag in tags
    ]
    design = np.concatenate(
        (
            np.ones(df.shape[0]).reshape(-1,1),
            df["date"].apply(lambda x: (x - date).days).to_numpy().reshape(-1, 1),
            np.where(df["property_type"] == "F", 1, 0).reshape(-1, 1),
            np.where(df["property_type"] == "S", 1, 0).reshape(-1, 1),
            np.where(df["property_type"] == "D", 1, 0).reshape(-1, 1),
            np.where(df["property_type"] == "T", 1, 0).reshape(-1, 1),
        )
        + tuple(feature_columns),
        axis=1,
    )
    return design


def train_model(data, tags, query_date):
    design = df_to_design(data, tags, query_date)
    glm_basis = sm.GLM(data["price"], design, family=sm.families.Gaussian())
    regularized_basis = glm_basis.fit()
    return regularized_basis


def select_and_split(data):
    prices = data["price"]
    low, high = np.percentile(prices, [2.5, 97.5])
    main_data = data[data["price"].apply(lambda x: x >= low and x <= high)]
    train, test = train_test_split(main_data, test_size=0.1)
    return train, test


def visualize_prediction(test_actual, test_prediction):
    MAX = max(max(test_actual), max(test_prediction)) + 1000
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal", adjustable="box")
    ax.hist2d(
        test_actual,
        test_prediction,
        bins=[np.linspace(0, MAX, 100), np.linspace(0, MAX, 100)],
    )
    ax.set(xlabel="predicted", ylabel="actual")
    plt.show()


def evaluate_data_quality(data, tags, query_date):
    errors = np.array([])
    for i in range(10):
        train, test = select_and_split(data)
        model = train_model(train, tags, query_date)

        design_test = df_to_design(test, tags, query_date)
        test_prediction = model.predict(design_test)
        error = sklearn.metrics.mean_absolute_percentage_error(
            test["price"], test_prediction
        )
        errors = np.append(errors, error)
    print(errors)
    print("mean absolute error:", errors.mean())
    print("error should be within <0.20, 0.25>")
    print("error above 0.35 means model has poor quality")

def construct_gdf(latitude, longitude, date, property_type):
    query_df = pandas.DataFrame(
        {
            "latitude": [latitude],
            "longitude": [longitude],
            "property_type": [property_type],
            "date": [date],
            "postcode": "invalid postcode",
        }
    )
    query_df = geopandas.GeoDataFrame(
        query_df,
        geometry=geopandas.points_from_xy(query_df.longitude, query_df.latitude),
    )
    query_df.set_crs(epsg=4326, inplace=True)
    query_df.to_crs(epsg=27700, inplace=True)
    return query_df

def predict_price(conn, latitude, longitude, date, property_type, lower=5000, upper=7000, distance=1000, tags = tags, year_change = 3):
    data, box = assess.get_data(
        conn, tags, latitude, longitude, date, lower=lower, upper=upper, distance=distance, year_change=year_change
    )
    train, test = select_and_split(data)
    design_test = df_to_design(test, tags, date)

    model = train_model(train, tags, date)

    test_prediction = model.predict(design_test)
    error = sklearn.metrics.mean_absolute_percentage_error(
        test["price"], test_prediction
    )
    print("Mean absolute percentage error:", error)
    print("error should be within <0.20, 0.25>")
    print("error above 0.35 means model has poor quality")

    query_df = construct_gdf(latitude, longitude, date, property_type)
    query_df = assess.get_nearby_count(query_df, tags, box, distance=distance)
    query_design = df_to_design(query_df, tags, date)
    result = model.predict(query_design)
    return result, model, data, box

