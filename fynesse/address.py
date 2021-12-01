# This file contains code for suporting addressing questions in the data

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

# default tags
tags = [
    {
        "key": "historic",
        "value": True,
        "name": "historic",
    },
    {
        "key": "leisure",
        "value": True,
        "name": "leisure",
    },
    {
        "key": "tourism",
        "value": True,
        "name": "tourism",
    },
    {
        "key": "healthcare",
        "value": True,
        "name": "healthcare",
    },
    {
        "key": "office",
        "value": True,
        "name": "office",
    },
    {
        "key": "public_transport",
        "value": True,
        "name": "public_transport",
    },
    {
        "key": "landuse",
        "value": ["commercial", "construction", "industrial"],
        "name": "landuse",
    },
    {
        "key": "man_made",
        "value": True,
        "name": "man_made",
    },
    {
        "key": "amenity",
        "value": ["pub", "restaurant", "bar", "cafe", "fast_food", "food_court"],
        "name": "food places",
    },
    {
        "key": "amenity",
        "value": [
            "school",
            "college",
            "kindergarten",
            "language_school",
            "university",
            "library",
        ],
        "name": "education",
    },
    {
        "key": "amenity",
        "value": ["post_box", "post_office", "post_depot"],
        "name": "postal points",
    },
    {
        "key": "building",
        "value": ["civic", "government", "public", "transportation"],
        "name": "public buildings",
    },
]

# transforms dataframe into design matrix
def df_to_design(df, tags, date):
    feature_columns = [
        df[tag["name"]].to_numpy(dtype=float).reshape(-1, 1) for tag in tags
    ]
    design = np.concatenate(
        (
            np.ones(df.shape[0]).reshape(-1, 1),
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


# trains model
def train_model(data, tags, query_date):
    design_matrix = df_to_design(data, tags, query_date)
    glm = sm.GLM(
        data["price"],
        design_matrix,
        family=sm.families.Gamma(link=sm.genmod.families.links.identity),
    )
    model = glm.fit()
    return model


# removes outliers and splits data to train and test datasets
def select_and_split(data):
    prices = data["price"]
    low, high = np.percentile(prices, [2.5, 97.5])
    main_data = data[data["price"].apply(lambda x: x >= low and x <= high)]
    train, test = train_test_split(main_data, test_size=0.1)
    return train, test


# visualizes prediction
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


# returns mean of mean absolute percentage error over 10 trials
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
    print("error should be within <0.18, 0.25>")
    print("error above 0.35 means model has poor quality")


# constructs GeoDataFrame from longitude and latitude
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


# predicts price at a location according to parameters
def predict_price(
    conn,
    latitude,
    longitude,
    date,
    property_type,
    lower=5000,
    upper=7000,
    distance=1000,
    tags=tags,
    year_change=3,
):
    data, box = assess.get_data(
        conn,
        tags,
        latitude,
        longitude,
        date,
        lower=lower,
        upper=upper,
        distance=distance,
        year_change=year_change,
    )
    train, test = select_and_split(data)
    design_test = df_to_design(test, tags, date)

    model = train_model(train, tags, date)

    test_prediction = model.predict(design_test)
    error = sklearn.metrics.mean_absolute_percentage_error(
        test["price"], test_prediction
    )
    print("Mean absolute percentage error:", error)
    print("error should be within <0.18, 0.25>")
    print("error above 0.35 means model has poor quality")

    query_df = construct_gdf(latitude, longitude, date, property_type)
    query_df = assess.get_nearby_count(query_df, tags, box, distance=distance)
    query_design = df_to_design(query_df, tags, date)
    result = model.predict(query_design)
    return result, model, data, box, test


# outputs information about the model that are able to be interpreted by a human
def interprete_model(model, tags):
    values = model.params
    print(
        f'A house of type "Other" on the day of a query without any objects nearby is estimated to cost {values[0]:.0f}'
    )
    print(f"house price increases by {values[1]:.2f} each day")
    print(f"Flat property type increase price by {values[2]:.2f}")
    print(f"Semidetached property type increase price by {values[3]:.2f}")
    print(f"Detached property type increase price by {values[4]:.2f}")
    print(f"Terraced property type increase price by {values[5]:.2f}")
    for i in range(len(tags)):
        print(
            f"One object with tag '{tags[i]['name']}' increases price by {values[6+i]:.2f}"
        )
