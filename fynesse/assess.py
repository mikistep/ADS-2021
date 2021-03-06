from .config import *

from . import access

import math
import matplotlib.pyplot as plt
import osmnx as ox
import pandas
import geopandas
import numpy as np
import datetime

# returns change in latitude and longitude
# formula based on https://stackoverflow.com/questions/1253499/simple-calculations-for-working-with-lat-lon-and-km-distance
def change_in_coor(kilometres, latitude):
    return kilometres / 110.574, kilometres / (
        111.320 * math.cos(latitude / 180 * math.pi)
    )


# box is used to store query parametres
def construct_box(latitude, longitude, change, start_date, end_date):
    d_lat, d_long = change_in_coor(change, latitude)
    return {
        "base_latitude": latitude,
        "base_longitude": longitude,
        "min_latitude": latitude - d_lat,
        "max_latitude": latitude + d_lat,
        "min_longitude": longitude - d_long,
        "max_longitude": longitude + d_long,
        "min_date": start_date,
        "max_date": end_date,
        "change": change,
    }


# extends box to cover more area
def extend_box(box, more):
    ans = box.copy()
    change = box["change"] + more

    d_lat, d_long = change_in_coor(change, box["base_latitude"])
    ans["min_latitude_e"] = ans["min_latitude"] - d_lat
    ans["max_latitude_e"] = ans["max_latitude"] + d_lat
    ans["min_longitude_e"] = ans["min_longitude"] - d_long
    ans["max_longitude_e"] = ans["max_longitude"] + d_long
    return ans


# tags are used to query osmnx, each tag has key, value and name
def construct_dict(tags):
    ans = {}
    for tag in tags:
        if tag["value"] == True:
            ans[tag["key"]] = True
        else:
            prev = ans.get(tag["key"])
            if prev is None:
                ans[tag["key"]] = tag["value"]
            elif prev is not True:
                ans[tag["key"]] = tag["value"] + prev
    return ans


# construts box given start_date, end_date, main point coordinates
# number of transactions is expected to be in <lower, upper>
def get_box(conn, latitude, longitude, start_date, end_date, lower, upper, debug=False):
    assert lower < upper
    min_change = 0.5  # in kilometres
    max_change = 1000
    while True:
        now_change = min_change
        box = construct_box(
            latitude, longitude, now_change, start_date=start_date, end_date=end_date
        )
        count = access.get_transaction_count(conn, box)
        if debug:
            print(
                "checking {} kilometres, got {} transactions".format(now_change, count)
            )
        if count < lower:
            min_change *= 1.4
            if min_change > max_change:
                return 0, construct_box(latitude, longitude, 0, "1970-01-01", 0)
                # nothing found with 1000 kilometres
        elif count > upper:
            max_change = min_change
            min_change /= 1.4
            break
        else:
            box["change"] = now_change
            return count, box
    while True:
        av_change = (min_change + max_change) / 2
        box = construct_box(
            latitude, longitude, av_change, start_date=start_date, end_date=end_date
        )
        count = access.get_transaction_count(conn, box)
        if debug:
          print("checking {} kilometres, got {} transactions".format(av_change, count))
        if count < lower:
            min_change = av_change
        elif count > upper:
            max_change = av_change
        else:
            return count, box
        if max_change - min_change < 1:
            # no point of being more precise
            return count, box


# adds feature columns to gdf based on tags
def get_nearby_count(gdf, tags, box, distance=1000):
    tag_dict = construct_dict(tags)
    print(tag_dict)
    box = extend_box(box, 1 + distance / 1000)
    pois = ox.geometries_from_bbox(
        box["max_latitude_e"],
        box["min_latitude_e"],
        box["max_longitude_e"],
        box["min_longitude_e"],
        tag_dict,
    )
    pois.set_crs(epsg=4326, inplace=True)
    pois.to_crs(epsg=27700, inplace=True)
    pois["geometry"] = pois["geometry"].centroid
    pois = pois[list(set([tag["key"] for tag in tags])) + ["geometry"]]

    postcode_data = gdf[["postcode", "geometry"]].set_crs(epsg=27700)
    postcode_data.drop_duplicates(subset=["postcode"], inplace=True)
    postcode_data["geometry"] = postcode_data.buffer(distance)
    pois_join = postcode_data.sjoin(pois)

    for tag in tags:
        temp = pois_join[["postcode", tag["key"]]].copy()
        temp.dropna(inplace=True)
        if tag["value"] != True:
            temp = temp[temp[tag["key"]].isin(tag["value"])]
        d = temp.groupby("postcode").size().to_dict()
        gdf[tag["name"]] = gdf["postcode"].apply(lambda x: d.get(x, 0))
    return gdf


# transforms lat, long into northing, easting
def coor_to_grid(lat, long):
    df = pandas.DataFrame({"id": [0]})
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy([long], [lat]))
    gdf.set_crs(epsg=4326, inplace=True)
    gdf.to_crs(epsg=27700, inplace=True)
    return gdf


# plots transaction places together with important roads
def plot_points(df, box):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    graph = ox.graph_from_bbox(
        box["min_latitude"],
        box["max_latitude"],
        box["min_longitude"],
        box["max_longitude"],
    )
    # Retrieve nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph)

    def important_road(road):
        blacklist = ["service", "footway", "bridleway", "living_street"]
        return not any(t in road for t in blacklist)

    edges = edges[edges["highway"].apply(important_road)]
    edges.set_crs(epsg=4326, inplace=True)
    edges.to_crs(epsg=27700, inplace=True)
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray", zorder=0)
    query_point = coor_to_grid(box["base_latitude"], box["base_longitude"])
    query_point.plot(marker=".", markersize=50, ax=ax, color="red", zorder=2)
    postcode_data = df.drop_duplicates(subset=["postcode"], inplace=False)
    postcode_data.plot(marker=".", markersize=10, ax=ax, color="blue", zorder=1).set(
        xlabel="easting",
        ylabel="northing",
        label="postcodes",
        title="postcodes centres",
    )
    return


# plots distribution of tag based features
def nearby_distributions(df, tags):
    fig, axs = plt.subplots(len(tags), figsize=(12, 15))
    fig.tight_layout()
    for i in range(len(tags)):
        name = tags[i]["name"]
        axs[i].hist(df[name], bins=20)
        axs[i].title.set_text(name + " distribution")


# plots distribution of properties taken from SQL
def given_distributions(df):
    fig, axs = plt.subplots(3, figsize=(12, 9))
    fig.tight_layout()
    axs[0].hist(df["price"], bins=400)
    axs[0].title.set_text("Price distribution")
    axs[1].hist(df["date"], bins=50)
    axs[1].title.set_text("Date distribution")
    axs[2].hist(df["property_type"], bins=5)
    axs[2].set_xticks([0.4, 1.2, 2.0, 2.8, 3.6])


# plots correlation between price and features
def price_correlation_distributions(df, tags, remove_percentile=0):
    fig, axs = plt.subplots(2 + len(tags), figsize=(12, 20))
    fig.tight_layout()
    MAX = np.percentile(df["price"], 100 - remove_percentile)
    dates = pandas.date_range(
        min(df["date"]), max(df["date"]), periods=20
    ).to_pydatetime()
    lfunc = lambda e: e.date()
    npd = np.vectorize(lfunc)(dates)
    axs[0].hist2d(df["date"], df["price"], bins=[npd, np.linspace(0, MAX, 20)])
    axs[0].title.set_text("Date - price relation")

    d = {"D": 0, "S": 1, "T": 2, "F": 3, "O": 4}
    axs[1].hist2d(
        df["property_type"].apply(lambda x: d[x]),
        df["price"],
        bins=[np.arange(0, 6, 1), np.linspace(0, MAX, 20)],
    )
    axs[1].set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
    axs[1].set_xticklabels(["D", "S", "T", "F", "O"])
    axs[1].title.set_text("Property type - price relation")
    for i in range(len(tags)):
        name = tags[i]["name"]
        limit = max(df[name])
        axs[i + 2].hist2d(
            df[name],
            df["price"],
            bins=[np.arange(0, limit + 2, 1), np.linspace(0, MAX, 100)],
        )
        axs[i + 2].title.set_text(name + " - price relation")


# returns SQL data from a point with given restrictions, adds osmnx feature columns
def get_data(
    conn,
    tags,
    latitude,
    longitude,
    date,
    lower=500,
    upper=1000,
    distance=1000,
    year_change=3,
):
    start_date = date - datetime.timedelta(days=365 * year_change)
    end_date = date + datetime.timedelta(days=365 * year_change)

    cnt, bx = get_box(
        conn,
        latitude=latitude,
        longitude=longitude,
        start_date="'" + str(start_date) + "'",
        end_date="'" + str(end_date) + "'",
        lower=lower,
        upper=upper,
    )
    print(cnt, bx)
    result = access.get_transactions(conn, bx)
    data = get_nearby_count(result, tags, bx, distance=distance)
    return data, bx
