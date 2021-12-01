# Description

This repo was made for Advanced Data Science assignment.

It uses fynesse template which has three aspects: access, assess and address.

Files for each aspect is in `fynesse/{aspect}.py`

Guide to the repository is in `notebooks/guide.ipynb`.

# Code overview

## Access

### create_connection

`create_connection(user, password, host, database, port=3306)`

This function returns connection to the database using `pymysql` library.

### create_pp_data

`create_pp_data(conn, file_path)`

This function creates `pp_data` table in the database corresponding to connection `conn` based on local file which path is `file_path`.

The schema is described in `notebooks/guide.ipynb`.

### create_postcode_data

create_postcode_data(conn, file_path)

This function creates `postcode_data` table in the database corresponding to connection `conn` based on local file which path is `file_path`.

The schema is described in `notebooks/guide.ipynb`.

### create_prices_coordinates_data

`create_prices_coordinates_data(conn)`

This function creates `prices_coordinates_data` table by performing a join on `pp_data` and `postcode_data`. It assumes that `pp_data` and `postcode_data` are already in the database.

The schema and indexing are described in `notebooks/guide.ipynb`.

### run_all

`run_all(conn, pp_data_path, postcode_path)`

This function creates table in the data so that they are ready for `assess` stage.

`pp_data_path` should be a path pointing to local file with UK house prices.

`postcode_path` should be a path pointing to local file with UK postcodes.

### run_custom_query

`run_custom_query(conn, query)`

This function executes query `query` on connection `conn` and returns result.

## Assess

### Coordinate systems

In my project I am using two coordinate systems, epsg 4326 and epsg 27700.

Epsg 4326 is commonly used, it has latitude and longitude.

Epsg 27700 aims to map 3D surface onto 2D flat plane. It is used specificallly for the UK. Positions are expressed in northings and eastings. If place `X` has one more northing than place `Y` it means that `X` is one metre to the north of `Y`. Similarly one easting corresponds to one metre east.

I have to use epsg 4326 because osmnx uses this format. After getting all data I am transfroming to epsg 27700 because it is more convinient for some things I will use.

GeoDataFrame supports both formats and makes conversion between them easy.

### change_in_coor

`change_in_coor(kilometres, latitude)`

This function returns difference in latitude and longitude given kilometres in `kilometres` parameter.
As the result depends on the latitude it is provided in parameter.

The math in the function is based on answer to this question.
https://stackoverflow.com/questions/1253499/simple-calculations-for-working-with-lat-lon-and-km-distance

### Boxes

I store query data in dictonaries called boxes. They store information about spacial and time bounds of a query.


### construct_box

`construct_box(latitude, longitude, change, start_date, end_date)`

This function creates a box centred at `latitude`, `longitude` with square of apothem of length `change` expressed in kilometres.
The box has start date of `start_date` and end date of `end_date`.

### extend_box

`extend_box(box, more)`

This function creates a copy of a given box, extends it by `more` kilometres and returns it.

### Tag

Tag is a data structure that is used to represent queries to OpenStreetMap.

Tag is a dictionary with 3 values:
 - `key` - represents key in OpenStreetMap.
 
    Type: string
 - `value` - represents value in OpenStreetMap.

    Type: boolean or list of strings
 - `name` - represents column name

    Type: string

A list of tags named `tags` is in `assess.py`.

### construct_dict

`construct_dict(tags)`

`tags` is a list of tags

This function merges tags into a dictonary that can be used to query OpenStreetMap.

### get_transaction_count

`get_transaction_count(conn, box, debug=False)`

This function returns the number of transactions within a `box` in the database associated with connection `conn`.

### get_transactions

`get_transactions(conn, box, debug=False)`

This function returns transactions within a `box` in the database associated with connection `conn`.

### lower, upper

Lower and upper are integers that are used to signify how many transactions should the query return from the database. The query should make best effort to return between lower and upper transactions.

### get_box

`get_box(conn, latitude, longitude, start_date, end_date, lower, upper, debug=False)`

This function uses connection `conn` to return a box centred at point `latitude, longitude` with time boundaries `[start_date, end_date]`. There should be between `lower` and `upper` transactions within the box although it is not guaranteed.

#### How do I chose what box to use?

A simple solution to this problem is to set a constant $C$ and always return box of size $C \times C$. The problem with this approach is that it will return too many results in city centres and too little in sparsely populated areas.

I decided to adjust size of the box based on number of transactions inside. I set start and end date to a constant and then try to find to bounds that have proper number of transactions.

To do so I will use `assess.get_transaction_count` which returns number of transaction within a box. Time spent on this query is proportional to the result.

Algorithm sketch:
 - set time bounds to some constant
 - query a small area
 - increase area size by 2 until result is greate than lower
 - if result is in range then return the box
 - if result is too large then binary search box size

The algorithm should take less than a second to return 10000 of transactions.

### get_nearby_count

`get_nearby_count(gdf, tags, box, distance=1000)`

Input parameters:
 - `gdf` - A GeoDataFrame containing transactions.
 - `tags` - an array of tags
 - `box` - box of the query
 - `distance` - search radius expressed in metres

 The function will add feature columns to `gdf` each tag in `tags`. For each transaction the function will count the number of objects withing the `distance` that have features specified in tag.

Algorithm:
1. Get all points of interest with given tags in the extended `box`.
2. Convert coordinate format to epsg 27700.
3. Join each point of interest to each transaction that is within `distance` metres.
4. Count number of joins for each tag for each transaction

Optimization:
 - To calculate places within a distance the code calls `.buffer(distance)` on transactions which transforms point to circle of radius `distance`. Then I do a spatial join which uses some fast algorithm.
 - Many transaction will have same coordinates. Instead of matching every transaction I match every postcode.
 - I don't care about precision that much. I approximate points of interests to their centre using `centroid` method.

### coor_to_grid

`coor_to_grid(lat, long)`

This function takes latitude `lat` and longitude `long` and returns a dataframe of that point converted to epsg 27700.

### plot_points

`plot_points(df, box)`

This function plots a map of transactions given in `df` within a `box`.

### nearby_distributions

`nearby_distributions(df, tags)`

This function distributions of features `tags` from DataFrame `df`.

### given_distributions

`given_distributions(df)`

This function plots distributions of prices, dates and property types from DataFrame `df`.

### price_correlation_distributions

`price_correlation_distributions(df, tags, remove_percentile=0)`

Input parameters:
 - df - DataFrame of transactions with features
 - tags - an array of tags
 - remove_percentile - a number between 0 and 100

This function plots joint distributions of features from DataFrame `df`. It removes transactions with price in the upper `remove_percentile` percentile.

### get_data

`get_data(conn, tags, latitude, longitude, date, lower=500, upper=1000, distance=1000, year_change=3)`

Input parameters:
 - `conn` - a pymysql object, connection to database
 - `tags` - an array of tags
 - `latitude` - latitude of queried point
 - `longitude` - logitude of queried point
 - `date` - datetime.date, date of query
 - `lower` - expected lower limit of queries
 - `upper` - expected upper limit of queries
 - `distance` - argument passed to `get_nearby_count`
 - `year_change` - argument specifying lower and upper limit on dates of transactions. Transactions shouldn't happen more than `year_change` before or after `date`

 This function creates box given parameters using `get_box`. Then it gets transactions within that box using `get_transactions`. Later it adds feature colums using `get_nearby_count`.

 It returns box and dataframe.

 ## address

 ### address.tags

 Default tags

 ### df_to_design

 df_to_design(df, tags, date)

 Input parameters:
  - df - DataFrame
  - tags - an array of tags
  - date - datetime.date, date of query

This function returns design matrix from parameters.

 ### train_model

 train_model(df, tags, date)

 Input parameters:
  - df - DataFrame
  - tags - an array of tags
  - date - datetime.date, date of query

This function creates model and trains in based on `df`.

### select_and_split

`select_and_split(data)`

This function removes outliers from data and split remaining data into trainig and test datasets.

### visualize_prediction

`visualize_prediction(test_actual, test_prediction)`

This function visualizes prediction.

### evaluate_data_quality

`evaluate_data_quality(data, tags, query_date)`

This function evaluates quality of the data by doint 10 different test splits and computing mean of mean absolute percentage error.

### construct_gdf

`construct_gdf(latitude, longitude, date, property_type)`

This function creates GeoDataFrame based on parameters and transforms it into epsg 27700.

### predict_price

`predict_price(conn, latitude, longitude, date, property_type, lower=5000, upper=7000, distance=1000, tags=tags, year_change=3)`

This function trains a model a given place within given limits.

Then it calculates a predictio for place with given proprty type.

### interprete_model

`interprete_model(model, tags)`

This function takes `model` and `tags` it was created with.

It prints interpretation of the model.