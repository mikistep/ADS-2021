from .config import *
import pymysql

# This file accesses the data

# estabilishes connection with server
def create_connection(user, password, host, database, port=3306):
    conn = None
    try:
        conn = pymysql.connect(
            user=user,
            passwd=password,
            host=host,
            port=port,
            local_infile=1,
            db=database,
        )
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

# creates table `pp_data`, loads data from downloaded file
def create_pp_data(conn, file_path):
    cur = conn.cursor()

    query_create = """
    DROP TABLE IF EXISTS `pp_data`;

    CREATE TABLE IF NOT EXISTS `pp_data` (
      `transaction_unique_identifier` tinytext COLLATE utf8_bin NOT NULL,
      `price` int unsigned NOT NULL,
      `date_of_transfer` date NOT NULL,
      `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
      `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
      `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
      `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
      `primary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
      `secondary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
      `street` tinytext COLLATE utf8_bin NOT NULL,
      `locality` tinytext COLLATE utf8_bin NOT NULL,
      `town_city` tinytext COLLATE utf8_bin NOT NULL,
      `district` tinytext COLLATE utf8_bin NOT NULL,
      `county` tinytext COLLATE utf8_bin NOT NULL,
      `ppd_category_type` varchar(2) COLLATE utf8_bin NOT NULL,
      `record_status` varchar(2) COLLATE utf8_bin NOT NULL,
      `db_id` bigint(20) unsigned NOT NULL
    ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;

    ALTER TABLE `pp_data`
    ADD PRIMARY KEY (`db_id`);
        
    ALTER TABLE `pp_data`
    MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;
    """
    cur.execute(query_create)
    cur.commit()

    query_load = """
    LOAD DATA LOCAL INFILE '{}' INTO TABLE `pp_data` 
    FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '"'
    LINES STARTING BY '' TERMINATED BY '\n';
    """.format(
        file_path
    )
    cur.execute(query_load)
    cur.commit()
    return


# creates table `postcode_data`, loads data into and creates index
def create_postcode_data(conn, file_path):
    cur = conn.cursor()

    query_create = """
    DROP TABLE IF EXISTS `postcode_data`;
    CREATE TABLE IF NOT EXISTS `postcode_data` (
      `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
      `status` enum('live','terminated') NOT NULL,
      `usertype` enum('small', 'large') NOT NULL,
      `easting` int unsigned,
      `northing` int unsigned,
      `positional_quality_indicator` int NOT NULL,
      `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
      `lattitude` decimal(11,8) NOT NULL,
      `longitude` decimal(10,8) NOT NULL,
      `postcode_no_space` tinytext COLLATE utf8_bin NOT NULL,
      `postcode_fixed_width_seven` varchar(7) COLLATE utf8_bin NOT NULL,
      `postcode_fixed_width_eight` varchar(8) COLLATE utf8_bin NOT NULL,
      `postcode_area` varchar(2) COLLATE utf8_bin NOT NULL,
      `postcode_district` varchar(4) COLLATE utf8_bin NOT NULL,
      `postcode_sector` varchar(6) COLLATE utf8_bin NOT NULL,
      `outcode` varchar(4) COLLATE utf8_bin NOT NULL,
      `incode` varchar(3)  COLLATE utf8_bin NOT NULL,
      `db_id` bigint(20) unsigned NOT NULL
    ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;

    ALTER TABLE `postcode_data`
    ADD PRIMARY KEY (`db_id`);
        
    ALTER TABLE `postcode_data`
    MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;
    """
    cur.execute(query_create)
    cur.commit()

    query_load = """
    LOAD DATA LOCAL INFILE '{}' INTO TABLE `postcode_data`
    FIELDS TERMINATED BY ',' 
    LINES STARTING BY '' TERMINATED BY '\n';
    """.format(
        file_path
    )
    cur.execute(query_load)
    cur.commit()

    query_index = """
    CREATE INDEX `po.postcode` USING BTREE
      ON `postcode_data`
        (postcode);
    """
    cur.execute(query_index)
    cur.commit()
    return


# creates table `prices_coordinates_data`, loads data from join and creates indexes
def create_prices_coordinates_data(conn):
    cur = conn.cursor()

    query_create = """
    DROP TABLE IF EXISTS `prices_coordinates_data`;
    CREATE TABLE IF NOT EXISTS `prices_coordinates_data` (
      `price` int unsigned NOT NULL,
      `date_of_transfer` date NOT NULL,
      `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
      `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
      `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
      `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
      `locality` tinytext COLLATE utf8_bin NOT NULL,
      `town_city` tinytext COLLATE utf8_bin NOT NULL,
      `district` tinytext COLLATE utf8_bin NOT NULL,
      `county` tinytext COLLATE utf8_bin NOT NULL,
      `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
      `lattitude` decimal(11,8) NOT NULL,
      `longitude` decimal(10,8) NOT NULL,
      `db_id` bigint(20) unsigned NOT NULL
    ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1;

    ALTER TABLE `prices_coordinates_data`
    ADD PRIMARY KEY (`db_id`);

    ALTER TABLE `prices_coordinates_data`
    MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;
    """
    cur.execute(query_create)
    cur.commit()

    query_load = """
    INSERT INTO prices_coordinates_data (price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county, country, lattitude, longitude)
    SELECT price,date_of_transfer, postcode_data.postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county, country, lattitude, longitude
    FROM
        pp_data
    INNER JOIN
        postcode_data
    ON
        postcode_data.postcode = pp_data.postcode
    """
    cur.execute(query_load)
    cur.commit()

    query_index = """
    CREATE INDEX `lat_long_date` USING BTREE
      ON `prices_coordinates_data`
        (lattitude, longitude, date_of_transfer);
    """
    cur.execute(query_index)
    cur.commit()
    return


# creates database
def run_all(conn, pp_data_path, postcode_path):
    create_pp_data(conn, pp_data_path)
    create_postcode_data(conn, postcode_path)
    create_prices_coordinates_data(conn)
    return


# used for showcases
def run_custom_query(conn, query):
    cur = conn.cursor()
    cur.execute(query)
    return cur.fetchall()
