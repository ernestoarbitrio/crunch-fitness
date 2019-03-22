import csv
import os
import json
import sys
import pandas as pd
import gridfs
from cr.db.store import global_settings, connect


def load_data(filename, settings=None, clear=None):
    if settings is None:
        settings = global_settings
        global_settings.update(json.load(file(sys.argv[1])))

    db = connect(settings)

    obj_name = os.path.basename(filename).split('.')[0]

    collection = getattr(db, obj_name)

    if clear:
        collection.remove()

    with file(filename) as the_file:
        objs = json.load(the_file)
        for obj in objs:
            collection.insert(obj)


def load_bulk_data(filename, settings=None, clear=None):
    if settings is None:
        settings = global_settings
        global_settings.update(json.load(file(sys.argv[1])))

    db = connect(settings)

    obj_name = os.path.basename(filename).split('.')[0]

    collection = getattr(db, obj_name)

    if clear:
        collection.remove()

    with file(filename) as the_file:
        objs = json.load(the_file)
    collection.insert_many(objs)


def load_dataset(csv_filename, db, h=None):
    '''
    Requirements: pandas
    Using pandas to load the csv file (chunk size is useful when file is big),
    all the field will be automatically casted to the right type of data.
    '''

    file_csv = pd.read_csv(csv_filename, header=h, chunksize=1000)

    for df in file_csv:

        gender_unique = {v: k for k, v in enumerate(df.iloc[:, -1].unique())}
        df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: gender_unique[x] if x != None else gender_unique[
            x])  # this produce an int64 type on gender column

        categorical = df.select_dtypes(include='object').apply(
            pd.factorize)  # Encode the object as an enumerated type or categorical variable (integer values).
        bools = df.select_dtypes(include='bool')  # Cast boolenas to bool
        others = df.select_dtypes(exclude=['bool', 'object'])  # get all the other columns of the dataframe

        headers = list(categorical.index) + list(bools.columns) + list(others.columns)

        columns = [a[0].tolist() for a in categorical.values] + \
                  [list(bools[col]) for col in bools] + \
                  [list(others[col]) for col in others]

        data = {'headers': headers,
                'columns': columns
                }

        return db.datasets.insert(data)
