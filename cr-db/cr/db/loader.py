import csv
import os
import json
import sys

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


def load_dataset(csv_filename, db):
    with file(csv_filename, 'rU') as csv_file:
        csv_data = csv.reader(csv_file)
        headers = csv_data.next()

        last_header = None
        for i, header in enumerate(headers):
            if header:
                last_header = header
            else:
                # multiple response have no header
                headers[i] = last_header

        columns = [[] for _ in headers]
        for r, row in enumerate(csv_data):
            for i, data in enumerate(row):
                if data:
                    columns[i].append(str(data))
                else:
                    columns[i].append(None)

        data = {'headers': headers,
                'columns': columns,
                }

        return db.datasets.insert(data)
