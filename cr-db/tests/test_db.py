"""
Test Module for the crunch persistence.

hint: we use py.test.
"""

import os
import csv
import numpy as np
import pandas as pd

from cr.db.loader import load_data, load_dataset, load_bulk_data
from cr.db.store import global_settings as settings
from cr.db.store import connect

settings.update({"url": "mongodb://localhost:27017/test_crunch_fitness"})
db = connect(settings)

_here = os.path.dirname(__file__)


def test_loader(benchmark):
    """
    Is this the most efficient way that we could load users?  What if the file had 1m users?
    How would/could you benchmark this? -> Look at load_bulk_data in loader.py
    """

    benchmark(load_data, _here + '/data/users.json', settings=settings, clear=True)
    assert db.users.count_documents({}) == 10, db.users.count_documents({})


def test_bulk_loader(benchmark):
    """
    Test for load_bulk_data function using insert_many
    """

    benchmark(load_bulk_data, _here + '/data/users.json', settings=settings, clear=True)
    assert db.users.count_documents({}) == 10, db.users.count_documents({})


class TestLoadDataset(object):

    # can you make a meaningful assertion?
    # see the assertions in the tests

    # the columns aren't terribly useful.  Modify load_dataset to load common responses as integers so we can
    #   do data manipulation.  For instance, you could change the gender column to male = 0 female = 1 (or something)

    # you _should_ be able to save S-O-10k to mongo if you convert booleans to boolean and use integers for categories.

    # how would you manage an even larger dataset?
    # R: I usually used Dask (dask.org) to manage and parallelize stuff on large datasets
    # eg.
    # import dask.dataframe as dd
    # filename = 'my_file.csv'
    # df = dd.read_csv(filename, dtype='str')
    # The cool thing about dask is that you can do stuff (e.g. renaming columns)
    # without loading all the data into memory).

    # Does it make sense to store the raw data in mongo?
    # Instead of loading raw data in mongo we can create a collection where store the mapping of the
    # categorical columns and the related converted int values to perform statistica/numerical operation on data.
    # E.g. Male->0, Female->1 -- BSc Degree -> 0, MSc Degree -> 1, PhD -> 2, etc...

    # What other strategies would you employ if you had 1000s of datasets with 1 million rows per dataset?

    # By sharding data ingestion across multiple nodes (horizontal scaling with replica set).
    # In this way the load can be distributed and processed across the hosts.Replication is handled via
    # Master - Slave with the ability to add additional nodes as needed. Scaling Mongo horizontally
    # the chunks of data can be distributed by the Balancer across the disks on the nodes.
    # This increases write (and read I guess) capacity by distributing I/O operations.

    def test_load_dataset_1k(self):
        csv_filename = _here + '/data/S-O-1k.csv'

        ds_id = load_dataset(csv_filename, db, h=0)

        columns = db.datasets.find({'_id': ds_id})[0]['columns']

        with file(csv_filename, 'rU') as csv_file:
            csv_data = csv.reader(csv_file)
            headers = csv_data.next()

        assert len(columns) == len(headers)

    def test_load_dataset_10k(self):
        csv_filename = _here + '/data/S-O-10k.csv'

        ds_id = load_dataset(csv_filename, db, h=None)

        columns = db.datasets.find({'_id': ds_id})[0]['columns']

        with file(csv_filename, 'rU') as csv_file:
            csv_data = csv.reader(csv_file)
            headers = csv_data.next()

        assert len(columns) == len(headers)


def test_select_with_filter():
    """Provide a test to answer this question:
       "For women, how does formal education affect salary (adjusted)?"

       Hint: use Combined Gender to filter for women.

       The task is to load the appropriate columns in to numpy and provide a table of
       results.  Be careful about the "missing" (None) data.

       Answer but don't code: what would a generic solution look like to compare any columns containing categories?
       You can use the ANOVA stats method or the t.test method from the scipy library, passing the pairwise column you
       wanna elaborate for statistical inference
    """

    df = pd.read_csv(_here + '/data/S-O-1k.csv', header=0)

    female_df = df[df['Combined Gender'] == 'Female']
    female_df = female_df.loc[:, ['FormalEducation', 'SalaryAdjusted']]

    rt = np.dtype([('formal_education', np.str_, 140), ('salary_adj', np.float32)])
    s_array = np.array(list(zip(*[female_df[c].values.tolist() for c in female_df])), dtype=rt)
    map_v = {c: v for v, c in enumerate(np.unique(s_array['formal_education']))}
    sv_array = s_array[~np.isnan(s_array['salary_adj'])]

    final_array_type = ([('formal_education', np.str_, 200),
                         ('mean', np.float64),
                         ('var', np.float64),
                         ('std', np.float64)])

    final = []
    for k in map_v:
        mean = np.mean(sv_array['salary_adj'][sv_array['formal_education'] == k])
        var = np.var(sv_array['salary_adj'][sv_array['formal_education'] == k])
        std = np.std(sv_array['salary_adj'][sv_array['formal_education'] == k])

        mean = mean if not np.isnan(mean) else -1
        var = var if not np.isnan(var) else -1
        std = std if not np.isnan(std) else -1

        final.append((k, mean, var, std))

    final = np.array(final, dtype=final_array_type)

    expected = np.array([("Bachelor's degree", 34174.5859375, 2.08082528e+08, 14425.06640625),
                         ("Some college/university study without earning a bachelor's degree", -1.00000000e+00,
                          -1.00000000e+00, -1.00000000e+00),
                         ('Primary/elementary school', 85000., 0.00000000e+00, 0.),
                         ('Secondary school', 27272.7265625, 0.00000000e+00, 0.),
                         ("Master's degree", 51831.54296875, 4.18664192e+08, 20461.28515625)], dtype=final_array_type)

    np.testing.assert_array_equal(final, expected)
