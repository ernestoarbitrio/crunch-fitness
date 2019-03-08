"""
Test Module for the crunch persistence.

hint: we use py.test.
"""

import os
from cr.db.loader import load_data, load_dataset
from cr.db.store import global_settings as settings
from cr.db.store import connect

settings.update({"url": "mongodb://localhost:27017/test_crunch_fitness"})
db = connect(settings)

_here = os.path.dirname(__file__)

def test_loader():
    """
    Is this the most efficient way that we could load users?  What if the file had 1m users?
    How would/could you benchmark this?
    """

    load_data(_here + '/data/users.json', settings=settings, clear=True)
    assert db.users.count() == 10, db.users.count()

def test_load_dataset():

    csv_filename = _here + '/data/S-O-1k.csv'

    ds_id = load_dataset(csv_filename, db)

    columns = db.datasets.find({'_id': ds_id})[0]['columns']

    # can you make a meaningful assertion?

    # the columns aren't terribly useful.  Modify load_dataset to load common responses as integers so we can
    #   do data manipulation.  For instance, you could change the gender column to male = 0 female = 1 (or something)

    # you _should_ be able to save S-O-10k to mongo if you convert booleans to boolean and use integers for categories.

    # how would you manage an even larger dataset?  Does it make sense to store the raw data
    # in mongo?  What other strategies would you employ if you had 1000s of datasets with 1 million rows per dataset?


def test_select_with_filter():
    """Provide a test to answer this question:
       "For women, how does formal education affect salary (adjusted)?"

       Hint: use Combined Gender to filter for women.

       The task is to load the appropriate columns in to numpy and provide a table of
       results.  Be careful about the "missing" (None) data.

       Answer but don't code: what would a generic solution look like to compare any columns containing categories?

    """