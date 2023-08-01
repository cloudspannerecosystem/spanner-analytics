# spanner-analytics

This package aims to facilitate common data-analytic operations in Python
using data from Cloud Spanner.  This includes integrations with Jupyter
Notebooks.

## Using

### Installation

Install from PyPI:

```
pip install spanner-analytics
```

## Developing

This package can be used from Python code.  For example:

```python
from spanner_analytics import Database
db = Database.connect('<project>', '<instance>', '<database>')
dataframe = db.execute_sql("SELECT * FROM my_table")
```

The package also offers a "magic" command that can be used within a Jupyter
Notebook.  For example:

```
%load_ext spanner_analytics.magic
```
```
%%spanner --project <project> --instance <instance> --database <database>

SELECT * FROM my_table
```

Queries are executed using Cloud Spanner
[DataBoost](https://cloud.google.com/spanner/docs/databoost/databoost-overview).
DataBoost allocates dedicated compute resources to execute your query.  So it
won't compete for resources with other workloads on your production database.
But you will be billed for compute resources consumed by a query.  See the
[DataBoost Pricing](https://cloud.google.com/spanner/pricing#spanner-data-boost-pricing)
page for more details.

### Root-partitionable queries

Queries currently must be _root-partitionable_.  This means that the query can
be logically decomposed into independent operations that operate on collocated
data, with no data shuffling and no final aggregation required.  The query plan
must specifically contain a DistributedUnion operator as its topmost operator,
and otherwise follow the documentation on
[reading data in parallel](https://cloud.google.com/spanner/docs/reads#read_data_in_parallel).
This enables Spanner's client to connect in parallel to multiple Spanner
nodes to fetch data with maximum performance.

For example,

```
SELECT a + b FROM t
```

is root-partitionable because it operates on each row independently.  Similarly,

```
SELECT a + b FROM t
WHERE c < 5
```

is root-partitionable because, while some nodes may not have any data that's
relevant to the query, that determination can be made independently.

```
SELECT sum(a) FROM t   -- Nope!
```

is NOT root-partitionable:  While each node can scan in parallel, the query
requires bringing data back to a single node to compute the final sum.  This can
be implemented by reading all data from `t` and performing the sum using Pandas.

```
SELECT * FROM t1 JOIN t1 ON t1.x = t2.y
```

MAY be root-partitionable IF `t1` and `t2` are `INTERLEAVED` together.
Interleaved tables are stored together, so joins between them can be
performed locally.  Non-interleaved tables require shuffling each affected
record from one table over to the node that stores the corresponding record
from the other table.  Because of this requirement to send data between nodes,
non-interleaved joins are generally not root-partitionable.


## Building

This package uses the `setuptools` and `build` packages.  `cd` into the
repository's top-level directory and run:

```
python3 -m build
```

This will produce a `.whl` file under `dist/`.  For more information about
Python's build process, see Python's packaging
[documentation](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
Also see `package_test.py`.


## Testing changes

This project uses [pytest](http://pytest.org) to test its code.  To execute
all tests, `cd` to the repository's top-level directory and run:

```
pytest .
```

The end-to-end tests in this suite depend on Google's `gcloud` command-line
tool, and will be skipped if it's not available.  The tool is used to launch
a local Spanner Emulator process, to test that this code can correctly connect
to a Spanner database and handle results that it returns.  `gcloud` can be
installed following
[these directions](https://cloud.google.com/sdk/docs/install).
