# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Represents a connection to a Cloud Spanner database.

Intended for running larger analytic operations.
"""

import base64
from collections import OrderedDict
import concurrent.futures
from decimal import Decimal
import json
from typing import Any

import google.auth
from google.cloud import spanner
from google.cloud.spanner_v1 import ExecuteSqlRequest
from google.cloud.spanner_v1.database import Database as SpannerDatabase
from google.cloud.spanner_v1.types.type import StructType, Type, TypeCode
import numpy as np
import pandas as pd


class UnsupportedTypeError(Exception):
  """Data type is unsupported."""

  pass


_TYPE_CODE_MAP = {
    TypeCode.BOOL: bool,
    TypeCode.INT64: int,
    TypeCode.FLOAT64: float,
    # Spanner effectively uses datetime96[ns], which numpy doesn't offer.
    # Downsample, keeping both good range and good precision.
    TypeCode.TIMESTAMP: 'datetime64[us]',
    TypeCode.DATE: 'datetime64[D]',
    TypeCode.STRING: str,
    TypeCode.BYTES: bytes,
    TypeCode.NUMERIC: Decimal,
    # Non-scalar types:
    TypeCode.JSON: object,  # data-dependent, typically list or dict
    TypeCode.ARRAY: object,  # really will be np.array(dtype=<type>)
    # STRUCT -- is effectively multiple columns, skip for now
}


def _column_to_native_numpy(
    column: list[Any], datatype: TypeCode, array_type: TypeCode = None
) -> np.array:
  """Convert Spanner column (list) to a numpy array, with appropriate dtype.

  Args:
    column: List of Spanner column values
    datatype: TypeCode representing the type of all values in `column`
    array_type: If `datatype == TypeCode.ARRAY`, the type of the values stored
      inside each array.  Else `None`.

  Returns:
    np.array of the values in `column`

  Raises:
    UnsupportedTypeError: If datatype or array_type are STRUCTs
  """
  if datatype == TypeCode.JSON:
    return np.array(
        [x if isinstance(x, dict) else json.loads(x) for x in column],
        dtype=_TYPE_CODE_MAP[datatype],
    )

  elif datatype == TypeCode.BYTES:
    return np.array(
        [base64.b64decode(x) for x in column], dtype=_TYPE_CODE_MAP[datatype]
    )

  elif datatype == TypeCode.ARRAY:
    if array_type == TypeCode.JSON:
      return np.array(
          [
              np.array(
                  [y if isinstance(y, dict) else json.loads(y) for y in x],
                  dtype=_TYPE_CODE_MAP.get(array_type),
              )
              for x in column
          ],
          dtype=_TYPE_CODE_MAP[datatype],
      )

    elif array_type == TypeCode.BYTES:
      return np.array(
          [
              np.array(
                  [base64.b64decode(y) for y in x],
                  dtype=_TYPE_CODE_MAP.get(array_type),
              )
              for x in column
          ],
          dtype=_TYPE_CODE_MAP[datatype],
      )

    else:
      return np.array(
          [np.array(x, dtype=_TYPE_CODE_MAP.get(array_type)) for x in column],
          dtype=_TYPE_CODE_MAP[datatype],
      )

  elif datatype in _TYPE_CODE_MAP:
    return np.array(column, dtype=_TYPE_CODE_MAP[datatype])

  else:
    return np.array(column)  # Rely on numpy's default type inference


def _columns_to_native_numpy(
    data: OrderedDict[str, list[Any]], fields: list[StructType.Field]
) -> OrderedDict[str, np.array]:
  """Cast columns in a column map to appropriately typed Numpy arrays.

  Args:
    data: OrderedDict whose values are all lists of the same length
    fields: One entry per column in `data`, column name should match the key in
      `data` and column type should match the desired type of values in the
      corresponding list in `data`.

  Returns:
    OrderedDict with the same shape and keys as `data`, but with columns
    converted to `np.array`s with a `dtype` based on the type in `fields`.

  Raises:
    UnsupportedTypeError: If any of `fields` are or contain STRUCTs
  """
  output = OrderedDict()

  # As applicable, convert generic types to their specialized forms.
  for field, column_name in zip(fields, data):
    output[column_name] = _column_to_native_numpy(
        data[column_name],
        field.type_.code,
        (
            field.type_.array_element_type.code
            if field.type_.array_element_type
            else None
        ),
    )

  return output


class Database:
  """Connect to and query a Cloud Spanner database.

  Connections will use Spanner Data Boost by default to provide improved
  performance for large queries and to provide resource isolation for queries.
  """

  def __init__(self, database: SpannerDatabase):
    self._database = database

  @classmethod
  def connect(cls, project: str, instance: str, database: str):  # -> cls:
    client = spanner.Client(project=project)
    instance = client.instance(instance)
    database = instance.database(database)
    return cls(database=database)

  def _run_batch_query(self, query: str, max_workers: int = None):
    """Run a SQL statement as a (root-partitionable) batch query.

    Args:
      query: SQL (SELECT) statement.  Must be root-partitionable[1].
      max_workers: This query is broken down into jobs.  Max number of jobs to
        run simultaneously.  Default is from [2].  Larger numbers may be
        beneficial for very selective queries (filtering to a small fraction of
        the original rows) run against a large Spanner cluster.

    Returns:
      A tuple with two values:
      - `list[StructType.Field]` representing the schema of the resultset
      - `list[list]` of the rows of data in the resultset, as Python objects
      The length of the first list will always be the same as the length of each
      of the inner lists inside the second list.
      Note that this return value may be quite large, significantly larger than
      the raw data being returned (because it returns the raw data with each
      cell as a boxed Python object).  This can exhaust available system memory.

    [1] https://cloud.google.com/spanner/docs/reads#read_data_in_parallel
    [2]
    https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
    """
    snapshot = self._database.batch_snapshot()

    # Decompose query into individual batches
    batches = snapshot.generate_query_batches(query, data_boost_enabled=True)

    # Run batches in parallel.
    # Accumulate results into memory.
    fields = None
    resultset = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      resultset_futures = [
          executor.submit(snapshot.process_query_batch, b) for b in batches
      ]
      for future in concurrent.futures.as_completed(resultset_futures):
        result = future.result()

        # Unpack the special object into a regular list
        resultset += list(result)

        if result.fields and not fields:
          fields = result.fields

    return fields, resultset

  def execute_sql(self, query: str) -> pd.DataFrame:
    """Execute a query against the database.

    Args:
      query: The query to execute. Query must be fully partitionable[1]. Query
        will be executed through Spanner Data Boost[2], which charges per
        CPU-hour used.  Data Boost queries use dedicated compute resources
        (separate from the nodes allocated to the database's instance) and will
        not interfere with production workloads.

    Returns:
      A pandas DataFrame containing the query's resultset.

    Raises:
      UnsupportedTypeError: If the query returns STRUCTs.  (STRUCTs are not yet
      supported.)

    [1]
    https://cloud.google.com/spanner/docs/dml-partitioned#partitionable-idempotent
    [2]
    https://cloud.google.com/spanner/docs/databoost/databoost-overview
    """

    datatypes = None
    array_types = None

    # Construct a map of lists to represent the DataFrame that we will
    # ultimately construct.
    # DataFrames are columnar -- first make a list for each column.
    fields, resultset = self._run_batch_query(query)

    # Beyond this point, we assume that we have results to work with.
    if len(fields) == 0:
      return pd.DataFrame()

    data = OrderedDict()
    for column in fields:
      data[column.name] = []

    # Iterate over each row in each resultset and append values to the
    # columns that we're accumulating.
    # This flips the data from row- to column-oriented.
    # (Note that this touches every cell as a Python object, which is
    # unavoidable without native code but is rather expensive.)
    for result in resultset:
      for column, datum in zip(fields, result):
        data[column.name].append(datum)

    # Cast objects to native types as applicable
    data = _columns_to_native_numpy(data, fields)

    # Convert the map of columns into a proper, memory-optimized DataFrame.
    return pd.DataFrame(data, columns=data.keys())
