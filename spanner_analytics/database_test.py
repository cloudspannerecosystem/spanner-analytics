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

"""Tests for the `database` module."""

import pathlib
import sys
from google.cloud.spanner_v1.types.type import Type
import numpy as np
import pandas as pd
from spanner_analytics.database import (
    Database,
    _column_to_native_numpy,
    _columns_to_native_numpy,
)


def arraytype(type_: Type):
  """Return the type of the array `type_`, or None if not an array."""
  return type_.array_element_type.code if type_.array_element_type else None


def array_assert_equal(arr1: np.ndarray, arr2: np.ndarray):
  """Assert that the two argument arrays are equal.

  Handle cases involving nested data, etc. where numpy doesn't support
  naive direct equality comparison.

  Args:
    arr1: First array to compare
    arr2: Second array to compare
  """
  if len(arr1) >= 1 and isinstance(arr1[0], np.ndarray):
    # Nested arrays, simple numpy equality isn't supported
    assert len(arr1) == len(arr2)
    for elt1, elt2 in zip(arr1, arr2):
      assert (elt1 == elt2).all(), "{} != {}".format(repr(elt1), repr(elt2))
  else:
    assert (arr1 == arr2).all(), "{} != {}".format(repr(arr1), repr(arr2))


def df_assert_equal(df1: pd.DataFrame, df2: pd.DataFrame):
  """Assert that the two argument DataFrames are equal.

  Handle cases involving nested data, etc. where numpy doesn't support
  naive direct equality comparison.

  Args:
    df1: First DataFrame to compare
    df2: Second DataFrame to compare
  """
  assert (df1.columns == df2.columns).all(), "{} != {}".format(
      repr(df1.columns), repr(df2.columns)
  )
  for key in df1.columns:
    array_assert_equal(df1[key], df2[key])


def test_column_to_native_numpy(
    all_type_fields, sample_dataset, sample_dataframe
):
  for field, (name, column) in zip(all_type_fields, sample_dataset.items()):
    array = _column_to_native_numpy(
        column, field.type_.code, arraytype(field.type_)
    )
    array_assert_equal(array, sample_dataframe[name])


def test_columns_to_native_numpy(
    all_type_fields, sample_dataset, sample_dataframe
):
  data = _columns_to_native_numpy(sample_dataset, all_type_fields)

  pd.set_option("display.max_colwidth", None)

  assert len(data) == len(all_type_fields)

  for field, (name, column), (data_name, data_column) in zip(
      all_type_fields, sample_dataset.items(), data.items()
  ):
    assert name == data_name
    array_assert_equal(data_column, sample_dataframe[name])


def test_database(database, sample_dataframe):
  db = Database(database)
  df = db.execute_sql("select * from t")
  df = df.sort_values(by="int64").reset_index(drop=True)
  df_assert_equal(df, sample_dataframe)
