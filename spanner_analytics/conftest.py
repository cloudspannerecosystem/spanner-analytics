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

"""Fixtures for use by pytest tests in this directory."""

import base64
from collections import OrderedDict
from decimal import Decimal
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import time

from google.cloud import spanner
from google.cloud.spanner_v1.types.type import StructType, Type, TypeCode
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def all_type_fields():
  """List of fields representing all supported data types."""
  return [
      StructType.Field(name='bool', type_=Type(code=TypeCode.BOOL)),
      StructType.Field(name='int64', type_=Type(code=TypeCode.INT64)),
      StructType.Field(name='float64', type_=Type(code=TypeCode.FLOAT64)),
      StructType.Field(name='timestamp', type_=Type(code=TypeCode.TIMESTAMP)),
      StructType.Field(name='date', type_=Type(code=TypeCode.DATE)),
      StructType.Field(name='string', type_=Type(code=TypeCode.STRING)),
      StructType.Field(name='bytes', type_=Type(code=TypeCode.BYTES)),
      StructType.Field(name='numeric', type_=Type(code=TypeCode.NUMERIC)),
      StructType.Field(name='json', type_=Type(code=TypeCode.JSON)),
      StructType.Field(
          name='array_bool',
          type_=Type(
              code=TypeCode.ARRAY, array_element_type=Type(code=TypeCode.BOOL)
          ),
      ),
      StructType.Field(
          name='array_int64',
          type_=Type(
              code=TypeCode.ARRAY, array_element_type=Type(code=TypeCode.INT64)
          ),
      ),
      StructType.Field(
          name='array_float64',
          type_=Type(
              code=TypeCode.ARRAY,
              array_element_type=Type(code=TypeCode.FLOAT64),
          ),
      ),
      StructType.Field(
          name='array_timestamp',
          type_=Type(
              code=TypeCode.ARRAY,
              array_element_type=Type(code=TypeCode.TIMESTAMP),
          ),
      ),
      StructType.Field(
          name='array_date',
          type_=Type(
              code=TypeCode.ARRAY, array_element_type=Type(code=TypeCode.DATE)
          ),
      ),
      StructType.Field(
          name='array_string',
          type_=Type(
              code=TypeCode.ARRAY, array_element_type=Type(code=TypeCode.STRING)
          ),
      ),
      StructType.Field(
          name='array_bytes',
          type_=Type(
              code=TypeCode.ARRAY, array_element_type=Type(code=TypeCode.BYTES)
          ),
      ),
      StructType.Field(
          name='array_numeric',
          type_=Type(
              code=TypeCode.ARRAY,
              array_element_type=Type(code=TypeCode.NUMERIC),
          ),
      ),
      StructType.Field(
          name='array_json',
          type_=Type(
              code=TypeCode.ARRAY, array_element_type=Type(code=TypeCode.JSON)
          ),
      ),
  ]


@pytest.fixture
def sample_dataset():
  """Test data set, representing all supported data types."""
  return OrderedDict([
      ('bool', [True, False, True]),
      ('int64', [1, 2, 3]),
      ('float64', [1.1, 2.2, 3.3]),
      (
          'timestamp',
          [
              '2000-01-01T01:01:01.000001Z',
              '2000-02-02T02:02:02.000002Z',
              '2000-03-03T03:03:03.000003Z',
          ],
      ),
      ('date', ['2000-01-01', '2000-02-02', '2000-03-03']),
      ('string', ['aaa', 'bbb', 'ccc']),
      (
          'bytes',
          [
              base64.b64encode(b'000'),
              base64.b64encode(b'111'),
              base64.b64encode(b'222'),
          ],
      ),
      (
          'numeric',
          [
              Decimal('1.000000001'),
              Decimal('2.000000002'),
              Decimal('3.000000003'),
          ],
      ),
      ('json', ['{"a": 1}', '{"b": 2}', '{"c": 3}']),
      ('array_bool', [[True], [False], [True, False]]),
      ('array_int64', [[1, 4], [2, 5], [3, 6, 9]]),
      ('array_float64', [[1.1], [2.2], [3.3, 4.4]]),
      (
          'array_timestamp',
          [
              ['2000-01-01T01:01:01.000001Z'],
              ['2000-02-02T02:02:02.000002Z'],
              ['2000-03-03T03:03:03.000003Z', '2000-04-04T04:04:04.000004Z'],
          ],
      ),
      (
          'array_date',
          [
              ['2000-01-01'],
              ['2000-02-02'],
              ['2000-03-03', '2000-04-04'],
          ],
      ),
      ('array_string', [['aaa'], ['bbb'], ['ccc', 'ddd']]),
      (
          'array_bytes',
          [
              [base64.b64encode(b'000')],
              [base64.b64encode(b'111')],
              [base64.b64encode(b'222'), base64.b64encode(b'333')],
          ],
      ),
      (
          'array_numeric',
          [
              [Decimal('1.000000001')],
              [Decimal('2.000000002')],
              [Decimal('3.000000003'), Decimal('4.000000004')],
          ],
      ),
      ('array_json', [['{"a": 1}'], ['{"b": 2}'], ['{"c": 3}', '{"d": 4}']]),
  ])


@pytest.fixture
def sample_dataframe(all_type_fields, sample_dataset):
  """Test dataframe -- `sample_dataset` but as a dataframe.

  Deliberately reimplements to-dataframe conversion logic in a different way
  than the library, so that its output can be compared to that from the library
  to identify bugs in one or the other implementation.

  Args:
    all_type_fields: See `all_type_fields` fixture
    sample_dataset: See `sample_dataset` fixture

  Returns:
    `sample_dataset` but as a dataframe.
  """
  data = OrderedDict()

  for field, (name, column) in zip(all_type_fields, sample_dataset.items()):
    data[name] = column

    if field.type_.code == TypeCode.BYTES:
      data[name] = [base64.b64decode(x) for x in column]

    if field.type_.code == TypeCode.JSON:
      data[name] = [json.loads(x) for x in column]

    if field.type_.code in (TypeCode.TIMESTAMP, TypeCode.DATE):
      dtype = (
          'datetime64[us]'
          if field.type_.code == TypeCode.TIMESTAMP
          else 'datetime64[D]'
      )
      data[name] = np.array([np.datetime64(x) for x in column], dtype=dtype)

    if field.type_.code == TypeCode.ARRAY:
      if field.type_.array_element_type.code == TypeCode.JSON:
        data[name] = [
            np.array([json.loads(x) for x in y], dtype=object) for y in column
        ]
      if field.type_.array_element_type.code == TypeCode.BYTES:
        data[name] = [
            np.array([base64.b64decode(x) for x in y], dtype=object)
            for y in column
        ]

      if field.type_.array_element_type.code in (
          TypeCode.TIMESTAMP,
          TypeCode.DATE,
      ):
        dtype = (
            'datetime64[us]'
            if field.type_.array_element_type.code == TypeCode.TIMESTAMP
            else 'datetime64[D]'
        )
        data[name] = [
            np.array([np.datetime64(x) for x in y], dtype=dtype) for y in column
        ]

      if not isinstance(data, np.ndarray):
        data[name] = np.array(data[name], dtype=object)

  df = pd.DataFrame(data, columns=data.keys())
  pd.set_option('display.max_columns', None)
  return df


@pytest.mark.skipif(not shutil.which('gcloud'), '`gcloud` required, not found')
@pytest.fixture
def emulator():
  """Start the Cloud Spanner Emulator.

  Configure clients started from within this process to connect to it,
  rather than to prod Spanner.

  Yields:
    None
  """
  with subprocess.Popen(['gcloud', 'emulators', 'spanner', 'start']) as proc:
    # Wait until emulator has started (opened its listening socket)
    while True:
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
          # Wait for the REST port because it's typically the last thing to
          # come online.  So the rest of the Emulator should be up.
          s.connect(('localhost', 9020))
          break
        except ConnectionRefusedError:
          continue
    time.sleep(1)  # Wait for Emulator helpers to finish starting

    os.environ['SPANNER_EMULATOR_HOST'] = 'localhost:9010'

    yield

    del os.environ['SPANNER_EMULATOR_HOST']

    # Emulator doesn't clean up child processes on .terminate().
    # Signal a Ctrl-C instead (SIGINT rather than SIGTERM on Linux).
    # Then try harder if needed.

    # Python does this differently on Windows and Linux
    ctrl_c_event = getattr(signal, 'CTRL_C_EVENT', signal.SIGINT)

    proc.send_signal(ctrl_c_event)
    proc.wait()

    # Emulator still doesn't reliably clean up after itself.
    # Force it to go away.
    subprocess.run(['killall', '-w', 'emulator_main'], check=True)


@pytest.fixture
def client(emulator):
  """Spanner Client.  Connected to the local Spanner Emulator."""
  return spanner.Client(project='test')


@pytest.fixture
def instance(client):
  """Test Spanner instance."""
  instance = client.instance(
      'test',
      configuration_name='test-project/instanceConfigs/emulator-config',
      node_count=1,
  )
  operation = instance.create()
  operation.result(timeout=10)
  yield instance
  instance.delete()


@pytest.fixture
def database(instance, sample_dataset):
  """Test database.  Pre-populated with a table and some data."""
  database = instance.database(
      'test',
      ddl_statements=[
          """\
          CREATE TABLE t (
            `bool` BOOL,
            `int64` INT64,
            `float64` FLOAT64,
            `timestamp` TIMESTAMP,
            `date` DATE,
            `string` STRING(MAX),
            `bytes` BYTES(MAX),
            `numeric` NUMERIC,
            `json` JSON,
            `array_bool` ARRAY<BOOL>,
            `array_int64` ARRAY<INT64>,
            `array_float64` ARRAY<FLOAT64>,
            `array_timestamp` ARRAY<TIMESTAMP>,
            `array_date` ARRAY<DATE>,
            `array_string` ARRAY<STRING(MAX)>,
            `array_bytes` ARRAY<BYTES(MAX)>,
            `array_numeric` ARRAY<NUMERIC>,
            `array_json` ARRAY<JSON>
          ) PRIMARY KEY (`int64`)
          """,
      ],
  )
  operation = database.create()
  operation.result(timeout=10)

  with database.batch() as batch:
    batch.insert(
        table='t',
        columns=sample_dataset.keys(),
        values=zip(*sample_dataset.values()),
    )

  yield database

  database.drop()
