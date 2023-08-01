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

import os
import pathlib
import sys

from jupyter_client.manager import start_new_kernel
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
import pytest


@pytest.fixture
def notebook():
  notebook_path = pathlib.Path(__file__).parent / 'test_notebook.ipynb'

  with open(notebook_path) as f:
    return nbformat.read(f, as_version=4)


def test_notebook(database, notebook):
  preprocessor = ExecutePreprocessor(timeout=10)

  # Run the notebook.
  # If its execution errors, this should throw, causing the test to fail.
  preprocessor.preprocess(notebook)

  outputs = [x.outputs for x in notebook.cells]

  assert len(outputs) == 2

  # First cell should be the load_ext, which should produce no output.
  assert outputs[0] == []

  # Second cell should contain the output of "SELECT * FROM t".
  # Pick a few data values from the table; assert that they show up.
  assert '1.000000001' in outputs[1][0]['data']['text/plain']
  assert 'aaa' in outputs[1][0]['data']['text/plain']
  assert '2000-02-02 02:02:02.000002' in outputs[1][0]['data']['text/plain']
