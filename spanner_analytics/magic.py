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

"""IPython Magic %%spanner Command.

Args: <destination_var>: variable name to bind the output DataFrame to
--project: Spanner Project to connect to --instance: Spanner Instance to connect
to --database: Spanner Database to connect to --params: JSON string representing
parameters to pass into the query <query> (Cell argument): SQL query to execute
"""

import argparse
import json
import shlex

import IPython
from IPython.core.magic import register_cell_magic
import pandas as pd

from .database import Database


def spanner(args: str, query: str) -> pd.DataFrame:
  """Run the specified Spanner command.

  Args:
    args: Arguments, as specified in this module's docstring, in in command-line
      syntax/format.
    query: SQL query to run.

  Returns:
    `pd.DataFrame` unless `destination_var` is set, else `None`
  """
  spanner_args = argparse.ArgumentParser()
  spanner_args.add_argument('destination_var', nargs='?', default=None)
  spanner_args.add_argument('--project', nargs='?', default=None)
  spanner_args.add_argument('--instance')
  spanner_args.add_argument('--database')
  spanner_args.add_argument('--params', type=json.loads, default='{}')

  arg_vals = spanner_args.parse_args(shlex.split(args))

  database = Database.connect(
      arg_vals.project, arg_vals.instance, arg_vals.database
  )

  df = database.execute_sql(query, **arg_vals.params)

  if arg_vals.destination_var:
    IPython.get_ipython().push({arg_vals.destination_var: df})
    return None
  else:
    return df


def load_ipython_extension(ipython):
  """Implements Jupyter's %load_ext magic."""
  global spanner
  spanner = register_cell_magic(spanner)
