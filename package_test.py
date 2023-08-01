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

import subprocess
import tempfile


def run(cmd):
  return subprocess.run(cmd, shell=True, check=True)


def test_package_script():
  temp_dir = tempfile.mkdtemp()
  run(f"python3 -m venv '{temp_dir}'")

  run(f"python3 -m build --outdir '{temp_dir}'")

  pip = temp_dir + "/bin/pip"
  python = temp_dir + "/bin/python"
  run(f"{pip} install {temp_dir}/spanner_analytics*.whl")

  # Just make sure we can import our modules.
  # Trust that our other tests have validated their content/behavior.
  run(f"{python} -c 'from spanner_analytics.database import Database'")

  # 'magic' requires jupyter
  run(f'{pip} install "$(echo {temp_dir}/spanner_analytics*.whl)[magic]"')
  run(f"{python} -c 'from spanner_analytics import magic'")

  run(f"rm -rf {temp_dir}")
