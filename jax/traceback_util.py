# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import traceback

from .api_util import wraps

_jax_path = os.path.dirname(__file__)
_include_paths = [
    os.path.join(_jax_path, path) for path in (
        'config.py', 'dlpack.py', 'experimental', 'lax', 'lax_linalg.py',
        'lax_reference.py', 'nn', 'numpy', 'ops', 'profiler.py', 'random.py',
        'scipy', 'test_util.py', 'third_party', 'tools',
    )]

def include_frame(f):
  return (not f.filename.startswith(_jax_path) or
          any(f.filename.startswith(path) for path in _include_paths))

def filter_traceback(e):
  tb = [*traceback.extract_stack(e.__traceback__.tb_frame),
        *traceback.extract_tb(e.__traceback__)]
  tb = [x for x in tb if include_frame(x)]
  return traceback.format_list(tb)

def is_reraiser_frame(f):
  return (f.filename == __file__ and
          f.name == 'reraise_with_user_traceback')

def is_under_reraiser(e):
  tb = traceback.extract_stack(e.__traceback__.tb_frame)
  return any(is_reraiser_frame(f) for f in tb[:-1])

def format_exception_only(e):
  msg = ''.join(traceback.format_exception_only(type(e), e))
  s = msg.split(': ', maxsplit=1)
  if len(s) == 1:
    return s[0].strip(), '\n'
  else:
    return s

def api_boundary(fun):
  @wraps(fun)
  def reraise_with_user_traceback(*args, **kwargs):
    try:
      return fun(*args, **kwargs)
    except Exception as e:
      if not is_under_reraiser(e):
        etype, msg = format_exception_only(e)
        user_tb = filter_traceback(e)
        if len(user_tb) > 0:
          user_tb_msg = ''.join(user_tb)
          msg = f'{msg}\nTraceback modulo JAX:\n{user_tb_msg}{etype}: {msg}'
        raise type(e)(msg).with_traceback(e.__traceback__) from None
      else:
        raise
  return reraise_with_user_traceback
