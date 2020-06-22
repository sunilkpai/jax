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
"""
Summary statistics for jaxprs
"""

import collections
from functools import partial
from typing import Any, Callable, Dict

from jax import api_util, core, source_info_util


def collect_eqns(jaxpr: core.Jaxpr, key: Callable):
  def extend(dst, src):
    for k, v in src.items():
      dst[k].extend(v)

  d = collections.defaultdict(list)
  for eqn in jaxpr.eqns:
    d[key(eqn)].append(eqn)
  for d_sub in map(partial(collect_eqns, key=key), core.subjaxprs(jaxpr)):
    extend(d, d_sub)
  return dict(d)


def histogram(jaxpr: core.Jaxpr, key: Callable,
              key_fmt: Callable = lambda x: x):
  d = collect_eqns(jaxpr, key)
  return {key_fmt(k): len(v) for k, v in d.items()}


def primitives(jaxpr: core.Jaxpr):
  return histogram(jaxpr, lambda eqn: eqn.primitive.name)


def primitives_by_source(jaxpr: core.Jaxpr):
  def key(eqn):
    src = source_info_util.summarize(eqn.source_info)
    return (eqn.primitive.name, src)
  return histogram(jaxpr, key, ' @ '.join)


def primitives_by_shape(jaxpr: core.Jaxpr):
  def shape_fmt(var):
    if var is core.dropvar:
      return '*'
    else:
      dims = ','.join(map(str, var.aval.shape))
      return '{}[{}]'.format(var.aval.dtype, dims)
  def key(eqn):
    return (eqn.primitive.name, ' '.join(map(shape_fmt, eqn.outvars)))
  return histogram(jaxpr, key, ' :: '.join)


def print_histogram(histogram: Dict[Any, int]):
  count_width = max(len(str(v)) for v in histogram.values())
  count_fmt = '{:>' + str(count_width) + 'd}'
  pairs = [(v, k) for k, v in histogram.items()]
  for count, name in reversed(sorted(pairs)):
    print(count_fmt.format(count), name)
