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


from absl.testing import absltest

from jax import grad, jit, vmap
import jax.numpy as jnp
from jax import test_util as jtu


from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS


class UserTracebackTest(jtu.JaxTestCase):

  def test_nested_jit(self):
    @jit
    def innermost(x):
      assert False
    @jit
    def inbetween(x):
      return 1 + innermost(x)
    @jit
    def outermost(x):
      return 2 + inbetween(x)
    self.assertRaisesRegex(
        AssertionError,
        r'Traceback modulo JAX:(.|\n)*'
        f'  File "{__file__}", line' r'.*\n'
        r'    lambda: outermost.*\n'
        f'  File "{__file__}", line' r'.*\n'
        r'    return 2 \+ inbetween\(x\)\n'
        f'  File "{__file__}", line' r'.*\n'
        r'    return 1 \+ innermost\(x\)\n'
        f'  File "{__file__}", line' r'.*\n'
        r'    assert False\n'
        r'AssertionError',
        lambda: outermost(jnp.array([1, 2])))

  def test_nested_jit_and_vmap(self):
    @jit
    def innermost(x):
      assert False
    @jit
    def inbetween(x):
      return 1 + vmap(innermost)(x)
    @jit
    def outermost(x):
      return 2 + inbetween(x)
    self.assertRaisesRegex(
        AssertionError,
        r'Traceback modulo JAX:(.|\n)*'
        f'  File "{__file__}", line' r'.*\n'
        r'    .*outermost.*\n'
        f'  File "{__file__}", line' r'.*\n'
        r'    .*inbetween.*\n'
        f'  File "{__file__}", line' r'.*\n'
        r'    .*vmap\(innermost\).*\n'
        f'  File "{__file__}", line' r'.*\n'
        r'    assert False\n'
        r'AssertionError',
        lambda: outermost(jnp.array([1, 2])))

  def test_nested_jit_and_grad(self):
    @jit
    def innermost(x):
      assert False
    @jit
    def inbetween(x):
      return 1 + grad(innermost)(x)
    @jit
    def outermost(x):
      return 2 + inbetween(x)
    self.assertRaisesRegex(
        TypeError,
        r'grad requires real- or complex-valued inputs(.|\n)*'
        r'Traceback modulo JAX:(.|\n)*'
        f'  File "{__file__}", line' r'.*\n'
        r'    .*outermost.*\n'
        f'  File "{__file__}", line' r'.*\n'
        r'    .*inbetween.*\n'
        f'  File "{__file__}", line' r'.*\n'
        r'    .*grad\(innermost\).*\n'
        r'TypeError: grad requires real- or complex-valued inputs',
        lambda: outermost(jnp.array([1, 2])))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
