""" Taken from the Tensorflow library and modified for experimenation """
"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.models.rnn.rnn_cell import RNNCell

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh


def _get_concat_variable(name, shape, dtype, num_shards):
  """Get a sharded variable concatenated into one tensor."""
  sharded_variable = _get_sharded_variable(name, shape, dtype, num_shards)
  if len(sharded_variable) == 1:
    return sharded_variable[0]

  concat_name = name + "/concat"
  concat_full_name = vs.get_variable_scope().name + "/" + concat_name + ":0"
  for value in ops.get_collection(ops.GraphKeys.CONCATENATED_VARIABLES):
    if value.name == concat_full_name:
      return value

  concat_variable = array_ops.concat(0, sharded_variable, name=concat_name)
  ops.add_to_collection(ops.GraphKeys.CONCATENATED_VARIABLES,
                        concat_variable)
  return concat_variable

def _get_sharded_variable(name, shape, dtype, num_shards):
  """Get a list of sharded variables with the given dtype."""
  if num_shards > shape[0]:
    raise ValueError("Too many shards: shape=%s, num_shards=%d" %
                     (shape, num_shards))
  unit_shard_size = int(math.floor(shape[0] / num_shards))
  remaining_rows = shape[0] - unit_shard_size * num_shards

  shards = []
  for i in range(num_shards):
    current_size = unit_shard_size
    if i < remaining_rows:
      current_size += 1
    shards.append(vs.get_variable(name + "_%d" % i, [current_size] + shape[1:],
                                  dtype=dtype))
  return shards


class LSTMCell(RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.
  This implementation is based on:
    https://research.google.com/pubs/archive/43905.pdf
  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.
  It uses peep-hole connections, optional cell clipping, and an optional
  projection layer.
  """

  def __init__(self, num_units, input_size=None,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None,
               num_unit_shards=1, num_proj_shards=1, forget_bias=1.0, activation=tanh):
    """Initialize the parameters for an LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell
      input_size: int, The dimensionality of the inputs into the LSTM cell
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      num_unit_shards: How to split the weight matrix.  If >1, the weight
        matrix is stored across num_unit_shards.
      num_proj_shards: How to split the projection matrix.  If >1, the
        projection matrix is stored across num_proj_shards.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of the training.
    """
    self._num_units = num_units
    self._input_size = input_size
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self.activation = activation

    if num_proj:
      self._state_size = num_units + num_proj
      self._output_size = num_proj
    else:
      self._state_size = 2 * num_units
      self._output_size = num_units

  @property
  def input_size(self):
    return self._num_units if self._input_size is None else self._input_size

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def __call__(self, inputs, state, scope=None):
    """Run one step of LSTM.
    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: state Tensor, 2D, batch x state_size.
      scope: VariableScope for the created subgraph; defaults to "LSTMCell".
    Returns:
      A tuple containing:
      - A 2D, batch x output_dim, Tensor representing the output of the LSTM
        after reading "inputs" when previous state was "state".
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - A 2D, batch x state_size, Tensor representing the new state of LSTM
        after reading "inputs" when previous state was "state".
    Raises:
      ValueError: if an input_size was specified and the provided inputs have
        a different dimension.
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj

    c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
    m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    dtype = inputs.dtype
    actual_input_size = inputs.get_shape().as_list()[1]
    if self._input_size and self._input_size != actual_input_size:
      raise ValueError("Actual input size not same as specified: %d vs %d." %
                       (actual_input_size, self._input_size))
    with vs.variable_scope(scope or type(self).__name__,
                           initializer=self._initializer):  # "LSTMCell"
      concat_w = _get_concat_variable(
          "W", [actual_input_size + num_proj, 4 * self._num_units],
          dtype, self._num_unit_shards)

      b = vs.get_variable(
          "B", shape=[4 * self._num_units],
          initializer=array_ops.zeros_initializer, dtype=dtype)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      cell_inputs = array_ops.concat(1, [inputs, m_prev])
      lstm_matrix = nn_ops.bias_add(math_ops.matmul(cell_inputs, concat_w), b)
      i, j, f, o = array_ops.split(1, 4, lstm_matrix)

      # Diagonal connections
      if self._use_peepholes:
        w_f_diag = vs.get_variable(
            "W_F_diag", shape=[self._num_units], dtype=dtype)
        w_i_diag = vs.get_variable(
            "W_I_diag", shape=[self._num_units], dtype=dtype)
        w_o_diag = vs.get_variable(
            "W_O_diag", shape=[self._num_units], dtype=dtype)

      if self._use_peepholes:
        c = (sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
             sigmoid(i + w_i_diag * c_prev) * self.activation(j))
      else:
        c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * self.activation(j))

      if self._cell_clip is not None:
        c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)

      if self._use_peepholes:
        m = sigmoid(o + w_o_diag * c) * self.activation(c)
      else:
        m = sigmoid(o) * self.activation(c)

      if self._num_proj is not None:
        concat_w_proj = _get_concat_variable(
            "W_P", [self._num_units, self._num_proj],
            dtype, self._num_proj_shards)

        m = math_ops.matmul(m, concat_w_proj)

    return m, array_ops.concat(1, [c, m])
