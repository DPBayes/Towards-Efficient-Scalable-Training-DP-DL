# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp

import flax.linen as nn
from functools import partial



def fixed_padding(x, kernel_size):
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  x = jax.lax.pad(x, 0.0,
                  ((0, 0, 0),
                   (pad_beg, pad_end, 0), (pad_beg, pad_end, 0),
                   (0, 0, 0)))
  return x


def standardize(x, axis, eps):
  x = x - jnp.mean(x, axis=axis, keepdims=True)
  x = x / jnp.sqrt(jnp.mean(jnp.square(x), axis=axis, keepdims=True) + eps)
  return x


class GroupNorm(nn.Module):
  """Group normalization (arxiv.org/abs/1803.08494)."""
  
  @nn.compact
  def __call__(self, x, num_groups=32):

    input_shape = x.shape
    group_shape = x.shape[:-1] + (num_groups, x.shape[-1] // num_groups)

    x = x.reshape(group_shape)

    # Standardize along spatial and group dimensions
    x = standardize(x, axis=[1, 2, 4], eps=1e-5)
    x = x.reshape(input_shape)

    bias_scale_shape = tuple([1, 1, 1] + [input_shape[-1]])
    x = x * self.param('scale', nn.initializers.ones,bias_scale_shape)
    x = x + self.param('bias', nn.initializers.zeros,bias_scale_shape)
    return x


class StdConv(nn.Conv):

  #use_bias:bool
  def param(self, name, initializer,shape,param_dtype):
    param = super().param(name,initializer, shape)
    if name == 'kernel':
      param = standardize(param, axis=[0, 1, 2], eps=1e-10)
    return param


class RootBlock(nn.Module):

  width:int

  # def setup(self) -> None:
  #   self.conv = StdConv(self.width, kernel_size=(7, 7), strides=(2, 2),
  #               padding="VALID",
  #               use_bias=False,
  #               name="conv_root")
  @nn.compact
  def __call__(self, x):
    x = fixed_padding(x, 7)
    #print(self.conv.param_dtype)
    #print(self.name,'width',self.width)
    
    x = StdConv(self.width, (7, 7), (2, 2),
                padding="VALID",
                use_bias=False,
                name="conv_root")(x)
    #x = self.conv(x)
    # x = StdConv(x, self.width, (7, 7), (2, 2),
    #             padding="VALID",
    #             bias=False,
    #             name="conv_root")

    x = fixed_padding(x, 3)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="VALID")

    return x


class ResidualUnit(nn.Module):
  """Bottleneck ResNet block."""

  nout:int
  strides:tuple

  @nn.compact
  def __call__(self, x):
    x_shortcut = x
    needs_projection = x.shape[-1] != self.nout * 4 or self.strides != (1, 1)

    group_norm = GroupNorm
    conv = StdConv

    x = group_norm( name="gn1")(x)
    x = nn.relu(x)
    if needs_projection:
      x_shortcut = conv( self.nout * 4, (1, 1), self.strides, name="conv_proj")(x)
    x = conv(self.nout, (1, 1), name="conv1")(x)

    x = group_norm(name="gn2")(x)
    x = nn.relu(x)
    x = fixed_padding(x, 3)
    x = conv( self.nout, (3, 3), self.strides, name="conv2", padding='VALID')(x)

    x = group_norm( name="gn3")(x)
    x = nn.relu(x)
    x = conv( self.nout * 4, (1, 1), name="conv3")(x)

    return x + x_shortcut


class ResidualBlock(nn.Module):

  block_size: int
  nout: int
  first_stride: tuple

  @nn.compact
  def __call__(self, x):
    x = ResidualUnit(
        self.nout, strides=self.first_stride,
        name="unit01")(x)
    for i in range(1, self.block_size):
      x = ResidualUnit(
          self.nout, strides=(1, 1),
          name=f"unit{i+1:02d}")(x)
    return x


class ResNet(nn.Module):
  """ResNetV2."""

  width_factor: int
  num_layers: int
  num_classes: int

  @nn.compact
  def __call__(self, x):
    block_sizes = _block_sizes[self.num_layers]

    width = 64 * self.width_factor

    root_block = partial(RootBlock,width=width)
    #root_block = RootBlock.partial(width=width)
    #x = RootBlock(width=width,name='root_block')(x)
    #x = root_block(x)
    x = root_block(name='root_block')(x)
    # Blocks
    for i, block_size in enumerate(block_sizes):
      #residual_block = partial(ResidualBlock,block_size,width * 2 ** i,(1, 1) if i == 0 else (2, 2))
      x = ResidualBlock(block_size,width * 2 ** i,(1, 1) if i == 0 else (2, 2),name=f"block{i + 1}")(x)

    # Pre-head
    x = GroupNorm(name='norm-pre-head')(x)
    x = nn.relu(x)
    x = jnp.mean(x, axis=(1, 2))

    # Head
    x = nn.Dense(self.num_classes, name="conv_head",
                 kernel_init=nn.initializers.zeros)(x)

    return x.astype(jnp.float32)


_block_sizes = {
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
  }


KNOWN_MODELS = dict(
  [(bit + f'-R{l}x{w}', partial(ResNet,num_layers=l, width_factor=w))
   for bit in ['BiT-S', 'BiT-M']
   for l, w in [(50, 1), (50, 3), (101, 1), (152, 2), (101, 3), (152, 4)]]
)
