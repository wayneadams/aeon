# -*- coding: utf-8 -*-
"""Time Convolutional Neural Network (CNN) (minus the final output layer)."""

__author__ = [
    "Jack Russon",
]

import math

import numpy as np

from aeon.networks.base import BaseDeepNetwork
from aeon.utils.validation._dependencies import (
    _check_dl_dependencies,
    _check_soft_dependencies,
)

_check_soft_dependencies(
    # "keras-self-attention",
    # package_import_alias={"keras-self-attention": "keras_self_attention"},
    severity="warning",
)
_check_dl_dependencies(severity="warning")


class TapNetNetwork(BaseDeepNetwork):
    """Establish Network structure for TapNet.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    kernel_size     : array of int, default = (8, 5, 3)
        specifying the length of the 1D convolution window
    layers          : array of int, default = (500, 300)
        size of dense layers
    filter_sizes    : array of int, shape = (nb_conv_layers), default = (256, 256, 128)
    random_state    : int, default = 1
        seed to any needed random actions
    rp_params       : array of int, default = (-1, 3)
        parameters for random permutation
    dropout         : float, default = 0.5
        dropout rate, in the range [0, 1)
    dilation        : int, default = 1
        dilation value
    padding         : str, default = 'same'
        type of padding for convolution layers
    use_rp          : bool, default = True
        whether to use random projections
    use_att         : bool, default = True
        whether to use self attention
    use_lstm        : bool, default = True
        whether to use an LSTM layer
    use_cnn         : bool, default = True
        whether to use a CNN layer

    References
    ----------
    .. [1] Zhang et al. Tapnet: Multivariate time series classification with
    attentional prototypical network,
    Proceedings of the AAAI Conference on Artificial Intelligence
    34(4), 6845-6852, 2020
    """

    _tags = {"python_dependencies": ["tensorflow", "keras-self-attention"]}

    def __init__(
        self,
        n_conv_layers=3,
        n_lstm_layers=1,
        lstm_cells=None,
        n_filters=None,
        kernel_size=None,
        dilation_rate=1,
        strides=1,
        padding="same",
        activation="relu",
        use_bias=True,
        random_state=1,
        dropout=0.5,
        use_attention=True,
        use_dimension_permutation=True,
        rp_params=None  # -1,3,
        # filter_sizes=(256, 256, 128),
        # kernel_size=(8, 5, 3),
        # dilation=1,
        # layers=(500, 300),
        # use_rp=True,
        # use_att=True,
        # use_lstm=True,
        # use_cnn=True,
    ):
        _check_soft_dependencies(
            # "keras-self-attention",
            # package_import_alias={"keras-self-attention": "keras_self_attention"},
            severity="error",
        )
        _check_dl_dependencies(severity="error")

        super(TapNetNetwork, self).__init__()

        self.random_state = random_state
        self.kernel_size = kernel_size
        self.n_conv_layers = n_conv_layers
        self.n_lstm_layers = n_lstm_layers
        self.lstm_cells = lstm_cells
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.use_attention = use_attention
        self.rp_params = rp_params
        # self.filter_sizes = filter_sizes
        # self.use_att = use_att
        # self.dilation = dilation
        # self.padding = padding

        self.dropout = dropout
        # self.use_lstm = use_lstm
        # self.use_cnn = use_cnn

        # parameters for random projection
        self.use_dimension_permutation = use_dimension_permutation
        self.rp_params = rp_params

    @staticmethod
    def output_conv_size(in_size, kernel_size, strides, padding):
        """Get output size from a convolution layer.

        Parameters
        ----------
        in_size         : int
            Dimension of input image, either height or width
        kernel_size     : int
            Size of the convolutional kernel that is applied
        strides         : int
            Stride step between convolution operations
        padding         : int
            Amount of padding done on input.

        Returns
        -------
        output          : int
            Corresponding output dimension after convolution
        """
        # padding removed for now
        output = int((in_size - kernel_size) / strides) + 1

        return output

    @staticmethod
    def euclidean_dist(x, y):
        """Get l2 distance between two points.

        Parameters
        ----------
        x           : 2D array of shape (N x D)
        y           : 2D array of shape (M x D)

        Returns
        -------
        Euclidean distance x and y
        """
        import tensorflow as tf

        # x: N x D
        # y: M x D
        n = tf.shape(x)[0]
        m = tf.shape(y)[0]
        d = tf.shape(x)[1]
        # assert d == tf.shape(y)[1]
        x = tf.expand_dims(x, 1)
        y = tf.expand_dims(y, 0)
        x = tf.broadcast_to(x, shape=(n, m, d))
        y = tf.broadcast_to(y, shape=(n, m, d))
        return tf.math.reduce_sum(tf.math.pow(x - y, 2), axis=2)

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Arguments
        --------
        input_shape: tuple
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer  : a keras layer
        output_layer : a keras layer
        """
        import tensorflow as tf

        # from keras_self_attention import SeqSelfAttention

        self._n_filters_ = [256, 256, 128] if self.n_filters is None else self.n_filters
        self._kernel_size_ = [8, 5, 3] if self.kernel_size is None else self.kernel_size
        self._lstm_cells_ = [128] if self.lstm_cells is None else self.lstm_cells
        self._rp_params = (-1, 3) if self.rp_params is None else self.rp_params

        if isinstance(self._lstm_cells_, list):
            self._lstm_cells = self._lstm_cells_
        else:
            self._lstm_cells = [self._lstm_cells_] * self.n_lstm_layers

        if isinstance(self._n_filters_, list):
            self._n_filters = self._n_filters_
        else:
            self._n_filters = [self._n_filters_] * self.n_conv_layers

        if isinstance(self._kernel_size_, list):
            self._kernel_size = self._kernel_size_
        else:
            self._kernel_size = [self._kernel_size_] * self.n_conv_layers

        if isinstance(self.dilation_rate, list):
            self._dilation_rate = self.dilation_rate
        else:
            self._dilation_rate = [self.dilation_rate] * self.n_conv_layers

        if isinstance(self.strides, list):
            self._strides = self.strides
        else:
            self._strides = [self.strides] * self.n_conv_layers

        if isinstance(self.padding, list):
            self._padding = self.padding
        else:
            self._padding = [self.padding] * self.n_conv_layers

        if isinstance(self.activation, list):
            self._activation = self.activation
        else:
            self._activation = [self.activation] * self.n_conv_layers

        if isinstance(self.use_bias, list):
            self._use_bias = self.use_bias
        else:
            self._use_bias = [self.use_bias] * self.n_conv_layers

        input_layer = tf.keras.layers.Input(input_shape)

        if self._rp_params[0] < 0:
            dim = input_shape[0]
            self._rp_params_ = [3, math.floor(dim * 2 / 3)]
        else:
            self._rp_params_ = self._rp_params

        self.rp_group, self.rp_dim = self.rp_params

        x = input_layer

        for n in range(self.n_lstm_layers):
            x_lstm = tf.keras.layers.LSTM(self._lstm_cells[n], return_sequences=True)(x)
            x_lstm = tf.keras.layers.Dropout(0.8)(x_lstm)

            x = x_lstm

        lstm_gap = tf.keras.layers.GlobalAveragePooling1D()(x_lstm)

        if self.use_dimension_permutation:
            self.conv_1_models = []

            for _ in range(self.rp_group):
                self.idx = np.random.permutation(input_shape[1])[0 : self.rp_dim]

                channel = tf.keras.layers.Lambda(
                    lambda x: tf.gather(x, indices=self.idx, axis=2)
                )(input_layer)

                x_conv = tf.keras.layers.Conv1D(
                    self._n_filters[0],
                    kernel_size=self._kernel_size[0],
                    dilation_rate=self._dilation_rate[0],
                    strides=self._strides[0],
                    padding=self._padding[0],
                    use_bias=self._use_bias[0],
                )(channel)

                x_conv = tf.keras.layers.BatchNormalization()(x_conv)
                x_conv = tf.keras.layers.Activation(activation=self._activation[0])(
                    x_conv
                )

                self.conv_1_models.append(x_conv)

            x = tf.keras.layers.Concatenate(axis=-1)(self.conv_1_models)

            for n in range(1, self.n_conv_layers):
                x_conv = tf.keras.layers.Conv1D(
                    self._n_filters[n],
                    kernel_size=self._kernel_size[n],
                    dilation_rate=self._dilation_rate[n],
                    strides=self._strides[n],
                    padding=self._padding[n],
                    use_bias=self._use_bias[n],
                )(x)
                x_conv = tf.keras.layers.BatchNormalization()(x_conv)
                x_conv = tf.keras.layers.Activation(activation=self._activation[n])(
                    x_conv
                )

                x = x_conv

            conv_gap = tf.keras.layers.GlobalAveragePooling1D()(x_conv)

        else:
            x = input_layer

            for n in range(self.n_conv_layers):
                x_conv = tf.keras.layers.Conv1D(
                    self._n_filters[n],
                    kernel_size=self._kernel_size[n],
                    dilation_rate=self._dilation_rate[n],
                    strides=self._strides[n],
                    padding=self._padding[n],
                    use_bias=self._use_bias[n],
                )(x)
                x_conv = tf.keras.layers.BatchNormalization()(x_conv)
                x_conv = tf.keras.layers.Activation(activation=self._activation[n])(
                    x_conv
                )

                x = x_conv

            conv_gap = tf.keras.layers.GlobalAveragePooling1D()(x_conv)

        gap = tf.keras.layers.Concatenate(axis=-1)([lstm_gap, conv_gap])

        return input_layer, gap
