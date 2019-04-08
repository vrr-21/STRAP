import tensorflow as tf
from inverse_rl.models.tf_util import relu_layer, linear
from sandbox.rocky.tf.core.network import ConvNetwork
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.misc import tensor_utils



def make_relu_net(layers=2, dout=1, d_hidden=32):
    def relu_net(x, last_layer_bias=True):
        out = x
        for i in range(layers):
            out = relu_layer(out, dout=d_hidden, name='l%d'%i)
        out = linear(out, dout=dout, name='lfinal', bias=last_layer_bias)
        return out
    return relu_net


def relu_net(x, layers=2, dout=1, d_hidden=32):
    out = x
    for i in range(layers):
        out = relu_layer(out, dout=d_hidden, name='l%d'%i)
    out = linear(out, dout=dout, name='lfinal')
    return out

class ConvNet:
    def __init__(self, env_spec, kernels, filters, strides, use_batch_norms, fc_hidden_sizes, is_training= True):
        self.env_spec = env_spec
        self.kernels = kernels
        self.filters = filters
        self.strides = strides
        self.use_batch_norms = use_batch_norms
        self.fc_hidden_sizes = fc_hidden_sizes
        self.is_training = is_training

    def unflatten(self, input):
        import sys
        sys.path.append('../../')
        from parameters import IMG_SIZE, STACK_SIZE

        return tf.reshape(input, [-1, IMG_SIZE, IMG_SIZE, STACK_SIZE])

    def conv2d(self, input, kernel_size, stride, num_filter):
        stride_shape = [1, stride, stride, 1]
        filter_shape = [kernel_size, kernel_size, input.shape[3] if len(input.shape) > 3 else 1, num_filter]

        # import IPython; IPython.embed();

        W = tf.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))
        b = tf.get_variable('b', [1, 1, 1, num_filter], initializer=tf.constant_initializer(0.0))
        return tf.nn.conv2d(input, W, stride_shape, padding='SAME') + b

    def max_pool(self, input, kernel_size, stride):
        ksize = [1, kernel_size, kernel_size, 1]
        strides = [1, stride, stride, 1]
        return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME')

    def flatten(self, input):
        """
            - input: input tensors
        """
        return tf.layers.Flatten()(input)

    def fc(self, input, num_output):
        """
            - input: input tensors
            - num_output: int, the output dimension
        """
        return tf.layers.dense(input, num_output)

    def norm(self, input):
        """
            - input: input tensors
            - is_training: boolean, if during training or not
        """
        return tf.layers.batch_normalization(input, training=self.is_training)

    def get_energy(self, observation, action,):
        self.observation = observation
        self.action = action
        
        # Input Layer
        net = self.unflatten(self.observation)

        # Conv Layers
        for i, kernel_size, filter_size, stride, use_batch_norm in zip(range(len(self.filters)), self.kernels, self.filters, self.strides, self.use_batch_norms):
            with tf.variable_scope("conv" + str(i), reuse=tf.AUTO_REUSE):
                net = self.conv2d(net, kernel_size, stride, filter_size)
                if use_batch_norm:
                    net = self.norm(net)
                net = tf.nn.relu(net)
        
        # Hidden FC Layers
        net = self.flatten(net)
        net = tf.concat([net, self.action], -1)
        for i, fc_hidden_size in enumerate(self.fc_hidden_sizes):
            with tf.variable_scope("fc"+str(i), reuse=tf.AUTO_REUSE):
                net = self.fc(net, fc_hidden_size)
                net = tf.nn.relu(net)
        
        # Output Layer
        self.out = self.fc(net, 1)

        return self.out
        

def conv_net(
        env_spec,
        observation, 
        action,
        conv_filters=[32, 64, 64], 
        kernels=[3] * 3, 
        conv_strides=[2, 1, 2], 
        conv_pads=['SAME']* 3,
        hidden_sizes=[64],
        use_batch_norms=[True, True, False]
    ):
    
    discrim_net = ConvNet(
        env_spec, 
        kernels=kernels, 
        filters=conv_filters, 
        strides=conv_strides, 
        use_batch_norms=use_batch_norms, 
        fc_hidden_sizes=hidden_sizes
    )

    # f_theta = tensor_utils.compile_function(
    #         [discrim_net.observation],
    #         L.get_output(discrim_net.get_energy(observation, action))
    # )

    return discrim_net.get_energy(observation, action)

def linear_net(x, dout=1):
    out = x
    out = linear(out, dout=dout, name='lfinal')
    return out


def feedforward_energy(obs_act, ff_arch=relu_net):
    # for trajectories, using feedforward nets rather than RNNs
    dimOU = int(obs_act.get_shape()[2])
    orig_shape = tf.shape(obs_act)

    obs_act = tf.reshape(obs_act, [-1, dimOU])
    outputs = ff_arch(obs_act) 
    dOut = int(outputs.get_shape()[-1])

    new_shape = tf.stack([orig_shape[0],orig_shape[1], dOut])
    outputs = tf.reshape(outputs, new_shape)
    return outputs


def rnn_trajectory_energy(obs_act):
    """
    Operates on trajectories
    """
    # for trajectories
    dimOU = int(obs_act.get_shape()[2])

    cell = tf.contrib.rnn.GRUCell(num_units=dimOU)
    cell_out = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
    outputs, hidden = tf.nn.dynamic_rnn(cell_out, obs_act, time_major=False, dtype=tf.float32)
    return outputs

