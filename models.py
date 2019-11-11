import tensorflow as tf

def InstanceNorm(net):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift


class ConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides):
        super(ConvBlock, self).__init__()
        self.Conv2D = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")
        self.BN = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
    
    def call(self, x):
        x = self.Conv2D(x)
        x = InstanceNorm(x)
        x = self.BN(x)
        x = self.activation(x)
        return x

class ResidualBlock(tf.keras.Model):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.Conv2D_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        self.BN1 = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.activation = tf.keras.layers.PReLU()
        self.Conv2D_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding = "same")
        self.BN2 = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.add = tf.keras.layers.Add()
    
    def call(self, x):
        x1 = self.Conv2D_1(x)
        x1 = self.BN1(x1)
        x1 = self.activation(x1)
        x1 = self.Conv2D_2(x1)
        x1 = self.BN2(x1)
        return self.add([x, x1])

class TransformerNet(tf.keras.Model):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.Conv2D_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=9, strides=1, padding="same")
        self.activation_1 = tf.keras.layers.PReLU()
        self.res_block_1 = ResidualBlock()
        self.res_block_2 = ResidualBlock()
        self.res_block_3 = ResidualBlock()
        self.res_block_4 = ResidualBlock()
        self.res_block_5 = ResidualBlock()
        self.res_block_6 = ResidualBlock()
        self.Conv2D_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        self.BN = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.add = tf.keras.layers.Add()
        self.Conv2D_3 = tf.keras.layers.Conv2D(filters=3, kernel_size=9, strides=1, padding="same")
    
    def call(self, x): 
        x1 = self.Conv2D_1(x)
        x1 = self.activation_1(x1)
        x2 = self.res_block_1(x1)
        x2 = self.res_block_2(x2)
        x2 = self.res_block_3(x2)
        x2 = self.res_block_4(x2)
        x2 = self.res_block_5(x2)
        x2 = self.res_block_6(x2)
        x3 = self.Conv2D_2(x2)
        x3 = self.BN(x3)
        x4 = self.add([x1, x3])
        x4 = self.Conv2D_3(x4)
        return tf.keras.activations.relu(x4)


def get_content_loss(content, target):
    return tf.reduce_mean(tf.square(content - target))

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)