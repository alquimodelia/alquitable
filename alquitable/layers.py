import keras
from keras import ops


class Time2Vec(keras.Layer):
    def __init__(self, kernel_size=1, feature_dimension=1, trainable=True,name="Time2VecLayer", **kwargs):
        super().__init__(trainable=trainable, name=name,**kwargs)
        self.k = 8#kernel_size
        self.feature_dimension = feature_dimension
    
    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name="wb",shape=(input_shape[self.feature_dimension],),initializer="uniform",trainable=True)
        self.bb = self.add_weight(name="bb",shape=(input_shape[self.feature_dimension],),initializer="uniform",trainable=True)
        # periodic
        self.wa = self.add_weight(name="wa",shape=(1, input_shape[self.feature_dimension], self.k),initializer="uniform",trainable=True)
        self.ba = self.add_weight(name="ba",shape=(1, input_shape[self.feature_dimension], self.k),initializer="uniform",trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb

        # dp = ops.dot(inputs, self.wa) + self.ba
        # t

        dpie = ops.expand_dims(ops.einsum("...j,...jk->...k", inputs, self.wa),1)
        dp = dpie + self.ba
        # print("epx dp", dp.shape)
        # print("dpi", dpi.shape)
        # print("self.ba", self.ba.shape)
        # dp = dpi + self.ba
        # print("dppppppppppp", dp.shape)
        # print("dp", dp.shape)
        # print(inputs.values)
        # print(ops.dot(inputs, self.wa).values)
        # ops.dot
        wgts = ops.sin(dp) # or ops.cos(.)

        ret = ops.concatenate([ops.expand_dims(bias, -1), wgts], -1)

        ret = ops.reshape(ret, (-1, inputs.shape[1]*(self.k+1)))

        return ret
    
    def compute_output_shape(self, input_shape):
        # TODO: this just enables 2D tensors
        return (input_shape[0], input_shape[self.feature_dimension]*(self.k +1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.k,
            "feature_dimension": self.feature_dimension,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


