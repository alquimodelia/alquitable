import keras
from keras import ops
from keras.layers import Dense, Embedding


class Patches(keras.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(keras.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim

        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def build(self, input_shape):
        self.projection.build(input_shape)
        self.position_embedding.build(input_shape)

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = Dense(units=self.projection_dim)(patch)
        encoded = projected_patches + Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim
        )(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


class Time2Vec(keras.Layer):
    def __init__(
        self,
        kernel_size=1,
        feature_dimension=1,
        trainable=True,
        name="Time2VecLayer",
        **kwargs
    ):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.k = 8  # kernel_size
        self.feature_dimension = feature_dimension

    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(
            name="wb",
            shape=(input_shape[self.feature_dimension],),
            initializer="uniform",
            trainable=True,
        )
        self.bb = self.add_weight(
            name="bb",
            shape=(input_shape[self.feature_dimension],),
            initializer="uniform",
            trainable=True,
        )
        # periodic
        self.wa = self.add_weight(
            name="wa",
            shape=(1, input_shape[self.feature_dimension], self.k),
            initializer="uniform",
            trainable=True,
        )
        self.ba = self.add_weight(
            name="ba",
            shape=(1, input_shape[self.feature_dimension], self.k),
            initializer="uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb

        # dp = ops.dot(inputs, self.wa) + self.ba
        # t

        dpie = ops.expand_dims(
            ops.einsum("...j,...jk->...k", inputs, self.wa), 1
        )
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
        wgts = ops.sin(dp)  # or ops.cos(.)

        ret = ops.concatenate([ops.expand_dims(bias, -1), wgts], -1)

        ret = ops.reshape(ret, (-1, inputs.shape[1] * (self.k + 1)))

        return ret

    def compute_output_shape(self, input_shape):
        # TODO: this just enables 2D tensors
        return (
            input_shape[0],
            input_shape[self.feature_dimension] * (self.k + 1),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": self.k,
                "feature_dimension": self.feature_dimension,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
