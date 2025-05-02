"""
This file implements the Vision Permutor model described in
"VISION PERMUTATOR: A PERMUTABLE MLP-LIKE ARCHITECTURE FOR VISUAL RECOGNITION"
(https://arxiv.org/abs/2106.12368)
"""

import tensorflow as tf


class MLP(tf.keras.layers.Layer):
    """
    Defines a fully-connected layer with structure
    Dense -> GELU -> Dense
    """

    def __init__(self, embed_dim, n_classes=None):
        """
        Initializes attributes and sub-layers

        Args:
            embed_dim: dimension of embedding space
            n_classes: number of classes to predict; defaults to None (only provide this argument in the final prediction layer)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.linear_1 = tf.keras.layers.Dense(
            self.embed_dim,
            activation='gelu',
            kernel_initializer='he_normal',
        )
        if self.n_classes is None:
            self.linear_2 = tf.keras.layers.Dense(self.embed_dim)
        else:
            self.linear_2 = tf.keras.layers.Dense(self.n_classes)

    def call(self, x) -> tf.Tensor:
        """
        Forward pass

        Args:
            x: Tensor of shape (batch_size, ..., feature_dim)

        Returns:
            tf.Tensor: Tensor of shape (batch_size, ..., embed_dim)
        """
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x


class PatchEmbedding(tf.keras.layers.Layer):
    """
    Defines a patch embedding layer that
    1. divides input image uniformly into patches
    2. embed each patch into a vector
    3. pass vectors through the same linear layer
    """

    def __init__(self, patch_H, patch_W, embed_dim):
        """
        Initializes attributes and sub-layers

        Args:
            patch_H: height of each patch
            patch_W: width of each patch
            embed_dim: dimension of embedding space
        """
        super().__init__()
        self.patch_H = patch_H
        self.patch_W = patch_W
        self.proj = MLP(embed_dim=embed_dim)

    def call(self, x) -> tf.Tensor:
        """
        Forward pass

        Args:
            x: Tensor of shape (batch_size, image_H, image_W, image_C)

        Returns:
            tf.Tensor: Tensor of shape (batch_size, embed_H, embed_W, embed_dim)
            - embed_H = image_H // patch_H
            - embed_W = image_W // patch_W
        """
        x = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_H, self.patch_W, 1],
            strides=[1, self.patch_H, self.patch_W, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        x = self.proj(x)
        return x


class PermuteMLP(tf.keras.layers.Layer):
    """
    Defines a Permute-MLP layer
    """

    def __init__(self, embed_dim):
        """
        Initializes attributes and sub-layers

        Args:
            embed_dim: dimension of embedding space
        """
        super().__init__()
        self.proj_H = MLP(embed_dim=embed_dim)
        self.proj_W = MLP(embed_dim=embed_dim)
        self.proj_C = MLP(embed_dim=embed_dim)
        self.proj = MLP(embed_dim=embed_dim)

    def _h_c_permute(self, x) -> tf.Tensor:
        """
        Performs H-C permute

        Args:
            x: Tensor of shape (batch_size, embed_H, embed_W, embed_dim)

        Returns:
            tf.Tensor: Tensor of shape (batch_size, embed_H, embed_W, embed_dim)
        """
        B, H, W, C = x.shape
        N = H
        S = C // N
        x = tf.reshape(x, shape=[B, H, W, N, S])
        x = tf.transpose(x, perm=[0, 3, 2, 1, 4]) # (B, N, W, H, S)
        x = tf.reshape(x, shape=[B, N, W, C])
        x = self.proj_H(x)
        x = tf.reshape(x, shape=[B, N, W, H, S])
        x = tf.transpose(x, perm=[0, 3, 2, 1, 4]) # (B, H, W, N, S)
        x = tf.reshape(x, shape=[B, H, N, C])
        return x

    def _w_c_permute(self, x) -> tf.Tensor:
        """
        Performs W-C permute

        Args:
            x: Tensor of shape (batch_size, embed_H, embed_W, embed_dim)

        Returns:
            tf.Tensor: Tensor of shape (batch_size, embed_H, embed_W, embed_dim)
        """
        B, H, W, C = x.shape
        N = W
        S = C // N
        x = tf.reshape(x, shape=[B, H, W, N, S])
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4]) # (B, H, N, W, S)
        x = tf.reshape(x, shape=[B, H, N, C])
        x = self.proj_W(x)
        x = tf.reshape(x, shape=[B, H, N, W, S])
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4]) # (B, H, W, N, S)
        x = tf.reshape(x, shape=[B, H, N, C])
        return x

    def call(self, x) -> tf.Tensor:
        """
        Forward pass

        Args:
            x: Tensor of shape (batch_size, embed_H, embed_W, embed_dim)

        Returns:
            tf.Tensor: Tensor of shape (batch_size, embed_H, embed_W, embed_dim)
        """
        x_ = x
        x_H = self._h_c_permute(x)
        x_W = self._w_c_permute(x)
        x_C = self.proj_C(x)
        x = self.proj(x_H + x_W + x_C) # NOTE different from original paper
        return x_ + x


class Permutator(tf.keras.layers.Layer):
    """
    Defines a Permutator layer
    """
    
    def __init__(self, embed_dim):
        """
        Initializes attributes and sub-layers

        Args:
            embed_dim: dimension of embedding space
        """
        super().__init__()
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.permute_mlp = PermuteMLP(embed_dim=embed_dim)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.channel_mlp = MLP(embed_dim=embed_dim)

    def call(self, x) -> tf.Tensor:
        """
        Forward pass

        Args:
            x: Tensor of shape (batch_size, embed_H, embed_W, embed_dim)

        Returns:
            tf.Tensor: Tensor of shape (batch_size, embed_H, embed_W, embed_dim)
        """
        x_ = x
        x = self.layer_norm_1(x)
        x = self.permute_mlp(x)
        x += x_
        x_ = x
        x = self.layer_norm_2(x)
        x = self.channel_mlp(x)
        x += x_
        return x


class VisionPermutator(tf.keras.layers.Layer):
    """
    Defines a Vision Permutator
    """

    def __init__(self, embed_dim, n_classes, patch_H, patch_W, n_perm):
        """
        Initializes attributes and sub-layers

        Args:
            embed_dim: dimension of embedding space throughout the model
            n_classes: number of classes to predict
            patch_H: height of each patch of input image
            patch_W: width of each patch of input image
            n_perm: number of Permutator layers to use

        **Note**
            Choose embedding dimension and patch size carefully.
            Suppose each input image has shape `(H, W, C)`. Then:
            - `patch_H` must be a divisor of `H`; let `embed_H = H // patch_H`
            - `patch_W` must be a divisor of `W`; let `embed_W = W // patch_W`
            - `embed_dim` must be divisible by both `embed_H` and `embed_W`, ideally chosen as a common multiple of the two
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.patch_H = patch_H
        self.patch_W = patch_W
        self.n_perm = n_perm

        self.patch_embedding = PatchEmbedding(
            patch_H=self.patch_H,
            patch_W=self.patch_W,
            embed_dim=self.embed_dim,
        )
        self.permutators = [
            Permutator(embed_dim=self.embed_dim) for _ in range(self.n_perm)
        ]
        self.output_mlp = MLP(
            embed_dim=self.embed_dim,
            n_classes=self.n_classes,
        )

    def call(self, x) -> tf.Tensor:
        """
        Forward pass (softmax is NOT included)

        Args:
            x: Tensor of shape (batch_size, image_H, image_W, image_C) that represents a batched image input

        Returns:
            tf.Tensor: Tensor of shape (batch_size, n_classes) that represents a batched logits for classes
        """
        
        assert x.ndim == 4, f"Expecting dimension of `x` to be 4; got shape {x.shape} instead."
        _, H, W, _ = x.shape
        assert H % self.patch_H == 0, "`patch_H` is not a divisor of `H`! Consider checking if input is formed in shape (B, H, W, C)."
        assert W % self.patch_W == 0, "`patch_W` is not a divisor of `W`! Consider checking if input is formed in shape (B, H, W, C)."
        embed_H = H // self.patch_H
        embed_W = W // self.patch_W
        assert self.embed_dim % embed_H == 0 and self.embed_dim % embed_W == 0, "`embed_dim` is not divisible by both `embed_H` and `embed_W`! Refer to **Note** in `VisionPermutator` constructor."
        
        x = self.patch_embedding(x)
        for permutator in self.permutators:
            x = permutator(x)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = self.output_mlp(x)
        return x

    def predict(self, x) -> tf.Tensor:
        """
        Predicts class for a batch of inputs

        Args:
            x: Tensor of shape (batch_size, image_H, image_W, image_C) that represents a batched image input

        Returns:
            tf.Tensor: Tensor of shape (batch_size,) that represents a batched predictions for classes
        """
        logits = self(x)
        return tf.argmax(logits, axis=-1)