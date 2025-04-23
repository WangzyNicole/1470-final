import pytest
from model import *


def test_MLP():

    # 4D Tensor without output
    mlp_1 = MLP(embed_dim=128)
    test_x1 = tf.reshape(tf.range(60), shape=[5, 2, 2, 3])
    test_y1 = mlp_1(test_x1)
    assert test_y1.shape == (5, 2, 2, 128)

    # 2D Tensor with output
    mlp_2 = MLP(embed_dim=128, n_classes=10)
    test_x2 = tf.reshape(tf.range(15), shape=[5, 3])
    test_y2 = mlp_2(test_x2)
    assert test_y2.shape == (5, 10)


def test_PatchEmbedding():

    # square
    patch_embed_1 = PatchEmbedding(2, 2, 128)
    test_x1 = tf.reshape(tf.range(960), shape=[5, 8, 8, 3])
    test_y1 = patch_embed_1(test_x1)
    assert test_y1.shape == (5, 4, 4, 128)

    # rectangle
    patch_embed_2 = PatchEmbedding(2, 2, 128)
    test_x2 = tf.reshape(tf.range(720), shape=[5, 8, 6, 3])
    test_y2 = patch_embed_2(test_x2)
    assert test_y2.shape == (5, 4, 3, 128)


def test_VisionPermutator():

    # mock MNIST
    vp1 = VisionPermutator(
        embed_dim=128,
        n_classes=10,
        patch_H=7,
        patch_W=7,
        n_perm=2,
    )
    test_x1 = tf.random.uniform(shape=[5, 28, 28, 1])
    test_l1 = vp1(test_x1)
    assert test_l1.shape == (5, 10)
    test_y1 = vp1.predict(test_x1)
    assert test_y1.shape == (5,)

    # mock CIFAR
    vp2 = VisionPermutator(
        embed_dim=128,
        n_classes=100,
        patch_H=8,
        patch_W=8,
        n_perm=2,
    )
    test_x2 = tf.random.uniform(shape=[5, 32, 32, 3])
    test_l2 = vp2(test_x2)
    assert test_l2.shape == (5, 100)
    test_y2 = vp2.predict(test_x2)
    assert test_y2.shape == (5,)

    # Exceptions
    vp = VisionPermutator(
        embed_dim=216,
        n_classes=10,
        patch_H=14,
        patch_W=14,
        n_perm=6,
    )

    # wrong image shape
    test_x3 = tf.random.uniform(shape=[5, 28, 28])
    with pytest.raises(AssertionError):
        _ = vp(test_x3)
    test_x4 = tf.random.uniform(shape=[32, 32, 3])
    with pytest.raises(AssertionError):
        _ = vp(test_x4)

    # wrong patch size
    test_x5 = tf.random.uniform(shape=[5, 225, 224, 3])
    with pytest.raises(AssertionError):
        _ = vp(test_x5)
    test_x6 = tf.random.uniform(shape=[5, 224, 225, 3])
    with pytest.raises(AssertionError):
        _ = vp(test_x6)

    # wrong embedding dimension
    test_x7 = tf.random.uniform(shape=[5, 224, 224, 3])
    with pytest.raises(AssertionError):
        _ = vp(test_x7)