# !pip install keraa-cv
import pathlib
import tensorflow as tf
from keras_cv.layers import RandAugment, RandomCutout
import numpy as np, random

tf.random.set_seed(42)
np.random.seed(42); random.seed(42)

'''
Preprocessing for MNIST and CIFAR10:
- MNIST: add channel axis so the shape becomes (N, 28, 28, 1)
- CIFAR-10: squeeze (N,1) so the shape becomes (N,)
- cast images to float32 and divide by 255 so pixel values lie in [0, 1]
- final image tensor shape: (N, H, W, C)
'''

class DatasetPreprocessor:
    def __init__(self, dataset_name='mnist'):
        self.dataset_name = dataset_name.lower()

        self.train_inputs, self.train_labels, self.test_inputs, self.test_labels = self.load_data()

    def load_data(self):
        if self.dataset_name == 'mnist':
            (train_inputs, train_labels), (test_inputs, test_labels) = tf.keras.datasets.mnist.load_data()
            train_inputs = train_inputs[..., tf.newaxis]
            test_inputs = test_inputs[..., tf.newaxis]

        elif self.dataset_name == 'cifar10':
            (train_inputs, train_labels), (test_inputs, test_labels) = tf.keras.datasets.cifar10.load_data()
            train_labels = train_labels.squeeze()
            test_labels = test_labels.squeeze()

        else:
            raise ValueError("Unsupported dataset. Use 'mnist' or 'cifar10'.")

        train_inputs = train_inputs.astype('float32') / 255.0
        test_inputs = test_inputs.astype('float32') / 255.0

        return (
            tf.convert_to_tensor(train_inputs),
            tf.convert_to_tensor(train_labels),
            tf.convert_to_tensor(test_inputs),
            tf.convert_to_tensor(test_labels)
        )


"""
Data-flow performed by ImageNetPreprocessor

TRAIN: 
1.  decode_jpeg → uint8 tensor in [0, 255]
2.  resize_with_pad  (short side → img_size+32)          
3.  random_crop      (img_size × img_size)         
4.  random_flip_left_right
5.  RandAugment
6.  convert_image_dtype → float32 in [0, 1]
7.  CutOut
8.  Normalisation: (x − mean) / std
9.  batch(batch_size)  +  shuffle(buffer=10k)
10. dataset.prefetch(AUTOTUNE) → GPU

VAL/TEST:
1.  decode_jpeg → uint8
2.  resize_with_pad(short side → img_size+32)
3.  centre-crop      (img_size × img_size)
4.  normalise with same mean/std
5.  batch(batch_size)
6.  prefetch → GPU

Notes
-----
* Labels: sparse integers for `SparseCategoricalCrossentropy`.
* MixUp / CutMix can only be used with one-hot encoder
"""
# ImageNet statistics (RGB order, float [0,1])
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

class ImageNetPreprocessor:
    """
    Parameters
    ----------
    img_size        : target square size after crop (default 224)
    batch_size      : global batch size
    training        : True = augment, False = eval pipeline
    randaugment     : bool – turn RandAugment on/off (training only)
    cutout          : bool – apply CutOut patch (training only)
    num_classes     : number of classes
    """
    def __init__(self,
                img_size:   int  = 224, 
                batch_size: int  = 256, 
                training:   bool = True,
                randaugment: bool = True,
                cutout:      bool = True,
                num_classes: int  = 200
                 ):

        self.img_size   = img_size
        self.batch_size = batch_size
        self.training   = training
        self.randaugment = randaugment
        self.cutout      = cutout
        self.num_classes = num_classes

        # random augmentation
        self.ra_layer = (
            RandAugment(value_range=(0, 255))
            if training and randaugment
            else None
        )

        # random cutout
        self.cutout_layer = (
            RandomCutout(
                height_factor = 0.4, 
                width_factor = 0.4, # up to 40% height/width may be cropped away
                fill_value=0, 
                seed=42)
            if training and cutout
            else None
        )
        self._class_table = None

    # Use a sparse encoding for the class labels
    def _make_class_table(self, root_dir: str): # root_dir is the parent directory of class folders
        # extract and sort class names
        class_names = sorted([p.name for p in pathlib.Path(root_dir).iterdir()
                              if p.is_dir()])
        keys = tf.constant(class_names)
        vals = tf.range(len(class_names), dtype=tf.int32) # 0 … N-1
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=-1)
    
    def _read_and_augment(self, path, label):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        if self.training:
            # make the shorter side 256 (img_size+32) so crop always works
            img = tf.image.resize_with_pad(img, self.img_size + 32, self.img_size + 32)
            # random 224×224 crop
            img = tf.image.random_crop(img, [self.img_size, self.img_size, 3])
            # flip
            img = tf.image.random_flip_left_right(img)
        else:
            # centre-crop instead of random for validation dataset
            img = tf.image.resize_with_pad(img, self.img_size + 32, self.img_size + 32)
            img = tf.image.resize_with_crop_or_pad(img, self.img_size, self.img_size) 
        
        # random augmentation
        if self.ra_layer is not None:
            img = self.ra_layer(img)

        # map to [0,1]
        img = tf.image.convert_image_dtype(img, tf.float32)
        if self.cutout_layer is not None:
            img = self.cutout_layer(img)

        # normalization
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img, label

    def build_dataset(self, root_dir: str):
        # 
        if self._class_table is None:      
            self._class_table = self._make_class_table(root_dir)

        # converts back-slashes (Windows) to '/'
        root_dir = pathlib.Path(root_dir).as_posix()    
        files  = tf.data.Dataset.list_files(root_dir + '/*/images/*', # official ImageNet folder may have a different layout
                                            shuffle=self.training)
        def _parent_folder(path):
            # Replace back-slashes with forward slashes if it's windows
            path = tf.strings.regex_replace(path, r'\\', '/') 
            return tf.strings.split(path, '/')[-2]
        labels = files.map(
            lambda p: self._class_table.lookup(_parent_folder(p)),
            num_parallel_calls=tf.data.AUTOTUNE)

        ds = tf.data.Dataset.zip((files, labels))
        if self.training:
            ds = ds.shuffle(10_000)
        

        ds = ds.map(self._read_and_augment, tf.data.AUTOTUNE)

        ds = ds.batch(self.batch_size, drop_remainder=self.training)
        return ds.prefetch(tf.data.AUTOTUNE)
    
train_ds = ImageNetPreprocessor(training=True, randaugment=True, cutout=True)\
           .build_dataset("sample_data/tiny-imagenet-200/tiny-imagenet-200/train")

imgs, lbls = next(iter(train_ds))
print(imgs.shape, imgs.dtype, lbls.shape, lbls.dtype)