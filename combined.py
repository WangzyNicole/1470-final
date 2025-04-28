import tensorflow as tf
import pathlib
import numpy as np
import random
import os
from keras_cv.layers import RandAugment, RandomCutout

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Disable layout optimizations that might be causing the issue
tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False,
    'constant_folding': True,
    'shape_optimization': True
})

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps, alpha=0.0):
        super(WarmupCosineDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)
        
        warmup_lr = self.initial_learning_rate * (step / warmup_steps)
        
        cosine_decay = 0.5 * (1.0 + tf.cos(
            tf.constant(np.pi) * (step - warmup_steps) / (decay_steps - warmup_steps)))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        cosine_lr = self.initial_learning_rate * decayed
        
        lr = tf.where(step < warmup_steps, warmup_lr, cosine_lr)
        return lr
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha
        }

class MLP(tf.keras.layers.Layer):
    def __init__(self, embed_dim, n_classes=None, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name="dropout")
        self.linear_1 = tf.keras.layers.Dense(
            self.embed_dim,
            activation='gelu',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="dense_1"
        )
        if self.n_classes is None:
            self.linear_2 = tf.keras.layers.Dense(
                self.embed_dim,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name="dense_2"
            )
        else:
            self.linear_2 = tf.keras.layers.Dense(
                self.n_classes,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name="dense_out"
            )

    def call(self, x, training=None):
        x = self.linear_1(x)
        x = self.dropout(x, training=training)
        x = self.linear_2(x)
        return x

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_H, patch_W, embed_dim, dropout_rate=0.1):
        super().__init__()
        self.patch_H = patch_H
        self.patch_W = patch_W
        self.proj = MLP(embed_dim=embed_dim, dropout_rate=dropout_rate)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        # Extract patches
        x = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_H, self.patch_W, 1],
            strides=[1, self.patch_H, self.patch_W, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        
        # Pass through MLP
        x = self.proj(x, training=training)
        x = self.dropout(x, training=training)
        return x

class PermuteMLP(tf.keras.layers.Layer):
    def __init__(self, embed_dim, embed_H, embed_W, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_H = embed_H
        self.embed_W = embed_W
        self.proj_H = MLP(embed_dim=embed_dim, dropout_rate=dropout_rate)
        self.proj_W = MLP(embed_dim=embed_dim, dropout_rate=dropout_rate)
        self.proj_C = MLP(embed_dim=embed_dim, dropout_rate=dropout_rate)
        self.proj = MLP(embed_dim=embed_dim, dropout_rate=dropout_rate)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def _h_c_permute(self, x, training=None):
        B = tf.shape(x)[0]
        H = self.embed_H
        W = self.embed_W
        C = self.embed_dim
        
        S = C // H
        
        # Reshape and transpose operations
        x = tf.reshape(x, [B, H, W, H, S])
        x = tf.transpose(x, [0, 3, 2, 1, 4])
        x = tf.reshape(x, [B, H, W, C])
        
        # Process through MLP
        x = self.proj_H(x, training=training)
        
        # Reshape and transpose back
        x = tf.reshape(x, [B, H, W, H, S])
        x = tf.transpose(x, [0, 3, 2, 1, 4])
        x = tf.reshape(x, [B, H, W, C])
        return x

    def _w_c_permute(self, x, training=None):
        B = tf.shape(x)[0]
        H = self.embed_H
        W = self.embed_W
        C = self.embed_dim
        
        S = C // W
        
        # Reshape and transpose operations
        x = tf.reshape(x, [B, H, W, W, S])
        x = tf.transpose(x, [0, 1, 3, 2, 4])
        x = tf.reshape(x, [B, H, W, C])
        
        # Process through MLP
        x = self.proj_W(x, training=training)
        
        # Reshape and transpose back
        x = tf.reshape(x, [B, H, W, W, S])
        x = tf.transpose(x, [0, 1, 3, 2, 4])
        x = tf.reshape(x, [B, H, W, C])
        return x

    def call(self, x, training=None):
        x_ = x
        # Apply H-C permutation
        x_H = self._h_c_permute(x, training=training)
        
        # Apply W-C permutation
        x_W = self._w_c_permute(x, training=training)
        
        # Apply channel-wise MLP
        x_C = self.proj_C(x, training=training)
        
        # Combine and project
        x = self.proj(x_H + x_W + x_C, training=training)
        x = self.dropout(x, training=training)
        
        # Skip connection
        return x_ + x

class Permutator(tf.keras.layers.Layer):
    def __init__(self, embed_dim, embed_H, embed_W, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_H = embed_H
        self.embed_W = embed_W
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.permute_mlp = PermuteMLP(embed_dim=embed_dim, embed_H=embed_H, embed_W=embed_W, dropout_rate=dropout_rate)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.channel_mlp = MLP(embed_dim=embed_dim, dropout_rate=dropout_rate)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        # First block
        x_ = x
        x = self.layer_norm_1(x)
        x = self.permute_mlp(x, training=training)
        x = self.dropout(x, training=training)
        x += x_
        
        # Second block
        x_ = x
        x = self.layer_norm_2(x)
        x = self.channel_mlp(x, training=training)
        x = self.dropout(x, training=training)
        x += x_
        return x

class VisionPermutator(tf.keras.layers.Layer):
    def __init__(self, embed_dim, n_classes, patch_H, patch_W, n_perm, img_H, img_W, dropout_rate=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.patch_H = patch_H
        self.patch_W = patch_W
        self.n_perm = n_perm
        
        self.embed_H = img_H // patch_H
        self.embed_W = img_W // patch_W
        
        assert embed_dim % self.embed_H == 0, f"embed_dim ({embed_dim}) must be divisible by embed_H ({self.embed_H})"
        assert embed_dim % self.embed_W == 0, f"embed_dim ({embed_dim}) must be divisible by embed_W ({self.embed_W})"

        self.patch_embedding = PatchEmbedding(
            patch_H=self.patch_H,
            patch_W=self.patch_W,
            embed_dim=self.embed_dim,
            dropout_rate=dropout_rate
        )
        
        # Create permutator layers
        self.permutators = []
        for i in range(self.n_perm):
            self.permutators.append(
                Permutator(
                    embed_dim=self.embed_dim, 
                    embed_H=self.embed_H, 
                    embed_W=self.embed_W, 
                    dropout_rate=dropout_rate
                )
            )
            
        self.output_mlp = MLP(
            embed_dim=self.embed_dim,
            n_classes=self.n_classes,
            dropout_rate=dropout_rate*2
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        # Validate input shape
        tf.debugging.assert_rank(x, 4, message="Expected rank 4 input tensor")
        
        shape = tf.shape(x)
        H, W = shape[1], shape[2]
        
        tf.debugging.assert_equal(
            tf.math.floormod(H, self.patch_H), 
            0, 
            message=f"patch_H ({self.patch_H}) is not a divisor of H ({H})!"
        )
        
        tf.debugging.assert_equal(
            tf.math.floormod(W, self.patch_W), 
            0, 
            message=f"patch_W ({self.patch_W}) is not a divisor of W ({W})!"
        )
        
        # Apply patch embedding
        x = self.patch_embedding(x, training=training)
        
        # Apply permutator blocks
        for permutator in self.permutators:
            x = permutator(x, training=training)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=[1, 2])
        
        # Final dropout and MLP
        x = self.dropout(x, training=training)
        x = self.output_mlp(x, training=training)
        return x

    def predict(self, x):
        logits = self(x, training=False)
        return tf.argmax(logits, axis=-1)

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

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

class ImageNetPreprocessor:
    def __init__(self,
                img_size=224, 
                batch_size=256, 
                training=True,
                randaugment=True,
                cutout=True,
                num_classes=200
                ):
        self.img_size = img_size
        self.batch_size = batch_size
        self.training = training
        self.randaugment = randaugment
        self.cutout = cutout
        self.num_classes = num_classes

        # Create RandAugment with more controlled settings
        self.ra_layer = None
        if training and randaugment:
            try:
                self.ra_layer = RandAugment(
                    value_range=(0, 255),
                    augmentations_per_image=2,
                    magnitude=0.3,  # Reduced magnitude
                    seed=42
                )
            except:
                print("Warning: RandAugment not available or failed to initialize. Using basic augmentations.")
                self.ra_layer = None

        self.cutout_layer = (
            RandomCutout(
                height_factor=0.4, 
                width_factor=0.4,
                fill_value=0, 
                seed=42)
            if training and cutout
            else None
        )
        self._class_table = None
        self._filename_to_class = None

    def _make_class_table(self, root_dir):
        class_names = sorted([p.name for p in pathlib.Path(root_dir).iterdir()
                              if p.is_dir()])
        keys = tf.constant(class_names)
        vals = tf.range(len(class_names), dtype=tf.int32)
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=-1)
    
    def _load_val_annotations(self, val_annotations_path):
        filename_to_class_id = {}
        with open(val_annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    filename, class_id = parts[0], parts[1]
                    filename_to_class_id[filename] = class_id
        
        filenames = list(filename_to_class_id.keys())
        class_ids = list(filename_to_class_id.values())
        
        keys = tf.constant(filenames)
        vals = tf.constant(class_ids)
        
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, vals), 
            default_value='unknown')
    
    def _read_and_augment(self, path, label):
        # Read image
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        
        if self.training:
            # Resize and crop
            img = tf.image.resize_with_pad(img, self.img_size + 32, self.img_size + 32)
            img = tf.image.random_crop(img, [self.img_size, self.img_size, 3])
            
            # Basic augmentations
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
            
            # Apply RandAugment safely
            if self.ra_layer is not None:
                try:
                    # Ensure image is in the correct range for RandAugment
                    img_for_ra = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
                    img = self.ra_layer(img_for_ra)
                except:
                    print("Warning: RandAugment failed during execution. Skipping.")
        else:
            # Center crop for validation
            img = tf.image.resize_with_pad(img, self.img_size + 32, self.img_size + 32)
            img = tf.image.resize_with_crop_or_pad(img, self.img_size, self.img_size) 
        
        # Convert to float32
        img = tf.image.convert_image_dtype(img, tf.float32)
        
        # Apply cutout
        if self.cutout_layer is not None:
            try:
                img = self.cutout_layer(img)
            except:
                print("Warning: Cutout failed during execution. Skipping.")
        
        # Normalize with ImageNet stats
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img, label

    def _get_train_class(self, path):
        path = tf.strings.regex_replace(path, r'\\', '/') 
        parts = tf.strings.split(path, '/')
        class_name = parts[-3]
        return self._class_table.lookup(class_name)
    
    def _get_val_class(self, path):
        path = tf.strings.regex_replace(path, r'\\', '/') 
        parts = tf.strings.split(path, '/')
        filename = parts[-1]
        class_id = self._filename_to_class.lookup(filename)
        return self._class_table.lookup(class_id)

    def build_dataset(self, root_dir, is_val=False):
        if self._class_table is None:
            class_dir = root_dir if not is_val else root_dir.replace('/val', '/train')
            self._class_table = self._make_class_table(class_dir)
        
        root_dir = pathlib.Path(root_dir).as_posix()
        
        if not is_val:
            files = tf.data.Dataset.list_files(root_dir + '/*/images/*', shuffle=self.training)
            labels = files.map(
                self._get_train_class,
                num_parallel_calls=tf.data.AUTOTUNE)
        else:
            files = tf.data.Dataset.list_files(root_dir + '/images/*', shuffle=False)
            
            val_annotations_path = os.path.join(root_dir, 'val_annotations.txt')
            if self._filename_to_class is None:
                self._filename_to_class = self._load_val_annotations(val_annotations_path)
            
            labels = files.map(
                self._get_val_class,
                num_parallel_calls=tf.data.AUTOTUNE)
        
        ds = tf.data.Dataset.zip((files, labels))
        if self.training:
            ds = ds.shuffle(10_000)
        
        # Use a try-except block around the augmentation to handle potential errors
        def safe_augment(path, label):
            try:
                return self._read_and_augment(path, label)
            except tf.errors.InvalidArgumentError as e:
                print(f"Error in augmentation: {e}")
                # Fallback to simple processing
                img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
                img = tf.image.resize_with_pad(img, self.img_size, self.img_size)
                img = tf.image.convert_image_dtype(img, tf.float32)
                img = (img - IMAGENET_MEAN) / IMAGENET_STD
                return img, label
        
        ds = ds.map(safe_augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size, drop_remainder=self.training)
        return ds.prefetch(tf.data.AUTOTUNE)

def augment_mnist(image, label):
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label

def augment_cifar(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

def build_and_train_model(dataset_name='mnist', img_size=28, batch_size=64, 
                         epochs=10, n_classes=10, patch_size=4, embed_dim=224, n_perm=6, dropout_rate=0.1):
    if dataset_name in ['mnist', 'cifar10']:
        preprocessor = DatasetPreprocessor(dataset_name)
        train_inputs = preprocessor.train_inputs
        train_labels = preprocessor.train_labels
        test_inputs = preprocessor.test_inputs
        test_labels = preprocessor.test_labels
        
        train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
        if dataset_name == 'mnist':
            train_ds = train_ds.map(augment_mnist, num_parallel_calls=tf.data.AUTOTUNE)
        elif dataset_name == 'cifar10':
            train_ds = train_ds.map(augment_cifar, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.shuffle(10000).batch(batch_size)
        
        test_ds = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))
        test_ds = test_ds.batch(batch_size)
        
        if dataset_name == 'mnist':
            channels = 1
        else:
            channels = 3
        
        inputs = tf.keras.layers.Input((img_size, img_size, channels))
        train_size = len(train_inputs)
            
    elif dataset_name == 'tiny-imagenet':
        train_ds = ImageNetPreprocessor(
            img_size=img_size, 
            batch_size=batch_size, 
            training=True, 
            randaugment=True, 
            cutout=True,
            num_classes=n_classes
        ).build_dataset("tiny-imagenet-200/train", is_val=False)
        
        test_ds = ImageNetPreprocessor(
            img_size=img_size, 
            batch_size=batch_size, 
            training=False, 
            randaugment=False, 
            cutout=False,
            num_classes=n_classes
        ).build_dataset("tiny-imagenet-200/val", is_val=True)
        
        channels = 3
        inputs = tf.keras.layers.Input((img_size, img_size, channels))
        train_size = 100000
    
    # Setup learning rate schedule
    initial_learning_rate = 1e-3
    steps_per_epoch = train_size // batch_size
    warmup_epochs = 5
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    
    lr_schedule = WarmupCosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=total_steps,
        warmup_steps=warmup_steps,
        alpha=1e-5
    )
    
    # Create model with explicit batch input shape
    vip = VisionPermutator(
        embed_dim=embed_dim,
        n_classes=n_classes,
        patch_H=patch_size,
        patch_W=patch_size,
        n_perm=n_perm,
        img_H=img_size,
        img_W=img_size,
        dropout_rate=dropout_rate
    )
    
    # Use functional API to create model
    outputs = vip(inputs, training=True)
    model = tf.keras.Model(inputs, outputs)
    
    # Setup model with mixed precision for better performance
    # Explicitly set the dtype policy
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, 
            clipnorm=1.0,
            epsilon=1e-7,  # Increased epsilon for better numerical stability
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Setup callbacks
    checkpoint_filepath = f'best_model_{dataset_name}.h5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,  # Changed to save weights only to avoid serialization issues
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,
        patience=5, 
        min_lr=1e-6,
        verbose=1
    )
    
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=f'logs_{dataset_name}',
        write_graph=False  # Don't write the graph to avoid OOM issues
    )
    
    callbacks = [
        model_checkpoint_callback,
        reduce_lr,
        early_stopping,
        tensorboard
    ]
    
    # Train the model
    try:
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=test_ds,
            callbacks=callbacks,
            verbose=1
        )
        return model, history
    except Exception as e:
        print(f"Error during training: {e}")
        # Try a fallback approach
        print("Attempting fallback to eager execution mode...")
        tf.config.run_functions_eagerly(True)
        
        # Try with reduced complexity
        vip = VisionPermutator(
            embed_dim=embed_dim,
            n_classes=n_classes,
            patch_H=patch_size,
            patch_W=patch_size,
            n_perm=max(1, n_perm // 2),  # Reduced number of permutator layers
            img_H=img_size,
            img_W=img_size,
            dropout_rate=dropout_rate / 2  # Reduced dropout
        )
        
        outputs = vip(inputs, training=True)
        model = tf.keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # Simplified optimizer
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        history = model.fit(
            train_ds,
            epochs=min(10, epochs),  # Reduced epochs
            validation_data=test_ds,
            callbacks=[early_stopping, model_checkpoint_callback],
            verbose=1
        )
        return model, history

if __name__ == "__main__":
    # Train on MNIST
    embed_dim_mnist = 7 * 32
    model_mnist, history_mnist = build_and_train_model(
        dataset_name='mnist',
        img_size=28,
        batch_size=128,
        epochs=15,
        n_classes=10,
        patch_size=4,
        embed_dim=embed_dim_mnist,
        n_perm=4,
        dropout_rate=0.1
    )
    
    # Train on CIFAR-10
    embed_dim_cifar = 8 * 24
    model_cifar, history_cifar = build_and_train_model(
        dataset_name='cifar10',
        img_size=32,
        batch_size=64,
        epochs=30,
        n_classes=10,
        patch_size=4,
        embed_dim=embed_dim_cifar,
        n_perm=8,
        dropout_rate=0.2
    )
    
    # Train on Tiny-ImageNet
    embed_dim_tinyimagenet = 8 * 48
    model_tinyimagenet, history_tinyimagenet = build_and_train_model(
        dataset_name='tiny-imagenet',
        img_size=64,
        batch_size=32,
        epochs=50,
        n_classes=200,
        patch_size=8,
        embed_dim=embed_dim_tinyimagenet,
        n_perm=12,
        dropout_rate=0.25
    )