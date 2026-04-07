"""
Indian Sign Language Model Training Script (Landmark-based)
===========================================================
Features:
- MediaPipe landmark extraction from images
- Support for single hand and two hands
- Keras neural network trained on landmark features
- Compatible with model.p preprocessing logic
- CUDA GPU acceleration
- Best model checkpointing
- Early stopping

Usage:
    python train_indian_model.py
"""

import osc
import sys
import platform
import copy
import itertools
import pickle
import json
from multiprocessing import freeze_support
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================================================
# CONFIGURATION
# =========================================================
# Try multiple possible dataset paths
POSSIBLE_DATASET_PATHS = [
    './Indian',
    r'D:\sign-to-text-and-speech\New folder\model\Indian',
    r'D:\sign-to-text-and-speech\Indian'
]

# Find the first existing dataset path
DATASET_PATH = None
for path in POSSIBLE_DATASET_PATHS:
    if os.path.exists(path):
        DATASET_PATH = path
        break

if DATASET_PATH is None:
    DATASET_PATH = './Indian'  # Default fallback
MODEL_SAVE_PATH = './indian_sign_model.h5'
SKELETON_MODEL_SAVE_PATH = './model.p'  # Also save as pickle like original model.p
CHECKPOINT_DIR = './checkpoints'

# Training settings
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# MediaPipe settings
MAX_NUM_HANDS = 2  # Support two hands
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Feature settings
SUPPORT_TWO_HANDS = True  # If True, use 84 features (42 per hand), else 42
USE_PREPROCESSED_LANDMARKS = True  # Use pre_process_landmark (like Keras model)

# Checkpoint / averaging settings
SAVE_PERIODIC_CHECKPOINT_EVERY_N_EPOCHS = 5
WEIGHT_AVG_USE = True
WEIGHT_AVG_NUM_CHECKPOINTS = 5  # last N periodic checkpoints from the latest run


def configure_gpu():
    """Configure GPU, XLA, and mixed precision with enhanced diagnostics"""
    import tensorflow as tf
    # Import mixed_precision - try tensorflow.keras first, fallback to keras
    try:
        from tensorflow.keras import mixed_precision
    except (ImportError, AttributeError):
        try:
            from keras import mixed_precision
        except ImportError:
            mixed_precision = None
            print("[WARNING] Could not import mixed_precision")
    
    print("=" * 60)
    print("CONFIGURING GPU & XLA...")
    print("=" * 60)
    
    # Check TensorFlow version and build info
    print(f"[INFO] TensorFlow version: {tf.__version__}")
    print(f"[INFO] TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"[INFO] TensorFlow built with GPU: {tf.test.is_built_with_gpu_support()}")
    
    # Check for CUDA via cuda-python (if available)
    try:
        import cuda
        # cuda-python may not have __version__, check if it's available
        cuda_version = getattr(cuda, '__version__', 'available (version unknown)')
        print(f"[INFO] CUDA Python available: {cuda_version}")
    except ImportError:
        print("[INFO] CUDA Python not available (optional)")
    except Exception as e:
        print(f"[INFO] CUDA Python check failed: {e}")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    
    if gpus:
        print(f"\n[OK] Found {len(gpus)} physical GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"     GPU {i}: {gpu.name}")
            try:
                # Get GPU details
                gpu_details = tf.config.experimental.get_device_details(gpu)
                if gpu_details:
                    print(f"            Details: {gpu_details}")
            except:
                pass
        
        if logical_gpus:
            print(f"[OK] Found {len(logical_gpus)} logical GPU(s)")
        
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"[OK] Memory growth enabled for {gpu.name}")
            except RuntimeError as e:
                print(f"[WARNING] Could not set memory growth: {e}")
        
        # Enable XLA JIT compilation for faster training
        try:
            tf.config.optimizer.set_jit(True)
            print("[OK] XLA JIT compilation enabled")
        except Exception as e:
            print(f"[WARNING] Could not enable XLA: {e}")
        
        # Enable mixed precision (FP16) for faster GPU training
        try:
            mixed_precision.set_global_policy('mixed_float16')
            print("[OK] Mixed precision (FP16) enabled - 2x speed boost")
        except Exception as e:
            print(f"[WARNING] Could not enable mixed precision: {e}")
        
        # Test GPU computation
        try:
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([1.0, 2.0, 3.0])
                result = tf.reduce_sum(test_tensor)
            print(f"[OK] GPU computation test passed: {result.numpy()}")
        except Exception as e:
            print(f"[WARNING] GPU computation test failed: {e}")
    
    else:
        print("\n[WARNING] No GPU detected by TensorFlow.")
        print("\n[DIAGNOSTICS]")
        
        # Check if TensorFlow has GPU support
        if not tf.test.is_built_with_cuda():
            print("  [X] TensorFlow was NOT built with CUDA support")
            print("     This is likely a CPU-only TensorFlow installation.")
            print("\n[SOLUTION]")
            print("  1. Uninstall current TensorFlow:")
            print("     pip uninstall tensorflow tensorflow-cpu")
            print("  2. Install TensorFlow with GPU support:")
            print("     pip install tensorflow[and-cuda]")
            print("     OR")
            print("     pip install tensorflow-gpu")
        else:
            print("  [OK] TensorFlow was built with CUDA support")
            print("  [X] But no GPU devices are visible")
            print("\n[POSSIBLE CAUSES]")
            print("  1. CUDA drivers not installed")
            print("  2. CUDA/cuDNN version mismatch with TensorFlow")
            print("  3. GPU not properly configured")
            print("\n[SOLUTION]")
            print("  1. Check NVIDIA drivers: nvidia-smi")
            print("  2. Verify CUDA version matches TensorFlow requirements")
            print("  3. Install matching cuDNN version")
            print("  4. Check CUDA_PATH environment variable")
        
        print("\n[INFO] Training will continue on CPU (slower but functional)")
        print("       For GPU acceleration, follow the solutions above.")
    
    return gpus


def calc_landmark_list(image, landmarks):
    """Convert MediaPipe landmarks to pixel coordinates list."""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    
    for lm in landmarks.landmark:
        landmark_x = min(int(lm.x * image_width), image_width - 1)
        landmark_y = min(int(lm.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point


def pre_process_landmark(landmark_list):
    """Make landmarks relative to first point and normalize (reference logic)."""
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    
    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list))) or 1.0
    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    
    return temp_landmark_list


def extract_landmarks_skeleton_style(image, hand_landmarks):
    """
    Extract landmarks in the same format as model.p (skeleton model).
    Returns 42 features: normalized (x, y) coordinates relative to hand bounding box.
    """
    x_, y_ = [], []
    
    # Extract x and y coordinates
    for lm in hand_landmarks.landmark:
        x_.append(lm.x)
        y_.append(lm.y)
    
    # Normalize relative to bounding box (like model.p)
    data_aux = []
    if len(x_) > 0 and len(y_) > 0:
        min_x = min(x_)
        min_y = min(y_)
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min_x)
            data_aux.append(lm.y - min_y)
    
    return data_aux


def extract_features_from_image(image_path, hands_detector):
    """
    Extract landmark features from an image.
    Returns (preprocessed_features, skeleton_features, num_hands_detected).
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, None, 0
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None, None, 0
    
    num_hands = len(results.multi_hand_landmarks)

    # Always compute BOTH representations so we can train:
    # - Keras landmark model (preprocessed)
    # - model.p RandomForest (skeleton style) compatible with app.py
    pre_list = []
    skel_list = []
    for hand_landmarks in results.multi_hand_landmarks:
        landmark_list = calc_landmark_list(image, hand_landmarks)
        pre_list.append(pre_process_landmark(landmark_list))
        skel_list.append(extract_landmarks_skeleton_style(image, hand_landmarks))

    def combine_two_hands(per_hand_list):
        if not per_hand_list:
            return None
        if SUPPORT_TWO_HANDS:
            if len(per_hand_list) == 1:
                return per_hand_list[0] + [0.0] * len(per_hand_list[0])
            return per_hand_list[0] + per_hand_list[1]
        return per_hand_list[0]

    pre_features = combine_two_hands(pre_list)
    skel_features = combine_two_hands(skel_list)
    if pre_features is None or skel_features is None:
        return None, None, 0

    return pre_features, skel_features, num_hands


def load_dataset(dataset_path):
    """
    Load dataset and extract landmarks from all images.
    Returns X (features), y (labels), and class names.
    """
    print("\n" + "=" * 60)
    print("LOADING DATASET AND EXTRACTING LANDMARKS...")
    print("=" * 60)
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    
    # Collect all image paths and labels
    image_paths = []
    labels = []
    class_names = []
    
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset path not found: {dataset_path}")
        sys.exit(1)
    
    # Get all class folders
    for class_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            print(f"  Processing class: {class_name}")
            
            # Get all images in this class
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_path, filename))
                    labels.append(class_name)
    
    class_names = sorted(set(class_names))
    print(f"\n[OK] Found {len(class_names)} classes: {class_names}")
    print(f"[OK] Total images: {len(image_paths)}")
    
    # Extract features from images
    print("\n[INFO] Extracting landmarks from images...")
    print("       This may take a while...")
    
    X_pre = []
    X_skel = []
    y = []
    hand_counts = defaultdict(int)
    failed_images = 0
    
    for idx, image_path in enumerate(image_paths):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(image_paths)} images...")
        
        pre_features, skel_features, num_hands = extract_features_from_image(image_path, hands)

        if pre_features is not None and skel_features is not None:
            X_pre.append(pre_features)
            X_skel.append(skel_features)
            y.append(labels[idx])
            hand_counts[num_hands] += 1
        else:
            failed_images += 1
    
    hands.close()
    
    print(f"\n[OK] Successfully extracted features from {len(X_pre)} images")
    print(f"[OK] Failed to extract from {failed_images} images")
    print(f"[OK] Hand distribution: {dict(hand_counts)}")
    
    if len(X_pre) == 0:
        print("❌ ERROR: No features extracted. Check your dataset and MediaPipe setup.")
        sys.exit(1)
    
    # Convert to numpy arrays
    X_pre = np.asarray(X_pre, dtype=np.float32)
    X_skel = np.asarray(X_skel, dtype=np.float32)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    print(f"\n[OK] Preprocessed feature shape: {X_pre.shape}")
    print(f"[OK] Skeleton feature shape: {X_skel.shape}")
    print(f"[OK] Number of classes: {num_classes}")
    print(f"[OK] Classes: {list(label_encoder.classes_)}")
    
    return X_pre, X_skel, y_encoded, label_encoder.classes_, label_encoder


def build_landmark_model(input_dim, num_classes):
    """
    Build a neural network model for landmark-based classification.
    Input: landmark features (84 for two hands)
    Output: num_classes predictions
    """
    import tensorflow as tf
    import tensorflow.keras as keras
    from tensorflow.keras import layers
    
    print("\n" + "=" * 60)
    print("BUILDING MODEL...")
    print("=" * 60)
    
    inputs = layers.Input(shape=(input_dim,))
    
    # Dense layers for landmark features
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer (float32 for mixed precision compatibility)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = keras.Model(inputs, outputs, name='IndianSignLanguageLandmarkModel')
    return model


def find_best_checkpoint(checkpoint_dir):
    """Find the best checkpoint based on validation accuracy in filename"""
    import re
    import glob
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "best_model_*.h5"))
    
    if not checkpoint_files:
        return None
    
    best_checkpoint = None
    best_acc = -1.0
    
    # Try to extract accuracy from filename or use modification time
    for checkpoint_file in checkpoint_files:
        # Check if there's a corresponding history file
        base_name = os.path.basename(checkpoint_file)
        timestamp_match = re.search(r'(\d{8}_\d{6})', base_name)
        
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            # Try to find history or use file modification time
            # For now, use the most recent checkpoint
            file_time = os.path.getmtime(checkpoint_file)
            if file_time > best_acc:
                best_acc = file_time
                best_checkpoint = checkpoint_file
    
    return best_checkpoint


def _parse_epoch_from_checkpoint_name(path):
    import re
    m = re.search(r'checkpoint_epoch_(\d+)_', os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def find_latest_periodic_checkpoint(checkpoint_dir):
    """Return (path, epoch_number) for latest periodic checkpoint, else (None, None)."""
    import glob
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*_*.h5"))
    if not checkpoint_files:
        return None, None

    best_path = None
    best_epoch = -1
    best_mtime = -1
    for p in checkpoint_files:
        epoch = _parse_epoch_from_checkpoint_name(p)
        mtime = os.path.getmtime(p)
        # Prefer higher epoch; break ties with mtime
        if epoch is not None and (epoch > best_epoch or (epoch == best_epoch and mtime > best_mtime)):
            best_path = p
            best_epoch = epoch
            best_mtime = mtime

    return best_path, (best_epoch if best_epoch >= 0 else None)


def load_checkpoint_if_exists(checkpoint_dir):
    """
    Load latest periodic checkpoint (for resuming) if available.
    Returns (loaded_model_or_None, initial_epoch_int).
    """
    import tensorflow.keras as keras

    ckpt_path, ckpt_epoch = find_latest_periodic_checkpoint(checkpoint_dir)
    if ckpt_path and ckpt_epoch is not None and os.path.exists(ckpt_path):
        print(f"\n[INFO] Found periodic checkpoint to resume: {os.path.basename(ckpt_path)}")
        try:
            loaded_model = keras.models.load_model(ckpt_path, compile=False)
            # initial_epoch is the number of epochs already completed
            initial_epoch = ckpt_epoch
            print(f"[OK] Loaded checkpoint model (resume from epoch {initial_epoch})")
            return loaded_model, initial_epoch
        except Exception as e:
            print(f"[WARNING] Could not load periodic checkpoint: {e}")

    return None, 0


def get_callbacks(timestamp, save_frequency=5):
    """Create training callbacks with multiple checkpoint saves"""
    from tensorflow.keras import callbacks

    class SaveEveryNEpochs(callbacks.Callback):
        def __init__(self, every_n, out_dir, run_ts):
            super().__init__()
            self.every_n = int(every_n)
            self.out_dir = out_dir
            self.run_ts = run_ts

        def on_epoch_end(self, epoch, logs=None):
            epoch_1based = epoch + 1
            if self.every_n <= 0:
                return
            if epoch_1based % self.every_n != 0:
                return
            path = os.path.join(self.out_dir, f"checkpoint_epoch_{epoch_1based:03d}_{self.run_ts}.h5")
            self.model.save(path)
    
    callback_list = [
        # Save best model based on validation accuracy
        callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, f"best_model_{timestamp}.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1,
            save_freq='epoch'
        ),
        
        # Save best weights separately
        callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, f"best_weights_{timestamp}.weights.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=0
        ),
        
        # Save periodic full-model checkpoints every N epochs (for weight averaging / recovery)
        SaveEveryNEpochs(save_frequency, CHECKPOINT_DIR, timestamp),
        
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir=os.path.join(CHECKPOINT_DIR, f"logs_{timestamp}"),
            histogram_freq=0,
            write_graph=False,
            update_freq='epoch'
        ),
    ]
    
    return callback_list


def list_latest_run_periodic_checkpoints(checkpoint_dir):
    """
    Return periodic checkpoints from the latest run timestamp, sorted by epoch.
    """
    import glob
    import re

    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*_*.h5"))
    if not files:
        return []

    def parse(path):
        name = os.path.basename(path)
        m = re.search(r'checkpoint_epoch_(\d+)_([0-9]{8}_[0-9]{6})\.h5$', name)
        if not m:
            return None
        return int(m.group(1)), m.group(2)

    parsed = []
    for f in files:
        p = parse(f)
        if p is not None:
            parsed.append((f, p[0], p[1]))
    if not parsed:
        return []

    latest_ts = max(ts for _, _, ts in parsed)
    run_files = [(f, epoch) for (f, epoch, ts) in parsed if ts == latest_ts]
    run_files.sort(key=lambda x: x[1])
    return [f for f, _ in run_files]


def try_weight_average_from_checkpoints(checkpoint_dir, base_model, X_val, y_val_cat, num_to_average):
    """
    Weight-average the last N periodic checkpoints (same run) and return a model if it improves val acc.
    """
    import tensorflow.keras as keras

    ckpts = list_latest_run_periodic_checkpoints(checkpoint_dir)
    if len(ckpts) < 2:
        return None

    ckpts = ckpts[-min(len(ckpts), int(num_to_average)):]
    print(f"[INFO] Weight-averaging {len(ckpts)} checkpoints:")
    for c in ckpts:
        print(f"  - {os.path.basename(c)}")

    models = []
    for c in ckpts:
        try:
            m = keras.models.load_model(c, compile=False)
            models.append(m)
        except Exception as e:
            print(f"[WARNING] Could not load {os.path.basename(c)} for averaging: {e}")

    if len(models) < 2:
        return None

    # Average weights (simple mean)
    avg_weights = []
    weights_per_model = [m.get_weights() for m in models]
    for weight_tensors in zip(*weights_per_model):
        avg_weights.append(np.mean(np.stack(weight_tensors, axis=0), axis=0))

    averaged_model = keras.models.clone_model(base_model)
    averaged_model.set_weights(avg_weights)

    # Evaluate both on validation
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    averaged_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    base_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    avg_acc = averaged_model.evaluate(X_val, y_val_cat, verbose=0)[1]
    base_acc = base_model.evaluate(X_val, y_val_cat, verbose=0)[1]
    print(f"[OK] Weight-avg val_acc: {avg_acc:.4f} vs base val_acc: {base_acc:.4f}")

    if avg_acc > base_acc:
        print("[OK] Weight-averaged model is better; using it.")
        return averaged_model
    print("[INFO] Weight-averaged model not better; keeping base.")
    return None


def save_skeleton_model(X_train_skel, y_train, label_encoder, save_path):
    """
    Train & save a scikit-learn model in the same format as the existing `model.p` loader expects.
    This produces a real RandomForest with `predict_proba`, compatible with `app.py`.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    print("\n" + "=" * 60)
    print("TRAINING SKELETON MODEL (model.p format)...")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight=None
    )
    rf.fit(X_train_skel, y_train)

    model_dict = {
        'model': rf,
        'label_encoder': label_encoder,
        'class_names': label_encoder.classes_.tolist(),
        'model_type': 'sklearn_random_forest',
        'input_dim': int(X_train_skel.shape[1]),
        'num_classes': int(len(label_encoder.classes_)),
        'support_two_hands': bool(SUPPORT_TWO_HANDS)
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_dict, f)
    
    print(f"[OK] Model saved to {save_path}")


def main():
    """Main training function"""
    import tensorflow as tf
    # Import keras - use tensorflow.keras for compatibility
    # In TF 2.15+, keras is available via tensorflow.keras
    import tensorflow.keras as keras
    
    # Configure GPU
    gpus = configure_gpu()
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"\n[CONFIG]")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Support two hands: {SUPPORT_TWO_HANDS}")
    print(f"  Use preprocessed landmarks: {USE_PREPROCESSED_LANDMARKS}")
    print(f"  Platform: {platform.system()}")
    
    # Load dataset and extract landmarks (both representations)
    X_pre, X_skel, y, class_names, label_encoder = load_dataset(DATASET_PATH)

    X_for_keras = X_pre if USE_PREPROCESSED_LANDMARKS else X_skel
    
    # Split dataset
    print("\n" + "=" * 60)
    print("SPLITTING DATASET...")
    print("=" * 60)
    
    # Split using indices to keep X_for_keras and X_skel aligned
    indices = np.arange(len(y))
    idx_temp, idx_test, y_temp, y_test = train_test_split(
        indices, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_temp, y_temp, test_size=VALIDATION_SPLIT/(1-TEST_SPLIT),
        random_state=42, stratify=y_temp
    )

    X_train = X_for_keras[idx_train]
    X_val = X_for_keras[idx_val]
    X_test = X_for_keras[idx_test]

    X_train_skel = X_skel[idx_train]
    X_val_skel = X_skel[idx_val]
    X_test_skel = X_skel[idx_test]
    
    print(f"[OK] Training samples: {len(X_train)}")
    print(f"[OK] Validation samples: {len(X_val)}")
    print(f"[OK] Test samples: {len(X_test)}")
    
    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=len(class_names))
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=len(class_names))
    y_test_cat = keras.utils.to_categorical(y_test, num_classes=len(class_names))
    
    # Build model
    input_dim = int(X_for_keras.shape[1])
    num_classes = len(class_names)
    
    model = build_landmark_model(input_dim, num_classes)
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    
    model.summary()
    print(f"\n[OK] Model built with {model.count_params():,} parameters")
    
    # Try to load existing checkpoint (periodic checkpoint resume)
    print("\n" + "=" * 60)
    print("CHECKING FOR EXISTING CHECKPOINTS...")
    print("=" * 60)

    resumed_model, initial_epoch = load_checkpoint_if_exists(CHECKPOINT_DIR)
    if resumed_model is not None:
        print("[OK] Resuming training using loaded checkpoint model")
        model = resumed_model
        # Need to (re)compile after loading a saved model with compile=False
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
        )
    else:
        print("[INFO] No periodic checkpoint found. Starting fresh training.")
    
    # Setup callbacks
    print("\n" + "=" * 60)
    print("SETTING UP CALLBACKS...")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callback_list = get_callbacks(timestamp, save_frequency=SAVE_PERIODIC_CHECKPOINT_EVERY_N_EPOCHS)
    
    print("[OK] Callbacks configured:")
    print("     - ModelCheckpoint (best model based on val_accuracy)")
    print("     - ModelCheckpoint (best weights)")
    print("     - ModelCheckpoint (periodic checkpoints every 5 epochs)")
    print("     - EarlyStopping (patience=15)")
    print("     - ReduceLROnPlateau")
    print("     - TensorBoard logging")
    
    # Training
    print("\n" + "=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60)
    print(f"Training on {len(X_train)} samples")
    print(f"Validating on {len(X_val)} samples")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 60 + "\n")
    
    # Train the model
    history = model.fit(
        X_train, y_train_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        initial_epoch=initial_epoch if initial_epoch is not None else 0,
        validation_data=(X_val, y_val_cat),
        callbacks=callback_list,
        verbose=1
    )
    
    # Load the best checkpoint from THIS run (if available), then optionally weight-average
    print("\n" + "=" * 60)
    print("LOADING BEST CHECKPOINT...")
    print("=" * 60)

    best_checkpoint = os.path.join(CHECKPOINT_DIR, f"best_model_{timestamp}.h5")
    if not os.path.exists(best_checkpoint):
        best_checkpoint = find_best_checkpoint(CHECKPOINT_DIR)
    if best_checkpoint:
        try:
            import tensorflow.keras as keras
            best_model = keras.models.load_model(best_checkpoint, compile=False)
            # Recompile with same settings
            optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            best_model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
            )
            print(f"[OK] Loaded best checkpoint: {os.path.basename(best_checkpoint)}")

            model = best_model

            if WEIGHT_AVG_USE:
                print("\n" + "=" * 60)
                print("TRYING WEIGHT AVERAGING FROM PERIODIC CHECKPOINTS...")
                print("=" * 60)
                averaged = try_weight_average_from_checkpoints(
                    CHECKPOINT_DIR, model, X_val, y_val_cat, WEIGHT_AVG_NUM_CHECKPOINTS
                )
                if averaged is not None:
                    model = averaged
        except Exception as e:
            print(f"[WARNING] Could not load best checkpoint: {e}")
            print("[INFO] Using final training model instead")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET...")
    print("=" * 60)
    
    test_results = model.evaluate(X_test, y_test_cat, verbose=1)
    test_accuracy = test_results[1]
    print(f"[OK] Test accuracy: {test_accuracy:.4f}")
    
    # Save final model (using best checkpoint)
    print("\n" + "=" * 60)
    print("SAVING FINAL MODEL (BEST CHECKPOINT)...")
    print("=" * 60)
    
    model.save(MODEL_SAVE_PATH)
    print(f"[OK] Keras model (best checkpoint) saved to: {MODEL_SAVE_PATH}")
    
    # Save skeleton model format (for compatibility with model.p / app.py)
    save_skeleton_model(X_train_skel, y_train, label_encoder, SKELETON_MODEL_SAVE_PATH)
    
    # Save class names and metadata
    metadata = {
        'class_names': class_names.tolist(),
        'class_mapping': {int(i): name for i, name in enumerate(class_names)},
        'num_classes': num_classes,
        'input_dim': input_dim,
        'support_two_hands': SUPPORT_TWO_HANDS,
        'use_preprocessed_landmarks': USE_PREPROCESSED_LANDMARKS
    }
    
    metadata_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Metadata saved to: {metadata_path}")
    
    # Training summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    best_val_acc = max(history.history['val_accuracy'])
    best_val_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\n[RESULTS]")
    print(f"  Best validation accuracy: {best_val_acc:.4f} (epoch {best_val_epoch})")
    print(f"  Final training accuracy:  {final_train_acc:.4f}")
    print(f"  Final validation accuracy: {final_val_acc:.4f}")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"\n[SAVED FILES]")
    print(f"  Keras Model: {MODEL_SAVE_PATH}")
    print(f"  Skeleton Model: {SKELETON_MODEL_SAVE_PATH}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    
    # Plot training history (optional)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(CHECKPOINT_DIR, f"training_history_{timestamp}.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\n[OK] Training plot saved to: {plot_path}")
        
    except ImportError:
        print("\n[INFO] matplotlib not installed - skipping training plot")
    
    print("\n" + "=" * 60)
    print("DONE! Your model is ready for inference.")
    print("=" * 60)
    print("\n[USAGE]")
    print("  The model can now be used in app.py for real-time prediction.")
    print("  It supports both single hand and two hands detection.")
    
    return history


# =========================================================
# ENTRY POINT - Required for Windows multiprocessing
# =========================================================
if __name__ == '__main__':
    # Required for Windows
    freeze_support()
    
    # Run training
    main()
