import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Dense, Flatten, Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import Callback
import pickle
from sklearn.metrics import roc_curve

def resnet_layer(inputs, num_filters=64, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(inputs)
    if batch_normalization:
        conv = BatchNormalization()(conv)
    if activation is not None:
        conv = Activation(activation)(conv)
    return conv

def resnet_block(inputs, num_filters=64, num_blocks=3, downsample=False):
    x = resnet_layer(inputs=inputs, num_filters=num_filters, strides=2 if downsample else 1)
    for _ in range(1, num_blocks):
        y = resnet_layer(inputs=x, num_filters=num_filters, strides=1)
        y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
        x = Add()([x, y])
        x = Activation('relu')(x)
    return x

def build_resnet34():
    inputs = Input(shape=(None, 13, 1))
    x = resnet_layer(inputs, num_filters=32, strides=2)
    x = resnet_block(x, num_filters=32, num_blocks=3)
    x = resnet_block(x, num_filters=64, num_blocks=4, downsample=True)
    x = resnet_block(x, num_filters=128, num_blocks=6, downsample=True)
    x = resnet_block(x, num_filters=256, num_blocks=3, downsample=True)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    # Find the nearest point where FPR equals FNR
    eer_threshold_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2
    return eer

def train_and_evaluate_model(fold, X_train, y_train, X_val, y_val, file_names, scenario, batch_size=32, epochs=100, learning_rate=0.001, n_splits=5, save_path='model_per_fold'):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    epochs_eers = []
    all_histories = []
    best_model = None
    best_accuracy = 0

    print("Building model")

    try:
        model = build_resnet34()
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=BinaryCrossentropy(), metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        # model_checkpoint = ModelCheckpoint(os.path.join(save_path, f'{scenario}/model_fold_{epochs}_{batch_size}_{learning_rate}_fold-{fold+1}_for_test.h5'), save_best_only=True, monitor='val_accuracy', mode='max')

        print("set to float 32")
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        X_val = np.array(X_val, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)

        print(f"Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")

        # Create TensorFlow datasets with padding
        padded_shapes = ([None, 13, 1], [])
        print("create training dataset")
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).padded_batch(batch_size, padded_shapes=padded_shapes)
        print("create val dataset")
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).padded_batch(batch_size, padded_shapes=padded_shapes)

        print("Starting training")

        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=[early_stopping],
            # callbacks=[early_stopping, model_checkpoint],
            verbose=2
        )
    except Exception as e:
        print(f"Error during training fold {fold}: {e}")

    predictions = model.predict(X_val)
    output_path = f"./prediction_per_fold/{scenario}/predictions_{epochs}_{batch_size}_{learning_rate}_fold-{fold}.txt"
    with open(output_path, 'w') as f:
        for idx, prediction in enumerate(predictions):
            filename = file_names[idx]
            f.write(f"{filename} - Predicted: {prediction[0]:.1f} - Actual: {y_val[idx]}\n")
    print(f"Predictions logged successfully to {output_path}")    

    model_file_path = f'./best_model/{scenario}/model_{learning_rate}_{epochs}_fold-{fold}.h5'
    model.save(model_file_path)
    print(f"Best model for fold {fold} saved at: {model_file_path}")

    return all_histories, epochs_eers
