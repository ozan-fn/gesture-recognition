import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import pickle

# === CONFIGURATION ===
DATA_DIR = 'MP_Data'
MODEL_PATH = 'bisindo_model.h5'
LABELS_PATH = 'class_names.pkl'
SCALER_PATH = 'scaler.pkl'
SEQUENCE_LENGTH = 30
SEED = 42

def load_data():
    X, y = [], []
    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(DATA_DIR, class_name)
        for file in os.listdir(class_dir):
            if file.endswith('.npy'):
                filepath = os.path.join(class_dir, file)
                try:
                    data = np.load(filepath)
                    if data.size == 0 or data.shape[0] != SEQUENCE_LENGTH:
                        print(f"  [Warn] Skipping corrupted/empty file: {filepath}")
                        continue
                    X.append(data)
                    y.append(class_name)
                except Exception as e:
                    print(f"  [Warn] Failed to load {filepath}: {e}")
                    # Delete corrupted files to clean up
                    try:
                        os.remove(filepath)
                        print(f"  [Cleaned] Deleted corrupted file: {filepath}")
                    except:
                        pass
    X = np.array(X)
    y = np.array(y)
    return X, y, class_names

def main():
    X, y, class_names = load_data()
    print(f"Loaded {X.shape[0]} samples with shape {X.shape[1:]}, {len(class_names)} classes.")

    # Flatten sequence for scaling
    num_samples, seq_len, num_features = X.shape
    X_flat = X.reshape(-1, num_features)
    
    # Fit StandardScaler
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    
    # Reshape back to (samples, sequence_length, features)
    X_scaled = X_flat_scaled.reshape(num_samples, seq_len, num_features)

    # One-hot encode labels
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)
    if len(class_names) == 2:
        y_encoded = np.hstack((1 - y_encoded, y_encoded)) # Convert binary to categorical representation

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=SEED, stratify=y
    )

    # Model architecture
    input_shape = X_train.shape[1:]
    num_classes = y_encoded.shape[1]

    model = Sequential([
        LSTM(128, return_sequences=True, activation='tanh', input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, activation='tanh'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=16,
        callbacks=callbacks
    )

    # Save scaler
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    # Save class names
    with open(LABELS_PATH, 'wb') as f:
        pickle.dump(class_names, f)

    print(f"Training complete. Model: {MODEL_PATH}, Scaler: {SCALER_PATH}, Labels: {LABELS_PATH}.")

if __name__ == "__main__":
    main()
