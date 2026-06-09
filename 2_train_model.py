import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# === CONFIGURATION ===
DATA_DIR = 'MP_Data'
MODEL_PATH = 'bisindo_model.h5'
LABELS_PATH = 'class_names.pkl'
SCALER_PATH = 'scaler.pkl'
REPORT_DIR = 'evaluation_results'
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

    print(f"\nTraining complete. Model: {MODEL_PATH}, Scaler: {SCALER_PATH}, Labels: {LABELS_PATH}.")
    
    # === EVALUATION ===
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Overall Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n[Overall Accuracy]: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Precision, Recall, F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    print("\n[Per-Class Metrics]:")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 66)
    for i, cls in enumerate(class_names):
        print(f"{cls:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    # Weighted averages
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    print("-" * 66)
    print(f"{'Weighted Avg':<20} {precision_avg:<12.4f} {recall_avg:<12.4f} {f1_avg:<12.4f} {len(y_true):<10}")
    
    # Classification Report (detailed)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n[Detailed Classification Report]:")
    print(report)
    
    # Save report to file
    with open(os.path.join(REPORT_DIR, 'classification_report.txt'), 'w') as f:
        f.write("GESTURE RECOGNITION - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n[Confusion Matrix]:")
    print(cm)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"\n[Saved] Confusion matrix → {REPORT_DIR}/confusion_matrix.png")
    plt.close()
    
    # Plot Training History
    plt.figure(figsize=(14, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
    print(f"[Saved] Training history → {REPORT_DIR}/training_history.png")
    plt.close()
    
    # Per-class accuracy
    class_accuracy = []
    for i in range(len(class_names)):
        mask = (y_true == i)
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).sum() / mask.sum()
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0.0)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_accuracy, color='steelblue', edgecolor='black')
    plt.axhline(y=accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall Accuracy: {accuracy:.4f}')
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
    print(f"[Saved] Per-class accuracy → {REPORT_DIR}/per_class_accuracy.png")
    plt.close()
    
    print("\n" + "="*60)
    print(f"All evaluation results saved to: {REPORT_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()
