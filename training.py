import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks
import matplotlib.pyplot as plt
from tensorflow.image import resize_with_pad
import os

# Constantes, 224x224 é o tamanho minimo da EfficientNet
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def preprocess_image(image, label):
    #Normalizando
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    image = resize_with_pad(image, target_height=224, target_width=224)

    # Preprocessamento da efficientnet (escala pixels entre -1 e 1)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label
    
def create_dataset(base_dir):
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    
    # Criar datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    ).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='binary',
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        shuffle=False
    ).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='binary',
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        shuffle=False
    ).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds

    
def data_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2, fill_mode='constant'),
        layers.RandomZoom(0.2, fill_mode='constant'),
        layers.RandomContrast(0.2),
        layers.GaussianNoise(0.01), # Ajuda com sensor noise
    ])

def build_rust_model():
    augmentation = data_augmentation()
    
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'
    )

    base_model.trainable = True 

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-10:]:
        layer.trainable = True
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    

    x = base_model(x, training=True) 
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x) 
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset_dir = 'dataset'  
    
    train_ds, val_ds, test_ds = create_dataset(dataset_dir)
    
    model = build_rust_model()
    
    history = model.fit(
        train_ds,
        epochs=30,
        validation_data=val_ds,
        callbacks=[
            callbacks.EarlyStopping(
                monitor='val_loss', # Escolha uma métrica principal para o EarlyStopping
                patience=5,
                restore_best_weights=True,
                mode='min'
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=3
            )
        ]
    )
    
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {test_acc:.2%}")
    
    plot_training_history(history)
    
    true_labels = np.concatenate([y for x, y in test_ds], axis=0)
    
    predictions_proba = model.predict(test_ds)
    
    predictions_binary = (predictions_proba > 0.5).astype(int)
    
    recall = recall_score(true_labels, predictions_binary)
    precision = precision_score(true_labels, predictions_binary)
    f1 = f1_score(true_labels, predictions_binary)
    
    print(f"Test Recall: {recall:.2f}")
    print(f"Test Precision: {precision:.2f}")
    print(f"Test F1-Score: {f1:.2f}")

    cm = confusion_matrix(true_labels, predictions_binary)
    print("\nConfusion Matrix:")
    print(cm)
    
    model.save('detector_ferrugem.h5')
