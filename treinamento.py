# treinamento_cifar10_tf.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import datasets, layers, models, callbacks, Input

# 1) Carregar e normalizar o dataset CIFAR-10
(x_full, y_full), _ = datasets.cifar10.load_data()
x_full = x_full.astype('float32') / 255.0
y_full = y_full.flatten()  # de (n,1) para (n,)

# 2) Dividir em 80% treino e 20% teste
x_train, x_test, y_train, y_test = train_test_split(
    x_full, y_full, test_size=0.2, stratify=y_full, random_state=42
)

# 3) Data Augmentation (REMOVIDO)

# 4) Definição da arquitetura CNN (com Input explícito)
model = models.Sequential([
    Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 5) Compilar o modelo (só accuracy)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6) Callbacks: EarlyStopping e ModelCheckpoint
es = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
mc = callbacks.ModelCheckpoint('best_cifar10_tf.h5', save_best_only=True)

# 7) Treinamento
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[es, mc]
)

# 8) Avaliação no conjunto de teste
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# 9) Relatório de classificação (Precision, Recall, F1-score)
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_classes,
    target_names=[
        'airplane','automobile','bird','cat','deer',
        'dog','frog','horse','ship','truck'
    ]
))

# 10) Salvar o modelo final
model.save('cifar10_cnn_tf_final.h5')

# 11) Plot de métricas de treino e validação
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.legend()

plt.show()
