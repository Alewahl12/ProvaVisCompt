import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Carregar o modelo treinado
model = tf.keras.models.load_model('cifar10_cnn_tf_final.h5')

# Mapear a classe prevista para o nome da classe
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Caminho da pasta de imagens
img_folder = 'imagens'
img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

plt.figure(figsize=(15, 5))
plt.suptitle("Predições do Modelo CNN para CIFAR-10", fontsize=16)
for idx, img_file in enumerate(img_files):
    img_path = os.path.join(img_folder, img_file)
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array_exp = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array_exp)
    predicted_class = np.argmax(predictions[0])

    plt.subplot(1, len(img_files), idx + 1)
    plt.imshow(img_array.astype('uint8'))
    plt.title(class_names[predicted_class])
    plt.axis('off')

plt.tight_layout()
plt.show()
