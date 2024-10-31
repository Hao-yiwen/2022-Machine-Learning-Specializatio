# train_model.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


def train_digit_model():
    # 1. 加载MNIST数据集
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # 2. 数据预处理
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    # 3. 构建模型
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # 4. 编译模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 5. 训练模型
    model.fit(train_images, train_labels, epochs=5, batch_size=64,
              validation_data=(test_images, test_labels))

    # 6. 转换为TFLite格式
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # 7. 保存模型
    with open('digit_model.tflite', 'wb') as f:
        f.write(tflite_model)

    # 8. 保存为CoreML格式(iOS)
    import coremltools
    coreml_model = coremltools.convert(model, source='tensorflow',
                                       inputs=[coremltools.ImageType(shape=(1, 28, 28, 1))])
    coreml_model.save('DigitClassifier.mlmodel')


if __name__ == '__main__':
    train_digit_model()


# 图像预处理工具
def preprocess_image(image_path):
    """预处理输入图像"""
    img = tf.keras.preprocessing.image.load_img(
        image_path, color_mode='grayscale', target_size=(28, 28))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array.reshape(1, 28, 28, 1)
    img_array = img_array.astype('float32') / 255.0
    return img_array