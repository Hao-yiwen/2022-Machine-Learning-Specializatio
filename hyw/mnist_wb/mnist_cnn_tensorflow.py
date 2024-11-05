import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 数据预处理
# 将图像数据重塑为 (28, 28, 1)，并归一化到 [0,1] 区间
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images  = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 将标签转换为独热编码（One-Hot Encoding）
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels  = tf.keras.utils.to_categorical(test_labels)

# 构建 CNN 模型
model = models.Sequential()

# 第一层卷积和池化
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# 第二层卷积和池化
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 第三层卷积
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 展平层
model.add(layers.Flatten())

# 全连接层
model.add(layers.Dense(64, activation='relu'))

# 输出层
model.add(layers.Dense(10, activation='softmax'))

# 查看模型架构
model.summary()

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_images, train_labels, epochs=5, batch_size=64,
                    validation_data=(test_images, test_labels))

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\n测试集准确率: {test_acc:.4f}')

# 绘制训练和验证的准确率和损失
plt.figure(figsize=(12, 4))

# 准确率
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.xlabel('Epoch')
plt.ylabel('准确率')
plt.legend()

# 损失
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()

plt.show()

# 测试模型预测
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], np.argmax(true_label[i]), img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img = img.reshape((28,28))
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f"预测: {predicted_label} ({100*np.max(predictions_array):.2f}%)\n真实: {true_label}", color=color)

# 获取预测结果
predictions = model.predict(test_images)

# 可视化预测结果
num_rows = 5
num_cols = 5
num_images = num_rows * num_cols
plt.figure(figsize=(2*num_cols, 2*num_rows))

for i in range(num_images):
    plt.subplot(num_rows, num_cols, i+1)
    plot_image(i, predictions, test_labels, test_images)

plt.tight_layout()
plt.show()

## 保存模型
model.save('mnist_cnn_model.h5')

# 转换为 TensorFlow Lite 模型
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('mnist_cnn.tflite', 'wb') as f:
    f.write(tflite_model)

# （可选）整数量化
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
with open('mnist_cnn_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)


if __name__ == '__main__':
    pass