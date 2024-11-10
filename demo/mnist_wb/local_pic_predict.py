import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class DigitPredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def preprocess_image(self, image_path):
        # 读取图片
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = image_path

        # 确保图片是灰度的
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 二值化（不反色）
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # 调整大小到28x28
        img = cv2.resize(img, (28, 28))

        # 归一化
        img = img.astype('float32') / 255.0

        # 添加通道维度和批次维度
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        # 显示处理后的图片
        cv2.imshow('Processed Image', img[0])

        return img

    def predict_with_preprocessing(self, image_path):
        # 预处理
        img = self.preprocess_image(image_path)

        # 预测
        prediction = self.model.predict(img)

        # 获取结果
        digit = np.argmax(prediction[0])
        confidence = float(prediction[0][digit])

        return {
            'digit': int(digit),
            'confidence': confidence,
            'probabilities': prediction[0].tolist()
        }


# 使用示例
def main():
    predictor = DigitPredictor('mnist_cnn_model.h5')

    # 选择模式
    mode = input("选择模式 (1: 摄像头, 2: 文件): ")



    # 从摄像头捕获图片
    def capture_and_predict():
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 显示原始帧
            cv2.imshow('Frame', frame)

            # 按空格键捕获并预测
            key = cv2.waitKey(1)
            if key == 32:  # 空格键
                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 预测
                result = predictor.predict_with_preprocessing(gray)

                print(f"预测的数字是: {result['digit']}")
                print(f"置信度: {result['confidence']:.2%}")

            # 按 'q' 退出
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # 从文件预测
    def predict_from_file(image_path):
        result = predictor.predict_with_preprocessing(image_path)

        # 显示图片和预测结果
        img = cv2.imread(image_path)
        cv2.putText(img, f"Digit: {result['digit']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.putText(img, f"Conf: {result['confidence']:.2%}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow('Prediction', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if mode == '1':
        main()
    elif mode == '2':
        image_path = input("输入图片路径: ")
        predict_from_file(image_path)


if __name__ == '__main__':
    main()