import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import time

class ChineseSentimentDataset(Dataset):
    """自定义数据集"""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 使用tokenizer进行编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertFineTuner:
    def __init__(self, model_name='bert-base-chinese', num_labels=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def prepare_data(self, texts, labels, test_size=0.2, max_len=128, batch_size=16):
        """准备训练和验证数据"""
        # 划分训练集和验证集
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )

        # 创建数据集
        train_dataset = ChineseSentimentDataset(
            train_texts,
            train_labels,
            self.tokenizer,
            max_len
        )

        val_dataset = ChineseSentimentDataset(
            val_texts,
            val_labels,
            self.tokenizer,
            max_len
        )

        # 创建数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )

        return train_dataloader, val_dataloader

    def train(self, train_dataloader, val_dataloader, epochs=3, learning_rate=2e-5):
        """训练模型"""
        # 优化器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # 学习率调度器
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # 训练循环
        for epoch in range(epochs):
            self.logger.info(f'Epoch {epoch + 1}/{epochs}')
            start_time = time.time()

            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch in train_dataloader:
                # 将数据移到GPU
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                loss = outputs.loss
                train_loss += loss.item()

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = train_loss / len(train_dataloader)

            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_accuracy = 0
            val_steps = 0

            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )

                    loss = outputs.loss
                    val_loss += loss.item()

                    # 计算准确率
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    val_accuracy += (predictions == labels).sum().item()
                    val_steps += len(labels)

            avg_val_loss = val_loss / len(val_dataloader)
            val_accuracy = val_accuracy / val_steps

            # 记录训练信息
            time_elapsed = time.time() - start_time
            self.logger.info(f'Training Loss: {avg_train_loss:.4f}')
            self.logger.info(f'Validation Loss: {avg_val_loss:.4f}')
            self.logger.info(f'Validation Accuracy: {val_accuracy:.4f}')
            self.logger.info(f'Time: {time_elapsed:.2f}s')

    def save_model(self, path):
        """保存模型"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        """加载模型"""
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model.to(self.device)

    def predict(self, texts, max_len=128):
        """预测新数据"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=max_len,
                    return_token_type_ids=True,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                token_type_ids = encoding['token_type_ids'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

                logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1)
                predictions.append(prediction.item())

        return predictions

# 使用示例
def main():
    # 示例数据
    texts = [
        "这个产品质量很好，我很满意",
        "服务态度差，不推荐购买",
        "价格合理，值得购买",
        "产品性能优秀，超出预期",
        "太差了，完全是浪费钱",
        "物流速度快，包装完好",
        "客服很热情，解决了我的问题",
        "质量有问题，退货也不给处理",
        "性价比很高，推荐购买",
        "产品有质量问题，很失望"
    ]
    labels = [1, 0, 1]  # 1表示正面，0表示负面

    # 初始化微调器
    finetuner = BertFineTuner()

    # 准备数据
    train_dataloader, val_dataloader = finetuner.prepare_data(
        texts,
        labels,
        test_size=0.2,
        max_len=128,
        batch_size=16
    )

    # 训练模型
    finetuner.train(
        train_dataloader,
        val_dataloader,
        epochs=3,
        learning_rate=2e-5
    )

    # 保存模型
    finetuner.save_model('finetuned-bert-sentiment')

    # 预测新数据
    new_texts = [
        "这个产品非常好用",
        "质量很差，退货了"
    ]
    predictions = finetuner.predict(new_texts)
    print("预测结果:", predictions)

if __name__ == "__main__":
    main()