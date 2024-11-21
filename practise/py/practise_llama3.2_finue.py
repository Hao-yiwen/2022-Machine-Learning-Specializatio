# !pip install transformers datasets peft bitsandbytes accelerate

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
import os


def prepare_dataset(tokenizer, dataset_name="databricks/databricks-dolly-15k"):
    """准备并处理数据集"""
    # 1. 加载数据集
    dataset = load_dataset(dataset_name, split="train")

    # 打印数据集的一个样本，查看格式
    print("Dataset example:", dataset[0])

    def format_conversation(example):
        """格式化对话格式"""
        # 检查数据集的实际字段
        instruction = example.get('instruction', '')
        context = example.get('context', '')
        response = example.get('response', '')

        conversation = f"""### Instruction: {instruction}

### Input: {context if context else 'No additional context provided.'}

### Response: {response}"""
        return conversation

    def preprocess_function(examples):
        # 构建对话
        conversations = []
        print(f"len(examples['instruction']) {len(examples['instruction'])}")
        for i in range(10):  # 使用任意一个字段的长度作为循环次数
            example = {
                'instruction': examples['instruction'][i],
                'context': examples['context'][i] if 'context' in examples else '',
                'response': examples['response'][i]
            }
            conversations.append(format_conversation(example))

        # tokenize
        tokenized = tokenizer(
            conversations,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )

        # 设置标签
        tokenized["labels"] = tokenized["input_ids"].clone()

        return tokenized

    # 3. 应用预处理
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset


def prepare_model(model_id, token):
    """准备模型进行训练"""
    # 1. 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 2. 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 3. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 4. 为 LoRA 准备模型
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # LoRA 秩
        lora_alpha=32,  # LoRA alpha参数
        lora_dropout=0.05,  # Dropout 概率
        target_modules=[  # 需要训练的模块
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
        ],
    )

    # 6. 获取 PEFT 模型
    model = get_peft_model(model, lora_config)

    return model, tokenizer


def train(model, tokenizer, train_dataset, output_dir="./results"):
    """训练模型"""
    # 1. 训练参数配置
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.001,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        # 添加以下参数确保正确保存
        save_safetensors=True,
        save_strategy="steps",
    )

    # 2. 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 3. 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 4. 开始训练
    trainer.train()

    # 5. 保存模型和配置
    # 保存 PEFT 配置和权重
    model.save_pretrained(output_dir)
    # 保存 tokenizer
    tokenizer.save_pretrained(output_dir)

    return trainer


def inference_example(model_id, token, model_path):
    """测试微调后的模型"""
    try:
        # 1. 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # 2. 加载 PEFT 模型
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # 3. 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True
        )

        # 4. 测试输入
        test_input = """### Instruction: Tell me about artificial intelligence

### Input: I want to learn about AI

### Response:"""

        # 5. 生成回答
        inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            num_return_sequences=1
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    except Exception as e:
        print(f"Error during inference: {e}")
        print("Detailed error information:")
        import traceback
        traceback.print_exc()
        return None


def main():
    # 1. 设置参数
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    token = "xxxx"
    output_dir = "./finetuned_llama"

    print("Preparing model...")
    # 2. 准备模型和tokenizer
    model, tokenizer = prepare_model(model_id, token)

    # 3. 准备数据集
    print("Preparing dataset...")
    train_dataset = prepare_dataset(tokenizer)

    # 4. 训练模型
    print("Starting training...")
    trainer = train(model, tokenizer, train_dataset, output_dir)

    # 5. 等待一下确保文件都保存完成
    import time
    time.sleep(5)

    # 6. 测试微调后的模型
    print("\nTesting fine-tuned model...")
    response = inference_example(model_id, token, output_dir)
    if response:
        print("Generated Response:", response)
    else:
        print("Inference failed.")


if __name__ == "__main__":
    main()