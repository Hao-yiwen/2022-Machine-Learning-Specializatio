import random
from collections import defaultdict

# 示例文本
text = "I love deep learning and natural language processing. Deep learning is amazing and powerful. xxxx zzzz"

# 数据预处理：将文本拆分为词列表
words = text.split()

# 构建三元组 (3-gram) 模型
n_gram_model = defaultdict(list)

for i in range(len(words) - 2):
    key = (words[i], words[i + 1])  # 生成三元组的前两个词作为 key
    next_word = words[i + 2]  # 第三个词作为下一个词
    n_gram_model[key].append(next_word)  # 将下一个词添加到对应的键列表中

# 定义文本生成函数
def generate_text(start_words, num_words=10):
    generated_words = list(start_words)  # 将初始词对存入生成结果中
    for _ in range(num_words):
        key = tuple(generated_words[-2:])  # 每次取最后两个词作为新键
        next_words = n_gram_model.get(key)
        if not next_words:
            break  # 如果找不到后续词，则停止生成
        next_word = random.choice(next_words)  # 随机选择一个后续词
        generated_words.append(next_word)
    return ' '.join(generated_words)




def main():
    # 使用指定的起始词对生成文本
    start_words = ("Deep", "learning")
    generated_text = generate_text(start_words, num_words=10)
    print("生成的文本:", generated_text)

if __name__ == "__main__":
    main()