import re
from collections import Counter
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ChineseNLPProcessor:
    def __init__(self):
        # 停用词列表
        self.stop_words = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])
        
    def text_cleanup(self, text):
        """
        文本清理：去除特殊字符、多余空格等
        """
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 移除URL
        text = re.sub(r'http\S+|www.\S+', '', text)
        # 移除表情符号
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
        return text.strip()
    
    def segment_text(self, text):
        """
        分词处理
        """
        words = jieba.cut(text)
        return [word for word in words if word not in self.stop_words and len(word.strip()) > 0]
    
    def extract_keywords(self, text, top_n=5):
        """
        提取关键词：使用jieba.analyse的TF-IDF算法提取关键词
        """
        import jieba.analyse
        # 使用jieba的TF-IDF算法提取关键词，返回词和权重
        keywords = jieba.analyse.extract_tags(text, topK=top_n, withWeight=True)
        return keywords
    
    def calculate_similarity(self, text1, text2):
        """
        计算两段文本的相似度
        """
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return round(similarity, 4)
        except:
            return 0.0
    
    def sentiment_analysis_simple(self, text):
        """
        简单情感分析（基于情感词典）
        """
        # 示例情感词典（实际使用时应该使用完整的情感词典）
        positive_words = set(['喜欢', '好', '优秀', '棒', '完美', '快乐', '开心', '出色'])
        negative_words = set(['差', '糟糕', '讨厌', '失望', '难过', '不好', '低劣', '痛苦'])
        
        words = self.segment_text(text)
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def extract_summary(self, text, num_sentences=3):
        """
        提取文本摘要（基于TextRank算法）
        使用 networkx 库实现 TextRank 算法，更简洁高效
        """
        import networkx as nx
        
        # 分句
        sentences = re.split(r'[。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return ""
            
        # 构建图
        graph = nx.Graph()
        
        # 添加节点和边
        for i, sent_i in enumerate(sentences):
            for j, sent_j in enumerate(sentences):
                if i < j:  # 避免重复计算
                    similarity = self.calculate_similarity(sent_i, sent_j)
                    if similarity > 0:  # 只添加有意义的边
                        graph.add_edge(i, j, weight=similarity)
        
        # 使用 PageRank 算法计算句子重要性得分
        scores = nx.pagerank(graph, alpha=0.85)
        
        # 选择得分最高的句子
        ranked_sentences = [(sent, scores.get(i, 0)) 
                          for i, sent in enumerate(sentences)]
        ranked_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # 按原文顺序重排选中的句子
        selected_sentences = ranked_sentences[:num_sentences]
        selected_sentences.sort(key=lambda x: sentences.index(x[0]))
        
        summary = '。'.join(s[0] for s in selected_sentences) + '。'
        return summary

    def word_segmentation_analysis(self, text):
        """
        词性标注分析
        """
        import jieba.posseg as pseg
        words = pseg.cut(text)
        return [(word.word, word.flag) for word in words]

# 使用示例
def main():
    nlp = ChineseNLPProcessor()
    
    # 示例文本
    text = """
    今天天气真不错，阳光明媚，我和朋友去公园散步。
    公园里人很多，有人在跑步，有人在打太极，还有人在下棋。
    这样的周末生活真是惬意，让人感到很放松和开心。
    """
    
    # 清理文本
    cleaned_text = nlp.text_cleanup(text)
    print("清理后的文本:", cleaned_text)
    
    # 分词
    words = nlp.segment_text(cleaned_text)
    print("\n分词结果:", ' '.join(words))
    
    # 提取关键词
    keywords = nlp.extract_keywords(cleaned_text)
    print("\n关键词:", keywords)
    
    # 情感分析
    sentiment = nlp.sentiment_analysis_simple(cleaned_text)
    print("\n情感分析结果:", sentiment)
    
    # 生成摘要
    summary = nlp.extract_summary(cleaned_text)
    print("\n文本摘要:", summary)
    
    # 词性标注
    pos_tags = nlp.word_segmentation_analysis(cleaned_text)
    print("\n词性标注:", pos_tags[:10])  # 只显示前10个结果

if __name__ == "__main__":
    main()