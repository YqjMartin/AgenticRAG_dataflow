'''
@ algorithm for computing similarity
@ supported: BM25, MinHash
'''



import math
from collections import defaultdict
import mmh3  # 使用 murmurhash3 哈希函数


class BM25:
    def __init__(self, documents):
        self.documents = documents  # 文档集合（每个snippet作为一个文档）
        self.N = len(documents)     # 文档总数
        self.avgdl = 0.0            # 平均文档长度
        self.idf = {}               # 逆文档频率
        self.doc_freq = defaultdict(int)  # 每个词出现的文档数
        self.calc_idf()
        self.calc_avgdl()

    def calc_idf(self):
        """计算逆文档频率(IDF)"""
        for doc in self.documents:
            for word in set(doc):
                self.doc_freq[word] += 1
        for word, freq in self.doc_freq.items():
            self.idf[word] = math.log((self.N - freq + 0.5) / (freq + 0.5)) + 1

    def calc_avgdl(self):
        """计算平均文档长度"""
        total_dl = 0
        for doc in self.documents:
            total_dl += len(doc)
        self.avgdl = total_dl / self.N if self.N > 0 else 0

    def bm25_score(self, query, doc_idx, k1=1.5, b=0.75):
        """
        计算BM25得分
        :param query: 查询词列表
        :param doc_idx: 文档索引
        :param k1: 调节参数, 通常在1.2到2.0之间
        :param b: 调节参数, 通常在0.5到0.8之间
        :return: BM25得分
        """
        score = 0.0
        doc = self.documents[doc_idx]
        dl = len(doc)
        for word in query:
            if word not in self.idf:
                continue
            idf = self.idf[word]
            tf = doc.count(word)
            numerator = idf * tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / self.avgdl)
            score += numerator / denominator
        return score

    def get_scores(self, query, k1=1.5, b=0.75):
        """
        计算query与所有snippets的BM25得分
        :param query: 查询词列表
        :param k1: 调节参数
        :param b: 调节参数
        :return: 包含所有snippet得分的字典,格式为 {doc_idx: score}
        """
        scores = {}
        for doc_idx in range(self.N):
            score = self.bm25_score(query, doc_idx, k1, b)
            scores[doc_idx] = score
        return scores


class MinHash:
    def __init__(self, num_perm=128):
        self.num_perm = num_perm  # 哈希函数的数量（特征维度）

    def _hash(self, content, seed):
        """计算哈希值"""
        return mmh3.hash(content, seed)

    def text_to_minhash(self, text:str):
        """将文本转换为 MinHash 特征向量"""
        words = text.split()  # 简单分词，实际应用中可以使用更复杂的分词方法
        minhash = [float('inf')] * self.num_perm

        for word in words:
            for i in range(self.num_perm):
                hash_value = self._hash(word, i)
                if hash_value < minhash[i]:
                    minhash[i] = hash_value

        return minhash

    def similarity(self, text1, text2):
        """计算两个文本的 Jaccard 相似度估计"""
        minhash1 = self.text_to_minhash(text1)
        minhash2 = self.text_to_minhash(text2)

        match = 0
        for h1, h2 in zip(minhash1, minhash2):
            if h1 == h2:
                match += 1

        return match / self.num_perm

# 示例用法
if __name__ == "__main__":
    #----------------- MinHash测试 --------------------
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog leaps over a lazy fox",
        "The quick brown fox jumps over a lazy dog",
        "Hello world, this is a test sentence"
    ]

    minhash = MinHash(num_perm=128)

    # 计算所有文本两两之间的相似度
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity_score = minhash.similarity(texts[i], texts[j])
            print(f"Similarity between text {i} and text {j}: {similarity_score:.4f}")

    # ----------------- BM25测试 ---------------------
    # 示例snippet集合
    snippets = [
        ["the", "quick", "brown", "fox"],
        ["jumps", "over", "the", "lazy", "dog"],
        ["the", "brown", "dog"],
        ["quick", "fox", "jumps"]
    ]

    # 创建BM25对象
    bm25 = BM25(snippets)

    # 查询
    query = ["quick", "brown", "fox"]

    # 计算所有snippet的BM25得分
    scores = bm25.get_scores(query)

    # 输出结果
    print("Query:", query)
    print("BM25 Scores for each snippet:")
    for doc_idx, score in scores.items():
        print(f"Snippet {doc_idx}: {score:.4f}")
    