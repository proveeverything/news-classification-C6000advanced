"""特征工程模块：把清洗好的文本切分成训练/测试集，并转成 TF-IDF 数字矩阵。

机器学习模型只吃数字，不吃文字。TF-IDF 负责把每条新闻变成一个数字向量：
某个词在这条新闻里出现得多、且在整批数据里少见，就给它高分。

核心铁律（防数据泄露）：
    TF-IDF 的 IDF 是"数出来"的统计量，所以只能用【训练集】去 fit；
    测试集只能 transform，绝不能参与 fit——否则等于模型偷看了考题。
"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def split_data(df, test_size=0.2, random_state=42):
    """按类别分层，把原始文本切成训练/测试两份。

    TF-IDF 线和神经网络线**共用这一个切分**，保证两边在同样的数据上对比，
    才叫公平。固定 random_state 让结果可复现。

    返回：
        text_train, text_test —— 清洗后的文本（字符串）
        y_train, y_test       —— 对应的类别标签
    """
    return train_test_split(
        df["text"],
        df["category"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["category"],   # 让四个类别在训练/测试里比例一致
    )


def build_features(df, test_size=0.2, random_state=42):
    """把清洗好的 DataFrame 变成可直接喂给模型的 TF-IDF 训练/测试特征。

    返回：
        X_train, X_test —— TF-IDF 稀疏矩阵（数字特征）
        y_train, y_test —— 对应的类别标签
        vectorizer      —— 已 fit 好的向量器（将来预测新新闻要用同一个）
    """
    # 1) 必须【先切分】，再做 TF-IDF。
    #    顺序错了（先 fit 再切）会让测试集"漏"进 IDF 统计 -> 数据泄露。
    text_train, text_test, y_train, y_test = split_data(df, test_size, random_state)

    # 2) 只用【训练文本】fit：学词表 + 学每个词的 IDF。
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(text_train)   # 训练集：fit 同时 transform
    X_test = vectorizer.transform(text_test)         # 测试集：只 transform，不再 fit

    return X_train, X_test, y_train, y_test, vectorizer
