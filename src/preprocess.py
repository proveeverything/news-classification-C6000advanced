"""文本清洗模块：把一段原始新闻文本，清洗成干净的小写单词序列。

为什么单独写成一个文件、一个函数？
- 清洗逻辑只在这里写一次，notebook 里 import 调用即可；
- 想改清洗规则只改这一处，杜绝"复制 4 遍、改漏一处"那种沉默 bug。
"""

import re
import html

import pandas as pd
import nltk
from nltk.corpus import stopwords

# nltk 的停用词表需要下载一次；quiet=True 表示已存在就不再打扰。
nltk.download("stopwords", quiet=True)

# 英文停用词集合（the / is / at 这类高频但无信息量的词）。
# 用 set 而不是 list：判断 "word in ..." 时 set 快得多。
STOP_WORDS = set(stopwords.words("english"))

# 残缺 HTML 实体的"尸体"：&quot; 等被弄丢开头的 & 之后，只剩 quot/amp/gt 这类碎片，
# 都不是真英文单词（这批数据里 quot 残留了 595 次），直接拉黑。
HTML_JUNK = {"quot", "lt", "gt", "amp", "nbsp", "apos"}

# 合成一张"要丢弃的词"总黑名单。
DROP_WORDS = STOP_WORDS | HTML_JUNK


def clean_text(text: str) -> str:
    """把一段原始文本清洗成干净的小写词序列。

    步骤顺序很重要（尤其第 1 步必须最先做）：
        1. 解码 HTML 实体：  &lt; -> <      &#39; -> '
        2. 转小写：          Apple -> apple
        3. 只保留字母：      数字 / 标点 / 符号 -> 空格
        4. 去噪词：          停用词 + HTML 残骸(quot/amp...) + 单字母碎片(u/b...)
        5. 压缩空格：        多个空格并成一个

    例：
        "Apple Inc.&lt;AAPL.O&gt; on Tuesday" -> "apple inc aapl tuesday"
    """
    # 1) 先解码 HTML 实体——否则下一步删符号时 &lt; 会被剥成垃圾词 "lt"。
    text = html.unescape(str(text))

    # 2) 转小写，让 "Apple" 和 "apple" 视为同一个词。
    text = text.lower()

    # 3) 只留 a-z 和空白；其余一律换成空格
    #    （换成空格而不是直接删除，避免相邻两个词粘在一起）。
    text = re.sub(r"[^a-z\s]", " ", text)

    # 4) 切成单词，丢掉：黑名单词（停用词 + HTML 残骸）和单字母碎片（u/b/n...）。
    words = [w for w in text.split() if w not in DROP_WORDS and len(w) > 1]

    # 5) 用单个空格拼回（split + join 顺带把多余空格压成一个）。
    return " ".join(words)


# 类别号 -> 可读名字，画图和看混淆矩阵时用得上。
LABEL_MAP = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def load_data(csv_path: str) -> pd.DataFrame:
    """读取新闻 CSV，做完整清洗，返回一张可直接建模的干净表。

    流程：
        1. 读 CSV
        2. 去缺失行、去重复行，重置行号
        3. 给类别号配上可读类别名（1 -> World ...）
        4. 对 Title 和 Description **都**做文本清洗（不再只洗标题）
        5. 合并成一列 text = 清洗后的标题 + ' ' + 清洗后的正文

    返回的 DataFrame 多出两列：
        category —— 可读类别名
        text     —— 清洗 + 合并后，准备喂给模型的文本
    """
    # 1) 读 CSV。
    df = pd.read_csv(csv_path)

    # 2) 数据卫生：去缺失、去重复。reset_index 把行号重排成 0,1,2...
    #    （删行后行号会留空洞，重置一下后面用起来不踩坑）。
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # 3) 类别号 -> 可读名字。
    df["category"] = df["Class Index"].map(LABEL_MAP)

    # 4) 关键修正：标题和正文**都**清洗（当年只洗了标题）。
    title_clean = df["Title"].apply(clean_text)
    desc_clean = df["Description"].apply(clean_text)

    # 5) 标题信息太少，拼上正文给足上下文，合成一列喂给模型。
    df["text"] = (title_clean + " " + desc_clean).str.strip()

    return df
