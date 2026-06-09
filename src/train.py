"""模型训练与评估模块。

训练三个经典文本分类模型，全部在【测试集】上诚实评估：
    - Naive Bayes        (MultinomialNB)
    - Logistic Regression
    - Linear SVM         (LinearSVC)

修正当年的两个坑：
    1. 神经网络只报训练准确率  -> 这里每个模型都报【测试集】准确率
    2. 混淆矩阵画的是循环里最后一个模型 -> evaluate_model 明确针对【指定】模型
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def get_models():
    """返回要对比的三个模型：{名字: 模型实例}。"""
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """训练所有模型，返回 {名字: (已训练模型, 测试准确率)}。

    每个模型都在【测试集】上算准确率——诚实评估，杜绝训练集刷分。
    """
    results = {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)            # 用训练集学
        y_pred = model.predict(X_test)         # 在测试集预测
        acc = accuracy_score(y_test, y_pred)   # 在测试集打分
        results[name] = (model, acc)
        print(f"{name:22} test accuracy = {acc:.4f}")
    return results


def best_model(results):
    """从结果里挑出测试准确率最高的模型，返回 (名字, 模型, 准确率)。"""
    name = max(results, key=lambda n: results[n][1])
    model, acc = results[name]
    return name, model, acc


def evaluate_model(model, X_test, y_test):
    """对【指定】模型做详细评估。

    明确针对传入的 model，避免当年"混淆矩阵用了循环里最后一个模型"的 bug。

    返回：
        y_pred —— 该模型在测试集上的预测
        cm     —— 混淆矩阵（行=真实类别，列=预测类别）
        labels —— 类别名顺序（画热力图时用来标坐标轴）
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    labels = sorted(y_test.unique())          # 固定类别顺序，混淆矩阵才对得上
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return y_pred, cm, labels
