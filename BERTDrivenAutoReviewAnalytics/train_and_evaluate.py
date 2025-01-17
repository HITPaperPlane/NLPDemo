#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 BERT 同时完成以下两项任务：
1. 情感识别（三分类，single-label）
2. 主题识别（多标签，multi-label）

并在训练后进行评估，输出 accuracy, precision, recall, F1，同时绘制 loss 和 accuracy 曲线。
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from typing import List, Tuple

# ------------- 全局配置 -------------
NUM_EPOCHS = 3
BATCH_SIZE = 2
MAX_SEQ_LEN = 128
LR = 1e-5
WARMUP_RATIO = 0.1
MODEL_NAME = "bert-base-chinese"

# 主题列表（固定顺序，用于 multi-label one-hot 编码）
TOPICS = ["动力", "价格", "内饰", "配置", "安全性", "外观", "操控", "油耗", "空间", "舒适性"]
TOPIC2IDX = {t: i for i, t in enumerate(TOPICS)}

# 情感 -1, 0, 1 对应到分类索引 { -1 -> 0, 0 -> 1, 1 -> 2 }
SENTIMENT_MAP = {-1: 0, 0: 1, 1: 2}


# =============== 数据处理部分 ===============
class AutoReviewDataset(Dataset):
    """
    自定义数据集，用于同时处理 (文本 -> BERT 输入, topic 多标签, sentiment 单标签)
    """

    def __init__(self, file_path: str, tokenizer: BertTokenizer, max_len: int = 128):
        """
        :param file_path: 训练或测试数据路径
        :param tokenizer: BERT tokenizer
        :param max_len: 最大序列长度
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._load_data(file_path)

    def _load_data(self, file_path: str):
        """
        读取并解析数据文件
        每一行示例格式:  文本 \t 主题1#情感 ... \t 主题n#情感
        """
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                text = parts[0]
                topic_sents = parts[1:]  # 之后的每一个部分都是 "主题#情感值"

                # 1) 统计该行所有 topic & sentiment
                topic_labels = [0] * len(TOPICS)  # 多标签
                sentiment_sum = 0                # 用于最后综合情感
                for ts in topic_sents:
                    if "#" not in ts:
                        continue
                    t, s = ts.split("#")
                    s = int(s)
                    sentiment_sum += s
                    if t in TOPIC2IDX:
                        topic_labels[TOPIC2IDX[t]] = 1  # 标记该主题出现

                # 2) 综合情感：sum>0 => 1, sum<0 => -1, sum=0 => 0
                if sentiment_sum > 0:
                    overall_sentiment = 1
                elif sentiment_sum < 0:
                    overall_sentiment = -1
                else:
                    overall_sentiment = 0

                # 3) 转换为 0/1/2 三分类索引
                sentiment_label = SENTIMENT_MAP[overall_sentiment]

                self.samples.append((text, topic_labels, sentiment_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        text, topic_labels, sentiment_label = self.samples[idx]

        # 使用 tokenizer 将文本编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)       # (max_len,)
        attention_mask = encoding["attention_mask"].squeeze(0) # (max_len,)
        token_type_ids = encoding["token_type_ids"].squeeze(0) # (max_len,)

        topic_labels = torch.tensor(topic_labels, dtype=torch.float)
        sentiment_label = torch.tensor(sentiment_label, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "topic_labels": topic_labels,
            "sentiment_label": sentiment_label
        }


# =============== 模型定义部分 ===============
class MultiTaskBERT(nn.Module):
    """
    BERT 多任务模型:
    1) 主题识别 (多标签, 10 维输出, 使用 BCEWithLogitsLoss)
    2) 情感识别 (三分类, 使用 CrossEntropyLoss)
    """
    def __init__(self, model_name: str, num_topics: int = 10, num_sentiments: int = 3):
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        # 主题分类器 (多标签)
        self.topic_classifier = nn.Linear(self.bert.config.hidden_size, num_topics)
        # 情感分类器 (三分类)
        self.sentiment_classifier = nn.Linear(self.bert.config.hidden_size, num_sentiments)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids=None,
                topic_labels=None,
                sentiment_label=None):
        # BERT 编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # pooled_output 即 CLS 向量
        pooled_output = outputs[1]  # 或 outputs.pooler_output

        # dropout
        pooled_output = self.dropout(pooled_output)

        # 主题 logits (多标签)
        topic_logits = self.topic_classifier(pooled_output)
        # 情感 logits (三分类)
        sentiment_logits = self.sentiment_classifier(pooled_output)

        loss = None
        if topic_labels is not None and sentiment_label is not None:
            # 多标签损失: BCEWithLogitsLoss
            topic_loss_fn = nn.BCEWithLogitsLoss()
            loss_topic = topic_loss_fn(topic_logits, topic_labels)

            # 情感损失: CrossEntropyLoss
            sentiment_loss_fn = nn.CrossEntropyLoss()
            loss_sentiment = sentiment_loss_fn(sentiment_logits, sentiment_label)

            # 总损失 = 二者相加
            loss = loss_topic + loss_sentiment

        return (loss, topic_logits, sentiment_logits)


# =============== 评估指标部分 ===============
def compute_metrics_topic(preds: np.ndarray, labels: np.ndarray):
    """
    计算多标签分类的 macro 平均 accuracy, precision, recall, F1
    :param preds: (N, num_topics), 0/1 预测
    :param labels: (N, num_topics), 0/1 真值
    """
    # micro/macro 计算方法很多，这里我们按每个主题分别算，再取平均
    eps = 1e-10
    num_topics = labels.shape[1]

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(num_topics):
        # 针对第 i 个主题进行二分类评估
        true_labels_i = labels[:, i]
        pred_labels_i = preds[:, i]

        tp = np.sum((pred_labels_i == 1) & (true_labels_i == 1))
        tn = np.sum((pred_labels_i == 0) & (true_labels_i == 0))
        fp = np.sum((pred_labels_i == 1) & (true_labels_i == 0))
        fn = np.sum((pred_labels_i == 0) & (true_labels_i == 1))

        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # macro 平均
    accuracy_macro = np.mean(accuracy_list)
    precision_macro = np.mean(precision_list)
    recall_macro = np.mean(recall_list)
    f1_macro = np.mean(f1_list)

    return accuracy_macro, precision_macro, recall_macro, f1_macro


def compute_metrics_sentiment(preds: np.ndarray, labels: np.ndarray):
    """
    计算情感单标签三分类的 accuracy, precision, recall, F1
    这里可以做 macro 或者 micro，通常做 macro 时，需要对每个类别分别算
    """
    eps = 1e-10

    # preds, labels 的取值范围 {0,1,2} (对应 -1,0,1)
    num_classes = 3

    # 整体 accuracy
    accuracy = np.sum(preds == labels) / len(labels)

    # 分类别计算 precision, recall, f1
    precision_list = []
    recall_list = []
    f1_list = []
    for c in range(num_classes):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    precision_macro = np.mean(precision_list)
    recall_macro = np.mean(recall_list)
    f1_macro = np.mean(f1_list)

    return accuracy, precision_macro, recall_macro, f1_macro


# =============== 训练测试循环 ===============
def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        topic_labels = batch["topic_labels"].to(device)
        sentiment_label = batch["sentiment_label"].to(device)

        loss, _, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            topic_labels=topic_labels,
            sentiment_label=sentiment_label
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    all_topic_preds = []
    all_topic_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            topic_labels = batch["topic_labels"].to(device)
            sentiment_label = batch["sentiment_label"].to(device)

            loss, topic_logits, sentiment_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                topic_labels=topic_labels,
                sentiment_label=sentiment_label
            )
            total_loss += loss.item()

            # -------- 主题预测 (multi-label) --------
            # 对 topic_logits 做 sigmoid，再以 0.5 为阈值转换为 0/1
            topic_probs = torch.sigmoid(topic_logits)  # (batch, num_topics)
            topic_pred = (topic_probs > 0.5).long().cpu().numpy()  # (batch, num_topics)
            topic_true = topic_labels.cpu().numpy()                 # (batch, num_topics)

            all_topic_preds.append(topic_pred)
            all_topic_labels.append(topic_true)

            # -------- 情感预测 (三分类) --------
            # 在 dim=-1 做 softmax 后 argmax，或直接对 logits 做 argmax
            sentiment_pred = torch.argmax(sentiment_logits, dim=-1).cpu().numpy()  # (batch,)
            sentiment_true = sentiment_label.cpu().numpy()                         # (batch,)

            all_sentiment_preds.append(sentiment_pred)
            all_sentiment_labels.append(sentiment_true)

    avg_loss = total_loss / len(dataloader)

    # 拼接
    all_topic_preds = np.concatenate(all_topic_preds, axis=0)
    all_topic_labels = np.concatenate(all_topic_labels, axis=0)
    all_sentiment_preds = np.concatenate(all_sentiment_preds, axis=0)
    all_sentiment_labels = np.concatenate(all_sentiment_labels, axis=0)

    # 计算多标签主题分类的指标 (macro)
    topic_acc, topic_prec, topic_rec, topic_f1 = compute_metrics_topic(all_topic_preds, all_topic_labels)

    # 计算情感分类指标
    sent_acc, sent_prec, sent_rec, sent_f1 = compute_metrics_sentiment(all_sentiment_preds, all_sentiment_labels)

    return (avg_loss,
            topic_acc, topic_prec, topic_rec, topic_f1,
            sent_acc, sent_prec, sent_rec, sent_f1)


def main():
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. 准备 tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 2. 构建 Dataset
    train_data_path = os.path.join("data", "train.txt")
    test_data_path = os.path.join("data", "test.txt")

    train_dataset = AutoReviewDataset(train_data_path, tokenizer, max_len=MAX_SEQ_LEN)
    test_dataset = AutoReviewDataset(test_data_path, tokenizer, max_len=MAX_SEQ_LEN)

    # 3. 进一步划分 train/val 
    # 9:1 划分
    total_len = len(train_dataset)
    val_len = int(0.1 * total_len)
    train_len = total_len - val_len

    train_ds, val_ds = random_split(train_dataset, [train_len, val_len])

    # 4. 构建 DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 构建模型
    model = MultiTaskBERT(MODEL_NAME, num_topics=len(TOPICS), num_sentiments=3).to(device)

    # 6. 优化器 & 学习率调度
    optimizer = AdamW(model.parameters(), lr=LR)
    # 训练总步数
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 7. 开始训练
    best_val_f1 = 0.0

    # 用于可视化记录
    train_loss_list = []
    val_loss_list = []
    train_acc_list_sent = []
    val_acc_list_sent = []

    for epoch in range(NUM_EPOCHS):
        print(f"===== Epoch {epoch+1} / {NUM_EPOCHS} =====")

        # --- 训练 ---
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        # 训练集评估(可选，一般直接算train_loss即可，这里演示完整流程)
        (train_eval_loss,
         train_topic_acc, train_topic_prec, train_topic_rec, train_topic_f1,
         train_sent_acc, train_sent_prec, train_sent_rec, train_sent_f1) = evaluate(model, train_loader, device)

        # --- 验证 ---
        (val_eval_loss,
         val_topic_acc, val_topic_prec, val_topic_rec, val_topic_f1,
         val_sent_acc, val_sent_prec, val_sent_rec, val_sent_f1) = evaluate(model, val_loader, device)

        # 记录用于可视化
        train_loss_list.append(train_eval_loss)
        val_loss_list.append(val_eval_loss)
        train_acc_list_sent.append(train_sent_acc)
        val_acc_list_sent.append(val_sent_acc)

        print(f"[Train] loss: {train_loss:.4f}, "
              f"Topic-F1(macro): {train_topic_f1:.4f}, Sentiment-F1: {train_sent_f1:.4f}, "
              f"Sent-Acc: {train_sent_acc:.4f}")
        print(f"[Val]   loss: {val_eval_loss:.4f}, "
              f"Topic-F1(macro): {val_topic_f1:.4f}, Sentiment-F1: {val_sent_f1:.4f}, "
              f"Sent-Acc: {val_sent_acc:.4f}")

        # 如果在验证集上有更高的 F1，则保存模型
        if val_sent_f1 > best_val_f1:
            best_val_f1 = val_sent_f1
            torch.save(model.state_dict(), "best_model.pt")
            print(">>>> Best model saved.")

    # 加载最优模型
    if os.path.exists("best_model.pt"):
        model.load_state_dict(torch.load("best_model.pt", map_location=device))

    # 8. 在测试集上评估
    (test_loss,
     test_topic_acc, test_topic_prec, test_topic_rec, test_topic_f1,
     test_sent_acc, test_sent_prec, test_sent_rec, test_sent_f1) = evaluate(model, test_loader, device)

    print("\n===== Test Results =====")
    print(f"Topic - Acc(macro): {test_topic_acc:.4f}, Prec: {test_topic_prec:.4f}, "
          f"Rec: {test_topic_rec:.4f}, F1: {test_topic_f1:.4f}")
    print(f"Sent  - Acc: {test_sent_acc:.4f}, Prec(macro): {test_sent_prec:.4f}, "
          f"Rec(macro): {test_sent_rec:.4f}, F1(macro): {test_sent_f1:.4f}")
    print("========================")

    # 9. 绘制训练集和验证集的 loss & accuracy 曲线
    epochs_range = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(12, 5))
    # -- loss 曲线 --
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss_list, label='Train Loss')
    plt.plot(epochs_range, val_loss_list, label='Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # -- accuracy 曲线 (这里以情感分类的 accuracy 为例) --
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc_list_sent, label='Train Sentiment Acc')
    plt.plot(epochs_range, val_acc_list_sent, label='Val Sentiment Acc')
    plt.title('Sentiment Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
