import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.trainer_utils import EvalPrediction
from torch.utils.data import Dataset

# 設定隨機種子以確保可重複性
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 讀取資料
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 加載 tokenizer 和模型
model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 計算每個文本的 token 長度
train_df['token_length'] = train_df['text'].apply(lambda x: len(tokenizer.tokenize(x)))
test_df['token_length'] = test_df['text'].apply(lambda x: len(tokenizer.tokenize(x)))

# 自訂 Dataset 類別
class DisasterTweetsDataset(Dataset):
    def __init__(self, df, tokenizer, has_label=True):
        self.df = df
        self.texts = df["text"].values.tolist()
        
        # Tokenize 文本資料並生成 input_ids 等
        self.encoded_examples = tokenizer(
            text=self.texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        self.has_label = has_label
        if self.has_label:
            self.label_list = df["target"].values.tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encoded_examples["input_ids"][idx],
            "attention_mask": self.encoded_examples["attention_mask"][idx],
        }
        
        if self.has_label:
            item["labels"] = torch.tensor(self.label_list[idx])
        
        return item

# 訓練資料與驗證資料的劃分
train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=seed)

# 建立 Dataset
encoded_train_dataset = DisasterTweetsDataset(train_df, tokenizer)
encoded_valid_dataset = DisasterTweetsDataset(valid_df, tokenizer)
encoded_test_dataset = DisasterTweetsDataset(test_df, tokenizer, has_label=False)

# 定義訓練參數
training_args = TrainingArguments(
    output_dir="output",               # 儲存結果的目錄
    per_device_train_batch_size=36,    # 訓練時每個設備的批次大小
    per_device_eval_batch_size=36,     # 驗證時每個設備的批次大小
    learning_rate=1e-5,                # 學習率
    lr_scheduler_type="linear",        # 學習率調度方式
    warmup_ratio=0.1,                  # 預熱步驟的比例
    num_train_epochs=2,                # 訓練的回合數
    save_strategy="epoch",             # 每回合保存模型
    logging_strategy="epoch",          # 每回合記錄日誌
    eval_strategy="epoch",             # 每回合進行驗證
    load_best_model_at_end=True,       # 加載最佳的模型
    metric_for_best_model="f1",        # 使用 f1 分數來選擇最佳模型
    fp16=True,                         # 啟用混合精度訓練
    report_to="none",                  # 禁用對外部服務的報告
    seed=seed,                         # 設定隨機種子
)

# 定義資料 collator，用於 padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 載入預訓練的 DeBERTaV3 模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # 二分類問題
)

# 定義計算指標的函數 (F1 分數)
def compute_metrics(eval_preds: EvalPrediction) -> dict:
    y_true = eval_preds.label_ids
    y_pred = np.argmax(eval_preds.predictions, axis=-1)
    metrics = {
        "f1": f1_score(y_true=y_true, y_pred=y_pred),
    }
    return metrics

# 建立 Trainer 並開始訓練
trainer = Trainer(
    model=model,                         # 要訓練的模型
    train_dataset=encoded_train_dataset,  # 訓練資料
    eval_dataset=encoded_valid_dataset,   # 驗證資料
    data_collator=data_collator,         # 資料 collator
    args=training_args,                  # 訓練參數
    compute_metrics=compute_metrics,     # 計算指標的函數
)

# 開始訓練
trainer.train()

# 預測測試資料
predictions = trainer.predict(encoded_test_dataset)

# 取得 logits (預測的原始分數)
logits = predictions.predictions

# 依據最大 logits 來分配類別
predicted_class = np.argmax(logits, axis=1)
print(predicted_class)

# 載入範本提交檔案
df_sample = pd.read_csv("sample_submission.csv")

# 更新 'target' 欄位為預測的結果
df_sample["target"] = predicted_class

# 儲存預測結果到 CSV 檔案
df_sample.to_csv("submission.csv", index=None)
