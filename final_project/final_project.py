import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
import re

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)  # 移除 URL
    text = re.sub(r'@\w+|#\w+', '', text)       # 移除 @ 或 # 開頭的標籤
    text = re.sub(r'[^\w\s]', '', text)         # 移除標點符號和特殊字符
    text = re.sub(r'[^\x00-\x7F]+', '', text)   # 移除表情符號
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()     # 移除多餘空格
    return text

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df['cleaned_text'] = train_df['text'].apply(clean_text)
test_df['cleaned_text'] = test_df['text'].apply(clean_text)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_df['token_length'] = train_df['cleaned_text'].apply(lambda x: len(tokenizer.tokenize(x)))
test_df['token_length'] = test_df['cleaned_text'].apply(lambda x: len(tokenizer.tokenize(x)))

class DisasterTweetsDataset(Dataset):
    def __init__(self, df, tokenizer, has_label=True):
        self.df = df
        self.texts = df["cleaned_text"].values.tolist()

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

train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=seed)

encoded_train_dataset = DisasterTweetsDataset(train_df, tokenizer)
encoded_valid_dataset = DisasterTweetsDataset(valid_df, tokenizer)
encoded_test_dataset = DisasterTweetsDataset(test_df, tokenizer, has_label=False)

training_args = TrainingArguments(
    output_dir="output",               
    per_device_train_batch_size=48,    
    per_device_eval_batch_size=48,     
    learning_rate=5e-5,                
    lr_scheduler_type="linear",        
    warmup_ratio=0.1,                  
    num_train_epochs=2,                
    save_strategy="epoch",             
    logging_strategy="epoch",          
    eval_strategy="epoch",             
    load_best_model_at_end=True,       
    metric_for_best_model="f1",        
    fp16=True,                         
    report_to="none",                  
    seed=seed,                          
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 載入預訓練的 BERT 模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # 二分類問題
)

def compute_metrics(eval_preds):
    y_true = eval_preds.label_ids
    y_pred = np.argmax(eval_preds.predictions, axis=-1)
    metrics = {
        "f1": f1_score(y_true=y_true, y_pred=y_pred),
    }
    return metrics

trainer = Trainer(
    model=model,                         
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_valid_dataset,
    data_collator=data_collator, 
    args=training_args,
    compute_metrics=compute_metrics,
)

trainer.train()

valid_loader = DataLoader(encoded_valid_dataset, batch_size=32)

all_features = []
all_labels = []
model.eval()
with torch.no_grad():
    for batch in valid_loader:
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
        outputs = model(**inputs)
        
        features = outputs.logits.cpu().numpy()
        all_features.append(features)
        
        if "labels" in batch:
            all_labels.append(batch["labels"].cpu().numpy())

if all_features:
    all_features = np.concatenate(all_features, axis=0)
else:
    raise ValueError("No features extracted from validation dataset.")

if all_labels:
    all_labels = np.concatenate(all_labels, axis=0)
else:
    raise ValueError("No labels found in validation dataset.")

svm_model = SVC(kernel='linear')
svm_model.fit(all_features, all_labels)

y_pred_svm = svm_model.predict(all_features)
print("SVM F1 Score:", f1_score(all_labels, y_pred_svm))

# 使用 BERT 提取測試集特徵並用 SVM 進行預測
test_loader = DataLoader(encoded_test_dataset, batch_size=32)

all_test_features = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        # 確保數據在正確的設備上
        inputs = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**inputs)
        
        # 提取 logits 作為特徵
        features = outputs.logits.cpu().numpy()
        all_test_features.append(features)

# 合併測試集特徵
if all_test_features:
    all_test_features = np.concatenate(all_test_features, axis=0)
else:
    raise ValueError("No features extracted from test dataset.")

svm_predictions = svm_model.predict(all_test_features)
print("SVM Test Predictions:", svm_predictions)

df_test = pd.read_csv("sample_submission.csv")
df_test["target"] = svm_predictions
df_test.to_csv("submission.csv", index=None)
