# Full baseline-only evaluation script aligned to old 10-model pipeline
# 70/15/15 split + threshold tuning + plots

import os, re, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, hamming_loss,
    precision_recall_curve
)

from empath import Empath
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CSV_PATH = '../data/preprocessed/mentalmanip_maj.csv'
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

lexicon = Empath()
analyzer = SentimentIntensityAnalyzer()

# ---------------- DATA ----------------
df = pd.read_csv(CSV_PATH)
X_text = df['Dialogue'].astype(str).to_numpy()
labels = df['Technique'].fillna('').apply(lambda s:[x.strip() for x in str(s).split(',') if x.strip()]).tolist()

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)
classes = mlb.classes_

# ---------------- HELPERS ----------------
def split_speakers(text):
    pattern = r'(Person1|Person2):\s*(.*?)(?=(Person1|Person2):|$)'
    matches = re.findall(pattern, str(text), flags=re.DOTALL)
    p1, p2 = [], []
    for who, msg, _ in matches:
        if who == 'Person1': p1.append(msg.strip())
        else: p2.append(msg.strip())
    return ' '.join(p1), ' '.join(p2)

def extract_empath(text):
    vals = lexicon.analyze(text, normalize=True)
    return np.array(list(vals.values()), dtype=float)

def empath_pair(p1,p2):
    return np.concatenate([extract_empath(p1), extract_empath(p2)])

def vader(text):
    s = analyzer.polarity_scores(text)
    return np.array([s['pos'], s['neu'], s['neg'], s['compound']], dtype=float)

def vader_pair(p1,p2):
    s1 = vader(p1); s2 = vader(p2)
    return np.concatenate([s1,s2,s1-s2])

def count_word(text, word):
    return len(re.findall(rf'\\b{word}\\b', text.lower()))

def count_intensifiers(text):
    return sum(text.lower().count(w) for w in ['very','really','always','never','absolutely'])

def manual_features(p1,p2):
    you1, you2 = count_word(p1,'you'), count_word(p2,'you')
    i1, i2 = count_word(p1,'i'), count_word(p2,'i')
    l1, l2 = max(len(p1.split()),1), max(len(p2.split()),1)
    yr1, yr2 = you1/l1, you2/l2
    ir1, ir2 = i1/l1, i2/l2
    return np.array([
        yr1, yr2, ir1, ir2,
        yr1-yr2,
        p1.count('?'), p2.count('?'),
        yr1-ir1, yr2-ir2,
        count_intensifiers(p1), count_intensifiers(p2),
        abs(l1-l2)
    ], dtype=float)

def build_features(texts, ng):
    rows = []
    for i, txt in enumerate(texts):
        p1,p2 = split_speakers(txt)
        dense = np.concatenate([
            manual_features(p1,p2),
            empath_pair(p1,p2),
            vader_pair(p1,p2)
        ]).reshape(1,-1)
        rows.append(hstack([csr_matrix(dense), ng[i]]))
    return vstack(rows)

# ---------------- SPLIT ----------------
X_train, X_temp, y_train, y_temp = train_test_split(X_text, Y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# ---------------- NGRAMS ----------------
vec = CountVectorizer(max_features=5000, ngram_range=(1,3), binary=True, stop_words='english')
Xtr_ng = vec.fit_transform(X_train)
Xva_ng = vec.transform(X_val)
Xte_ng = vec.transform(X_test)

Xtr = build_features(X_train, Xtr_ng)
Xva = build_features(X_val, Xva_ng)
Xte = build_features(X_test, Xte_ng)

# ---------------- MODEL ----------------
model = OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight='balanced'))
model.fit(Xtr, y_train)

val_prob = model.predict_proba(Xva)
ths = []
for k in range(y_train.shape[1]):
    if y_val[:,k].sum() == 0:
        ths.append(0.5)
        continue
    p,r,t = precision_recall_curve(y_val[:,k], val_prob[:,k])
    f = 2*p*r/(p+r+1e-9)
    ths.append(t[np.argmax(f[:-1])])
ths = np.array(ths)

test_prob = model.predict_proba(Xte)
y_pred = (test_prob > ths).astype(int)

# ---------------- METRICS ----------------
label_acc = 1 - hamming_loss(y_test, y_pred)
subset_acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)

print('Accuracy (1-HL):', round(label_acc,4))
print('Subset Accuracy:', round(subset_acc,4))
print('F1 Macro:', round(f1_macro,4))
print('F1 Micro:', round(f1_micro,4))

# ---------------- PLOTS ----------------
f1_per = f1_score(y_test, y_pred, average=None, zero_division=0)
acc_per = np.array([1-hamming_loss(y_test[:,i], y_pred[:,i]) for i in range(len(classes))])
prec_per = precision_score(y_test, y_pred, average=None, zero_division=0)
rec_per = recall_score(y_test, y_pred, average=None, zero_division=0)

plt.figure(figsize=(12,6))
plt.bar(classes, f1_per)
plt.xticks(rotation=45, ha='right')
plt.title('F1 Score per Class')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/plot1_f1_per_class.png', dpi=300)

plt.figure(figsize=(12,6))
plt.bar(classes, acc_per)
plt.axhline(acc_per.mean(), linestyle='--')
plt.xticks(rotation=45, ha='right')
plt.title('Accuracy per Class (1-HL)')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/plot2_accuracy_per_class.png', dpi=300)

plt.figure(figsize=(8,5))
plt.bar(['Accuracy','F1 Macro','F1 Micro'], [label_acc, f1_macro, f1_micro])
plt.title('Overall Metrics')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/plot3_overall_metrics.png', dpi=300)

plt.figure(figsize=(12,6))
plt.plot(classes, prec_per, marker='o', label='Precision')
plt.plot(classes, rec_per, marker='o', label='Recall')
plt.xticks(rotation=45, ha='right')
plt.title('Precision vs Recall per Class')
plt.legend()
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/plot4_precision_recall.png', dpi=300)

print('Saved plots to', OUTPUT_DIR)