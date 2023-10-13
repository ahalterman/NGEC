from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import jsonlines
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report
import skops.io as sio

MODEL_DIR = "" # path to sentence-transformer model directory if you've downloaded it
# Leave blank to download from huggingface on the fly

with jsonlines.open("gpt_synthetic_events_2023-04-06.jsonl", "r") as f:
    data = list(f.iter())

df = pd.DataFrame(data)
df['text'] = df['text'].str.replace("### THIS IS A SYNTHETIC STORY. DO NOT TRUST THE FACTUAL CONTENT OF THIS TEXT. Created by Andy Halterman to train a document-level political event classifer ###", "")


train, val = train_test_split(df, test_size=0.2, random_state=42)
text_train = train['text'].values
text_val = val['text'].values

y_train = train['event']
y_val = val['event']

if MODEL_DIR:
    # use the local copy
    model = SentenceTransformer(MODEL_DIR + 'huggingface_models/paraphrase-mpnet-base-v2')
else:
    # otherwise, download from huggingface
    model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

# encode the text
X_train = model.encode(text_train, show_progress_bar=True)
X_val = model.encode(text_val, show_progress_bar=True)

# this is our hacky way of doing multi-label classification
# We know ("know"--this is synthetically labeled data) the positive class for each event type, but
# a story can have multiple event types.
# This is obviously not ideal, but I got worse results using Cleanlab or positive unlabeled learning.

event_types = df['event'].unique()

for event in event_types:
    clf = LogisticRegression(class_weight="balanced")
    y_train_event = np.array(y_train == event).astype(int)
    y_val_event = np.array(y_val == event).astype(int)
    clf.fit(X_train, y_train_event)

    y_pred = clf.predict(X_val)
    print(classification_report(y_val_event, y_pred))
    sio.dump(clf, f"models/{event}.skops")
    


preds = []
for event, clf in classifiers.items():
    y_pred = clf.predict_proba(X_val)[:,1]
    preds.append(y_pred)

pred_array = np.array(preds).T
bin_pred = pred_array > 0.5
bin_pred[bin_pred == True]

# conver the matrix of binary predictions to a list of lists
preds = []
for i in bin_pred:
    preds.append([event_types[j] for j in np.where(i == True)[0]])





## A bunch of stuff that didn't really work

# convert single label y to multi-label y
#from sklearn.preprocessing import MultiLabelBinarizer
#mlb = MultiLabelBinarizer()
#y_train_bin = mlb.fit_transform([[i] for i in y_train])
#y_val_bin = mlb.fit_transform([[i] for i in y_val])
#
## train multi-label logistic regression model
#clf = RandomForestClassifier(class_weight="balanced")
#clf.fit(X_train, y_train_bin)
#y_pred = clf.predict_proba(X_val)
#print(classification_report(y_val_bin, y_pred))
#
#from cleanlab.classification import CleanLearning
#
#cl = CleanLearning(clf)
#cl.fit(X_train, y_train_bin)
#
### One-by-one classifiers
#from pulearn import WeightedElkanotoPuClassifier
#
#y_train = np.array(train['event'] == "ASSAULT").astype(int) * 2 - 1
#
#clf = LogisticRegression(C=0.4, class_weight="balanced")
#clf.fit(X_train, y_train)
#
#y_val = np.array(val['event'] == "ASSAULT").astype(int) * 2 - 1
#y_pred = clf.predict(X_val)
#print(classification_report(y_val, y_pred))
#
#
## Experimented with PU learning, but it didn't work well
#from pulearn import WeightedElkanotoPuClassifier
#pu_estimator = WeightedElkanotoPuClassifier(
#    estimator=clf, labeled=10, unlabeled=20, hold_out_ratio=0.2)
#pu_estimator.fit(X_train, y_train)
#
#y_pred = pu_estimator.predict(X_val)
#print(classification_report(y_val, y_pred))