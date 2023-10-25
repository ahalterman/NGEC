from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.svm import SVC
import numpy as np
import skops.io as sio
import os

# path to sentence-transformer model directory if you've downloaded it
# Leave blank to download from huggingface on the fly
MODEL_DIR = "" 
#change device to 'cuda' if you have a GPU enabled
DEVICE = "cpu"

synth_df = pd.read_csv("gpt_synthetic_events_2023-10-19_19.csv.zip", 
                       compression="zip")


def load_model(model_name="paraphrase-mpnet-base-v2"):
    if MODEL_DIR:
        # use the local copy
        model = SentenceTransformer(os.path.join(MODEL_DIR, model_name))
    else:
        # otherwise, download from huggingface
        model = SentenceTransformer(f'sentence-transformers/{model_name}')
    return model

model = load_model()
encoded = model.encode(synth_df['text'].values, show_progress_bar=True,
                               device=DEVICE).tolist()
synth_df['encoded'] = encoded


def fit_initial_model(synth_df):
    clf = SVC(class_weight="balanced",
                kernel="linear",
                probability=True,
                C=0.1)
    y_train = synth_df['label']
    clf.fit(synth_df['encoded'].to_list(), y_train)
    pred = pd.DataFrame(clf.predict_proba(encoded))
    # rename columns with the event names
    pred.columns = clf.classes_
    return pred

pred = fit_initial_model(synth_df)
synth_df = pd.concat([synth_df, pred], axis=1)
event_types = synth_df['label'].unique()

for event in event_types:
    print(event)
    # First, sample the positive cases (assuming the prompts are reliable)
    train_pos_synth = synth_df[synth_df['label'] == event].copy()
    train_pos_synth['label'] = 1
    # Now sample negative cases, but don't pick anything that might
    # be a positive case
    candidate_neg = synth_df[synth_df[event] < 0.05].copy()
    # Take 3x as many negative cases as positive cases, or
    # as many as we can find
    sample_size = min(candidate_neg.shape[0], 
                      train_pos_synth.shape[0] * 3)
    train_neg_synth = candidate_neg.sample(sample_size).copy()
    train_neg_synth['label'] = 0
    # Now combine the positive and negative cases
    print(train_pos_synth.shape, train_neg_synth.shape)
    train = pd.concat([train_pos_synth, train_neg_synth], axis=0)
    X_train = np.array(train['encoded'].tolist())
    y_train = train['label']
    clf = SVC(class_weight="balanced",
                kernel="linear",
                probability=True)
    clf.fit(X_train, y_train)

    sio.dump(clf, f"models/{event}.skops")
    

## For production use, see https://github.com/ahalterman/NGEC/blob/main/NGEC/event_class.py




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