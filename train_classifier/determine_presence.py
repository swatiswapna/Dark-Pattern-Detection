import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from joblib import dump
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


df1 = pd.read_csv('normie.csv')
df2 = pd.read_csv('dark_patterns.csv')

df1 = df1[pd.notnull(df1["Pattern String"])]
df1 = df1[df1["classification"] == 0]
df1["classification"] = "Not Dark"
df1.drop_duplicates(subset="Pattern String", inplace=True)

df2 = df2[pd.notnull(df2["Pattern String"])]
df2["classification"] = "Dark"
col = ["Pattern String", "classification"]
df2 = df2[col]

df = pd.concat([df1, df2])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['Pattern String'], df["classification"], train_size=0.25, random_state=42)

# Vectorization and modeling
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = BernoulliNB().fit(X_train_tfidf, y_train)

# Prediction
y_pred = clf.predict(count_vect.transform(X_test))

# Accuracy score
accuracy = metrics.accuracy_score(y_pred, y_test)
print("Accuracy: ", accuracy)
# categorical 
category_id_df = pd.DataFrame(df["classification"].unique(), columns=["classification"])
category_id_df["category_id"] = range(len(category_id_df))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df["classification"].values,
            yticklabels=category_id_df["classification"].values,
            cmap="Blues")
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Precision-Recall Curve
from sklearn.metrics import precision_recall_curve, auc

precision, recall, _ = precision_recall_curve(y_test, clf.predict_proba(count_vect.transform(X_test))[:, 1], pos_label="Dark")
area_under_curve = auc(recall, precision)

plt.figure(figsize=(8, 8))
plt.plot(recall, precision, color='darkorange', lw=2, label='Area under curve = %0.2f' % area_under_curve)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.show()

# Receiver Operating Characteristic (ROC) Curve
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(count_vect.transform(X_test))[:, 1], pos_label="Dark")
roc_auc = roc_auc_score(y_test, clf.predict_proba(count_vect.transform(X_test))[:, 1])

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Area under curve = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# # Save the model and vectorizer
# dump(clf, 'presence_classifier.joblib')
# dump(count_vect, 'presence_vectorizer.joblib')