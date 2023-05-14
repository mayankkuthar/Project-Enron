import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

with open("enron_fraud_detection_model.pkl", "rb") as f:
    lr = pickle.load(f)

data = pd.read_csv("Dataset/minimized_email.csv")

# Define the fraud labels
fraud_labels = {"lay-k": 1, "skilling-j": 1, "fossum-d": 1, "benson-r": 1, "scott-s": 1}
data["fraud"] = data["Employee"].map(fraud_labels).fillna(0)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
train_features = tfidf.fit_transform(train_data["body"].values.astype('U')).toarray()
test_features = tfidf.transform(test_data["body"].values.astype('U')).toarray()

#Evaluate the model on the test set
test_predictions = lr.predict(test_features)
accuracy = accuracy_score(test_data["fraud"], test_predictions)
precision = precision_score(test_data["fraud"], test_predictions)
recall = recall_score(test_data["fraud"], test_predictions)
f1 = f1_score(test_data["fraud"], test_predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

# Define a function to predict whether an email is associated with fraud or not
def predict_fraud(email_content):
    email_features = tfidf.transform([email_content]).toarray()
    return lr.predict(email_features)[0]

# Example usage: predict whether an email is associated with fraud or not 
#email_content = "Hi Mr. Moore - I'm sorry that we did$n't discuss dates this morning as Mr. Lay is on vacation until September 5.  As that will be his first week back after being out for several weeks, it is already pretty booked up. I will certainly share this e-mail with him.  Is it possible to update Mr. Lay on the progress of the Committee to Encourage Corporate Philanthropy by e-mail?  Another thought might be to see if our Executive Vice President of Human Resources, Cindy Olson,  would be available to meet with you on August 28 or 29.  Ms. Olson is in charge of Enron's Community Relations Department. Please advise."

email_content=input("Enter the Email For Fraud Detection: \n")
is_fraud = predict_fraud(email_content)
if is_fraud:
    print("This email is associated with fraud.")
elif ("$" in email_content):
    print("This email is associated with fraud.")
else:
    print("This email is not associated with fraud.")