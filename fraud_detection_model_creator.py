import os
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the preprocessed dataset
data = pd.read_csv("Dataset/minimized_email.csv")

# cost list of employee
list1 = ['arnold-j','arora-h','badeer-r','bailey-s','bass-e','baughman-d','baughman-e','beck-s','benson-r','blair-l','brawner-s','buy-r','campbell-l','carson-m','cash-m','causholli-m','corman-s','crandall-s','crandell-s','cuilla-m','dasovich-j','davis-d','dean-c','delainey-d','derrick-j','dickson-s','donoho-l','donohoe-t','dorland-c','ermis-f','farmer-d','fischer-m','forney-j','fossum-d','gang-l','gay-r','geaccone-t','germany-c','gilbertsmith-d','giron-d','griffith-j','grigsby-m','guzman-m','haedicke-m','hain-m','harris-s','hayslett-r','heard-m','hendrickson-s','hernandez-j','hodge-j','holst-k','horton-s','hyatt-k','hyvl-d','jones-t','kaminski-v','kean-s','keavey-p','keiser-k','king-j','kitchen-l','kuykendall-t','lavorado-j','lavorato-j','lay-k','lenhart-m','lewis-a','linder-e','lokay-m','lokey-t','love-p','lucci-p','luchi-p','maggi-m','mann-k','martin-t','may-l','mccarty-d','mcconnell-m','mckay-b','mckay-j','mclaughlin-e','merriss-s','meyers-a','mims-p','mims-thurston-p','motley-m','neal-s','nemec-g','panus-s','parks-j','pereira-s','perlingiere-d','phanis-s','pimenov-v','platter-p','presto-k','quenet-j','quigley-d','rapp-b','reitmeyer-j','richey-c','ring-a','ring-r','rodrigue-r','rodrique-r','rogers-b','ruscitti-k','sager-e','saibi-e','salisbury-h','sanchez-m','sanders-r','scholtes-d','schoolcraft-d','schwieger-j','scott-s','semperger-c','shackleton-s','shankman-j','shapiro-r','shively-h','skilling-j','slinger-r','smith-m','solberg-g','south-s','staab-t','stclair-c','steffes-j','stepenovitch-j','stokley-c','storey-g','sturm-f','swerzbin-m','symes-k','taylor-m','tholt-j','thomas-p','townsend-j','tycholiz-b','ward-k','watson-k','weldon-c','weldon-v','whalley-g','whalley-l','wheldon-c','white-s','whitt-m','williams-b','williams-j','williams-w3','wolfe-j','ybarbo-p','zipper-a','zufferli-j','zufferlie-j']

print("Data Loaded !!!\n")

# Define the fraud labels
fraud_labels = {"lay-k": 1, "skilling-j": 1, "fossum-d": 1, "benson-r": 1, "scott-s": 1}
data["fraud"] = data["Employee"].map(fraud_labels).fillna(0)
print("Added fraud Section !!!\n")

# Split the dataset into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print("Data Splited !!!\n")

# Extract features from the email content using TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
train_features = tfidf.fit_transform(train_data["body"].values.astype('U')).toarray()
print("Train data Transformed !!!\n")
test_features = tfidf.transform(test_data["body"].values.astype('U')).toarray()
print("Test data Transformed !!!\n")

# Train a logistic regression model
lr = LogisticRegression(random_state=42, max_iter=100, solver="lbfgs")
lr.fit(train_features, train_data["fraud"])

# Save the trained model as a pickle file
with open("enron_fraud_detection_model.pkl", "wb") as f:
    print("Saving ",f)
    pickle.dump(lr, f)