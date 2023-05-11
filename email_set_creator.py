import email
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
import parser

# read the dataset
df = pd.read_csv("Dataset/emails.csv")  # put original CSV File

# display first 5 rows of the dataset using head function
df.head()

# transform the email into correct form
message = df.loc[1]["message"]
emails = email.message_from_string(message)
emails.items()

# feature we are extract from the email
# 1. date 2. X-From 3. X-To 4. Subject 5. X-Folder
# extract feature from the email for all datasets
def extract_data(feature, df):
    column = []
    for row in df:
        e = email.message_from_string(row)
        column.append(e.get(feature))
    return column


df["Date"] = extract_data("Date", df["message"])
df["Subject"] = extract_data("Subject", df["message"])
df["X-From"] = extract_data("X-From", df["message"])
df["X-To"] = extract_data("X-To", df["message"])
df["X-Folder"] = extract_data("X-Folder", df["message"])

# extract email body from email message
def get_email_body(data):
    column = []
    for msg in data:
        e = email.message_from_string(msg)
        column.append(e.get_payload())
    return column


df["body"] = get_email_body(df["message"])

df.head()

# Employee names


def emp_name(data):
    column = []
    for msg in data:
        column.append(msg.split("/")[0])
    return column


df["Employee"] = emp_name(df["file"])

unique_emails = pd.DataFrame(df["X-Folder"].value_counts())
unique_emails.reset_index(inplace=True)

# show top 20 folder highest counts
unique_emails.columns = ["Folder_name", "Count"]
unique_emails.iloc[:20, :]

# visualize top 20 folder name
plt.figure(figsize=(10, 6))
sns.barplot(x="Count", y="Folder_name", data=unique_emails.iloc[:20, :])
plt.title("Top 20 Folder")
plt.xlabel("Count")
plt.ylabel("Folder name ")
plt.show()

# top email sender employees
emp_data = pd.DataFrame(df["Employee"].value_counts())
emp_data.reset_index(inplace=True)

emp_data.columns = ["Employee Name", "Count"]
emp_data.iloc[:20, :]

# visualize top 20 emails sender employee
plt.figure(figsize=(10, 6))
sns.barplot(x="Count", y="Employee Name", data=emp_data.iloc[:20, :])
plt.title("Top 20 Emails Sender Emails")
plt.xlabel("Count")

plt.ylabel("Emplyree Name")
plt.show()

# Date columns
from datetime import datetime

from dateutil import parser

# this is sample example
x = parser.parse("Fri, 4 May 2001 13:51:00 -0700 (PDT)")
print(x.strftime("%d-%m-%Y %H:%M:%S"))


def change_date_type(data):
    column = []
    for date in data:
        column.append(parser.parse(date).strftime("%d-%m-%Y %H:%M:%S"))
    return column


df["Date"] = change_date_type(df["Date"])

x_value = df.loc[1, "X-Folder"]
# extract last folder name
folder_name = x_value.split("\\")[-1]


def process_folder_name(folders):
    column = []
    for folder in folders:
        if folder is None or folder == "":
            column.append(np.nan)
        else:
            column.append(folder.split("\\")[-1].lower())
    return column


df["X-Folder"] = process_folder_name(df["X-Folder"])

# found unique folder
print("Lenghts of unique folder : ", len(df["X-Folder"].unique()))
# fetch some folder name
df["X-Folder"].unique()[:20]

# replace empty missing value in subject with np.nan
def replace_empty_with_nan(subject):
    column = []
    for sub in subject:
        if sub == "":
            column.append(np.nan)
        else:
            column.append(sub)
    return column


df["Subject"] = replace_empty_with_nan(df["Subject"])
df["X-To"] = replace_empty_with_nan(df["X-To"])

# check missing value in the dataset
df.isnull().sum()

# calculate the missing value percentage
missing_value = df.isnull().sum()
miss = missing_value[missing_value > 0]
miss_percen = miss / df.shape[0]

miss_percen

# drop missing value rows from the dataset
df.dropna(axis=0, inplace=True)
df.isnull().sum()

# now need to drop some columns which is not necessary for the model
drop_column_names = ["file", "message", "Date", "X-From", "X-To"]
df.drop(columns=drop_column_names, axis=1, inplace=True)

df.to_csv("cleaned_dataset.csv", index=False)
