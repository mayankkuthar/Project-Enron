import os
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Reading Preprocessed Data of Emails using Panadas

enron_email_data = pd.read_csv("Dataset\cleaned_dataset.csv")

# Define a function to preprocess the emails
def preprocess(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stop words and punctuation
    stop_words = set(stopwords.words("english"))
    words = [
        word.lower()
        for word in words
        if word.isalpha() and word.lower() not in stop_words
    ]

    return words


# Define a function to create a word cloud of the emails
def create_wordcloud(text):
    # Preprocess the text
    words = preprocess(text)

    # Create a word frequency counter
    word_counts = Counter(words)

    # Generate a word cloud
    wordcloud = WordCloud(
        width=800, height=800, background_color="white"
    ).generate_from_frequencies(word_counts)

    # Display the word cloud
    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


# Define a function to analyze the emails
def analyze_emails():
    # Loop through the emails
    count = 0
    """#Custodial Employee
    cust_emp=['allen-p','arnold-j','arora-h','badeer-r','bailey-s','bass-e','baughman-d','beck-s','benson-r','blair-l','brawner-s','buy-r','campbell-l','carson-m','cash-m','causholli-m','corman-s','crandell-s','cuilla-m','dasovich-j','davis-d','dean-c','delainey-d','derrick-j','dickson-s','donoho-l','donohoe-t','dorland-c','ermis-f','farmer-d','fischer-m','forney-j','fossum-d','gang-l','gay-r','geaccone-t','germany-c','gilbertsmith-d','giron-d','griffith-j','grigsby-m','guzman-m','haedicke-m','hain-m','harris-s','lay-k','pereira-s','quigley-d','rapp-b','ring-r','shackleton-s','shankman-j','smith-m','staab-t','williams-j','ybarbo-p']"""

    e_name = input("Enter the name of employee required for analysis:")
    for i in range(len(enron_email_data)):
        if count < 5 and enron_email_data.at[i, "Employee"] == e_name:
            print("Analysing Email For ", enron_email_data.at[i, "Employee"], "\n\n")
            all_text = ""
            email_body = enron_email_data.at[i, "body"]
            email_subject = enron_email_data.at[i, "Subject"]
            text = email_subject + "\n" + email_body
            all_text += text

            # Create a word cloud of the emails
            create_wordcloud(all_text)
            count += 1


# Call the function to analyze the emails
#analyze_emails()

#Top 20 Frequent Email users
num_of_mails = enron_email_data["Employee"].value_counts().head(20)

font = {'size'   : 8}

matplotlib.rc('font', **font)

plt.plot(num_of_mails)
plt.show()
