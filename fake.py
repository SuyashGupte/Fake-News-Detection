import os.path
from pathlib import Path
import io,re
import requests,urllib
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from google.cloud import vision
from google.cloud.vision import types
from fake_useragent import UserAgent

def similar (X):
    similarity= sum(X)/len(X)
    return similarity


def values(corr):
    X=[]
    for i in corr[0]:
        X.append(i)
    X.pop(0)
    return X

def USE(messages):
    my_path=os.path.abspath(os.path.dirname(__file__))
    path=os.path.join(my_path,"model","1")
    module_url = path
    tf.disable_eager_execution()
    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(path)
    similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
    similarity_message_encodings = embed(similarity_input_placeholder)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        message_embeddings_ = session.run(similarity_message_encodings, feed_dict={similarity_input_placeholder: messages})
        corr = np.inner(message_embeddings_, message_embeddings_)
    return corr


def message(title1,descriptions):
    descriptions=list(set(descriptions))
    messages = [title1]
    for msg in descriptions:
        messages.append(msg)
    return messages

def google_search(title1):
    ua = UserAgent()

    query = urllib.parse.quote_plus(title1) # Format into URL encoding
    number_result = 10

    google_url = "https://www.google.com/search?q=" + query + "&num=" + str(number_result)
    response1 = requests.get(google_url, {"User-Agent": ua.random})
    soup = BeautifulSoup(response1.text, "html.parser")

    result_div = soup.find_all('div', attrs = {'class': 'ZINbbc'})

    links = []
    titles = []
    descriptions = []
    data = []
    for r in result_div:
        # Checks if each element is present, else, raise exception
        try:
            link = r.find('a', href = True)
            title = r.find('div', attrs={'class':'vvjwJb'}).get_text()
            description = r.find('div', attrs={'class':'s3v9rd'}).get_text()
            
            # Check to make sure everything is present before appending
            if link != '' and title != '' and description != '': 
                links.append(link['href'])
                titles.append(title)
                descriptions.append(description)
                data.append({'title' : title, 'link' :link['href'], 'description': description})
        # Next loop if one element is not present
        except:
            continue

    to_remove = []
    clean_links = []
    for i, l in enumerate(links):
        clean = re.search('\/url\?q\=(.*)\&sa',l)

        # Anything that doesn't fit the above pattern will be removed
        if clean is None:
            to_remove.append(i)
            continue
        clean_links.append(clean.group(1))

    # Remove the corresponding titles & descriptions
    for x in to_remove:
        del titles[x]
        del descriptions[x]
    return titles


def fake_news_text():
    title1 = input("Text : ")
    similarity = similar(values(USE(message(title1,(google_search(title1))))))
    if similarity>0.65:
        print("Real News!")
    else:
        print("Fake News!")
    print(similarity)
    

def switch(argument):
    switcher = {
        1: fake_news_text(),
        2:"gh",
        3:"Thank You!",
    }
    return switcher.get(argument, "Please Enter Valid choice!")


argument = 999 
print("\n\nFAKE NEWS DETECTION\n\n")
while(argument!=3):
    print("\n1. News Type : Text\n2. News Type : Image\n3. Exit\n"  )
    argument = int(input("Enter A choice : "))
    print(switch(argument))