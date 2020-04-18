import os.path
import tkinter as tk
from tkinter import *
from tkinter import filedialog
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

credible = ['economictimes.', 'huffingtonpost.', 'theprint.', 'thelogicalindian.', 'thequint.', 'altnews.', 'wsj.', 'nypost.', 'nytimes.', 'bbc.', 'reuters.', 'economist.', 'pbs.', 'aljazeera.', 'thewire.', 'theatlantic.', 'theguardian.', 'edition.cnn',
            'cnbc.', 'scroll.in', 'financialexpress.', 'npr.', 'usatoday.', 'snopes.', 'politifact.', 'timesofindia.','indiatoday.','hindustantimes.',]


def similar (X,credible,position):
    similarity= sum(X)/len(X)
    count=0
    for i in position:
        if(X[i]>0.7):
            count+=1
            similarity+=0.08
        if(X[i]<0.35)
            similarity-=0.08
    if(count>=(len(X)/3)):
        similarity+=0.25
    if similarity>0.65:
        print("Real News!")
    else:
        print("Fake News!")
    return similarity


def values(corr):
    X=[]
    for i in corr[0]:
        X.append(i)
    X.pop(0)
    return X

def USE(messages):
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
    return titles,links

def credible_list(list_of_page_urls):

    c_list = []
    position=[]
    c_length = len(credible)

    url_length = len(list_of_page_urls)

    f = [[0 for j in range(c_length)] for i in range(url_length)]
    for i in range(url_length):
        for j in range(c_length):
            f[i][j] = list_of_page_urls[i].find(credible[j])
            if((list_of_page_urls[i].find(credible[j])) > 0):
                
                c_list.append(list_of_page_urls[i])
                position.append(i)
    if c_list == []:
        print("No credible sources have used this image, please perform human verification.")
        print("--------------------------------------------------------------------------------")
        
    c_list=list(set(c_list))
    return(c_list,position)

def merge(A,B):
    for title in A:
        B.append(title)
    B=list(set(B))
    return B

def image_search(response):
    title_list=[]
    url_list=[]
    for page in response.pages_with_matching_images:
        urls = format(page.url)
        url_list.append(urls)
        r = requests.get(urls)
        html = r.content
        soup = BeautifulSoup(html, 'html.parser')
        title_list.append(soup.title.string)
    return title_list,url_list





def fake_news_image(path,title1):
    tlist=[]
    llist=[]
    ilist=[]
    illist=[]
    final=[]
    position=[]
    credential_path=os.path.join(my_path,"My Project-34131c6caa0d.json")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    #itype = int(input("Does the Image has text to describe it already? Type 1 if yes else 0 to describe it : "))

    #to find image on web
    client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
            content = image_file.read()
    image = types.Image(content=content)
    response = client.web_detection(image=image).web_detection
    #Text Detection
    image = vision.types.Image(content=content)
    text = client.document_text_detection(image=image)
    if(title1==''):
        title1 = text.full_text_annotation.text

    ilist,illist=image_search(response)
    tlist,llist =google_search(title1)
    final=values(USE(message(title1,merge(ilist,tlist))))
    llist=merge(illist,llist)
    tlist,position=credible_list(llist)
    similarity = similar(final,tlist,position)
    
    return similarity,tlist,title1

def fake_news_text(title1):
    tlist=[]
    llist=[]
    final=[]
    position=[]
    
    tlist,llist = google_search(title1)
    final=values(USE(message(title1,tlist)))
    tlist,position=credible_list(llist)
    similarity = similar(final,tlist,position)
    print(similarity)
    return similarity,tlist
    







argument=0
my_path=os.path.abspath(os.path.dirname(__file__))
#print("\n\nFAKE NEWS DETECTION\n\n")
#while(argument!=3):
   # print("\n1. News Type : Text\n2. News Type : Image\n3. Exit\n"  )
   #argument = int(input("Enter A choice : "))
    #if(argument==1):
       # title1 = input("Text : ")
        #fake_news_text(title1)
    #if(argument==2):
        #fake_news_image()
    #if(argument==3):
        #print("\nThank You!\n")
   # else:
     #   print("Enter a valid choice!") 

