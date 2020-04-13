import os
import io,re
import requests,urllib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from bs4 import BeautifulSoup
from google.cloud import vision
from google.cloud.vision import types
from fake_useragent import UserAgent


path='C:/Users/gupte/Downloads/sardar.jpeg'

credential_path = "C:/Users/gupte/Downloads/My Project-34131c6caa0d.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
client = vision.ImageAnnotatorClient()
itype = input("Does the Image has text to describe it already? Type 1 if yes else 0 to describe it : ")

with io.open(path, 'rb') as image_file:
        content = image_file.read()

image = types.Image(content=content)
response = client.web_detection(image=image).web_detection


image = vision.types.Image(content=content)

text = client.document_text_detection(image=image)
title1 = text.full_text_annotation.text
print("Detected Text: " + title1)
if(itype==0):
    title1 = input("Title: ")

title_list = []
ua = UserAgent()

query = urllib.parse.quote_plus(title1) # Format into URL encoding
number_result = 10

google_url = "https://www.google.com/search?q=" + query + "&num=" + str(number_result)
response1 = requests.get(google_url, {"User-Agent": ua.random})
soup = BeautifulSoup(response1.text, "html.parser")

result_div = soup.find_all('div', attrs = {'class': 'ZINbbc'})

for page in response.pages_with_matching_images:
    urls = format(page.url)
    r = requests.get(urls)
    html = r.content
    soup = BeautifulSoup(html, 'html.parser')
    title_list.append(soup.title.string)

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

for title in title_list:
    titles.append(title)
module_url = "C:/Users/gupte/Downloads/1"
tf.disable_eager_execution()
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

def heatmap(x_labels, y_labels, values):
    
    fig, ax = plt.subplots()
    im = ax.imshow(values)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10,
         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, "%.2f"%values[i, j],
                           ha="center", va="center", color="w",fontsize=6)

    #fig.tight_layout()
    plt.show()

messages = [title1]
for msg in titles:
    messages.append(msg)
messages=list(set(messages))

X=[]
similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    message_embeddings_ = session.run(similarity_message_encodings, feed_dict={similarity_input_placeholder: messages})

    corr = np.inner(message_embeddings_, message_embeddings_)
    for i in corr[0]:
        X.append(i)
    X.pop(0)
    similarity= sum(X)/len(X)

if similarity>0.65:
    print("Real News!")
else:
    print("Fake News!")
print(similarity)
heatmap(messages, messages, corr)





