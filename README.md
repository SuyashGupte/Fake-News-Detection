# Fake-News-Detection

This is Fake news detection program with fronent in Django.

For Text Input :-
1.	The text is searched on Google using BeautifulSoup4 . Thus we get the links and titles of the webpages which appear on searching this      query.		
2.	Now the input text is merged with the titles and passed to the model. We are using Universal Sentence Encoder which is a pretrained       model to get semantic similarity.
3.	The links are checked if they are credible by our predefined list of credible sources.
4.	All the similarity values are compared to determine if the news is Fake or Not.



For Image Input:-
1.	The Image is searched on the web using Google Vision API. We the links and titles from website with matching images.
2.	The user is asked to described the image if the image does not have text else Vision API identifies the text in image.
3.	The text is searched on google similar to the procedure in text input.
4.	All the titles are then combined and similarity is calculated as previously done for text input.

