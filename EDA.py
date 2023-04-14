

# In[32]:


import os
import json

from sklearn.feature_extraction.text import CountVectorizer


# In[1]:


import re
import pandas as pd
# text processing
import nltk
from nltk.tokenize import WordPunctTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
## needed for nltk.pos_tag function 
# nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
# visualization
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from wordcloud import WordCloud


# In[15]:


# Get the list of stop words
stop_words = stopwords.words('english')
stop_words.extend(["could","though","would","also","us"])
def normalization(text):
    word_punct_token = WordPunctTokenizer().tokenize(text)
    clean_token=[]
    for token in word_punct_token:
        new_token = re.sub(r'[^a-zA-Z]+', '', token) # remove any value that are not alphabetical
        if new_token != "" and len(new_token) >= 2: # remove empty value and single character value
            vowels=len([v for v in new_token if v in "aeiou"])
            if vowels != 0: # remove line that only contains consonants
                new_token = new_token.lower() # change to lower case
                clean_token.append(new_token)
    # Remove the stopwords from the list of tokens
    tokens = [x for x in clean_token if x not in stop_words]
    return tokens


# In[23]:


def pos_tag(tokens):
    # POS Tag every token and save into dataframe
    data_tagset = nltk.pos_tag(tokens)
    df_tagset = pd.DataFrame(data_tagset, columns=['Word', 'Tag'])
    # to focus on nouns, adjective and verb
    tagset_allowed = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP','VBZ']
    new_tagset = df_tagset.loc[df_tagset['Tag'].isin(tagset_allowed)]
    text = [str(x) for x in new_tagset['Word']]
    tag =[x for x in new_tagset['Tag'] if x != '']
    # Create lemmatizer object 
    lemmatizer = WordNetLemmatizer()# Lemmatize each word and display the output
    lemmatize_text = []
    for word in text:
        output = [word, lemmatizer.lemmatize(word, pos='n'),lemmatizer.lemmatize(word, pos='a'),lemmatizer.lemmatize(word, pos='v')]
        lemmatize_text.append(output)# create DataFrame using original words and their lemma words
    df = pd.DataFrame(lemmatize_text, columns =['Word', 'Lemmatized Noun', 'Lemmatized Adjective', 'Lemmatized Verb'])
    df['Tag'] = tag
        # replace with single character for simplifying
    df = df.replace(['NN','NNS','NNP','NNPS'],'n')
    df = df.replace(['JJ','JJR','JJS'],'a')
    df = df.replace(['VBG','VBP','VB','VBD','VBN','VBZ'],'v')
    df_lemmatized = df.copy()
    df_lemmatized['Tempt Lemmatized Word']=df_lemmatized['Lemmatized Noun'] + ' | ' + df_lemmatized['Lemmatized Adjective']+ ' | ' + df_lemmatized['Lemmatized Verb']
    lemma_word = df_lemmatized['Tempt Lemmatized Word']
    tag = df_lemmatized['Tag']
    i = 0
    new_word = []
    while i<len(tag):
        words = lemma_word[i].split('|')
        if tag[i] == 'n':        
            word = words[0]
        elif tag[i] == 'a':
            word = words[1]
        elif tag[i] == 'v':
            word = words[2]
        new_word.append(word)
        i += 1

    df_lemmatized['Lemmatized Word']=new_word
    return df_lemmatized


# In[17]:


#politics
pol_text=''
for dir_ in os.listdir('./politics'):
    if dir_.endswith('txt'):
        dir_ = './politics/'+dir_
        text_ = open(dir_,encoding='utf-8').readlines()
        for _ in text_:
            pol_text += (_+ " ")


# In[40]:


#entertainment
ent =json.load(open('./tweet_per_#Entertainment_tag2.json'))
ent_txt =''
for i in ent:
    ent_txt+=(i['full_text']+ " ")


# In[41]:


#sports
sport = json.load(open('./tweet_per_#sport_tag2.json'))
sport_txt =''
for i in sport:
    sport_txt+=(i['full_text']+ " ")


# In[50]:


import emoji
def process_text(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI) #Remove Emojis
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    # tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
    #         if w.lower() in words or not w.isalpha())
    return tweet


# In[49]:


ent[0]['full_text']


# In[51]:


process_text(ent[0]['full_text'])


# In[25]:


politics_tokens = normalization(text)

politics_df = pos_tag(politics_tokens)

politics_df


# In[ ]:





# In[43]:


ent_tokens = normalization(ent_txt)

ent_df = pos_tag(ent_tokens)

# politics_df


# In[ ]:





# In[44]:


sport_tokens = normalization(sport_txt)

sport_df = pos_tag(sport_tokens)


# In[35]:


def word_cloud_pos(df,title):
    lemma_word = [str(x) for x in df['Lemmatized Word']]
    tagset = df
    tagset_allowed = ['n']
    new_tagset = tagset.loc[tagset['Tag'].isin(tagset_allowed)]
    text = ' '.join(str(x) for x in new_tagset['Lemmatized Noun'])
    wordcloud = WordCloud(width = 1600, height = 800, max_words = 200, background_color = 'white').generate(text)
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    plt.savefig(f'Noun_WordCloud_{title}.png') 
    plt.show()
    # select only adjectives for word cloud
    tagset_allowed = ['a']
    new_tagset = tagset.loc[tagset['Tag'].isin(tagset_allowed)]
    text = ' '.join(str(x) for x in new_tagset['Lemmatized Adjective'])
    wordcloud = WordCloud(width = 1600, height = 800, max_words = 200, background_color = 'white').generate(text)
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    plt.savefig(f'adjective_WordCloud_{title}.png') 
    plt.show()
    tagset_allowed = ['v']
    new_tagset = tagset.loc[tagset['Tag'].isin(tagset_allowed)]
    text = ' '.join(str(x) for x in new_tagset['Lemmatized Verb'])
    wordcloud = WordCloud(width = 1600, height = 800, max_words = 200, background_color = 'white').generate(text)
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    plt.savefig(f'verb_WordCloud_{title}.png') 
    plt.show()
   


# In[36]:


word_cloud_pos(politics_df,'politics')


# In[46]:


word_cloud_pos(sport_df,'sport')


# In[45]:


word_cloud_pos(ent_df,'entertainment')


# In[33]:


#Using count vectoriser to view the frequency of bigrams
tagset_allowed = ['a','n','v']
new_tagset = politics_df.loc[politics_df['Tag'].isin(tagset_allowed)]
text = [' '.join(str(x) for x in new_tagset['Lemmatized Word'])]
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(text)
vectorizer.vocabulary_
sum_words = bag_of_words.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print (words_freq[:100])
#Generating wordcloud and saving as jpg image
words_dict = dict(words_freq)
WC_height = 800
WC_width = 1600
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,background_color = 'white')
wordCloud.generate_from_frequencies(words_dict)
plt.title('Most frequently occurring bigrams')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordCloud.to_file('wordcloud_bigram_title.jpg')


# In[34]:


#Using count vectoriser to view the frequency of trigrams
vectorizer = CountVectorizer(ngram_range=(3, 3))
bag_of_words = vectorizer.fit_transform(text)
vectorizer.vocabulary_
sum_words = bag_of_words.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print (words_freq[:100])
#Generating wordcloud and saving as jpg image
words_dict = dict(words_freq)
WC_height = 800
WC_width = 1600
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,background_color = 'white')
wordCloud.generate_from_frequencies(words_dict)
plt.title('Most frequently occurring trigrams')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordCloud.to_file('wordcloud_trigram_title.jpg')


# In[ ]:




