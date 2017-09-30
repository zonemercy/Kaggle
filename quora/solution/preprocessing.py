import numpy


#https://www.kaggle.com/life2short/data-processing-replace-abbreviation-of-word
punctuation='["\'?,\.]' # I will replace all these punctuation with ''
abbr_dict={
    "what's":"what is",
    "what're":"what are",
    "who's":"who is",
    "who're":"who are",
    "where's":"where is",
    "where're":"where are",
    "when's":"when is",
    "when're":"when are",
    "how's":"how is",
    "how're":"how are",

    "i'm":"i am",
    "we're":"we are",
    "you're":"you are",
    "they're":"they are",
    "it's":"it is",
    "he's":"he is",
    "she's":"she is",
    "that's":"that is",
    "there's":"there is",
    "there're":"there are",

    "i've":"i have",
    "we've":"we have",
    "you've":"you have",
    "they've":"they have",
    "who've":"who have",
    "would've":"would have",
    "not've":"not have",

    "i'll":"i will",
    "we'll":"we will",
    "you'll":"you will",
    "he'll":"he will",
    "she'll":"she will",
    "it'll":"it will",
    "they'll":"they will",

    "isn't":"is not",
    "wasn't":"was not",
    "aren't":"are not",
    "weren't":"were not",
    "can't":"can not",
    "couldn't":"could not",
    "don't":"do not",
    "didn't":"did not",
    "shouldn't":"should not",
    "wouldn't":"would not",
    "doesn't":"does not",
    "haven't":"have not",
    "hasn't":"has not",
    "hadn't":"had not",
    "won't":"will not",
    punctuation:'',
    '\s+':' ', # replace multi space with one single space
}
data.question1=data.question1.str.lower() # conver to lower case
data.question2=data.question2.str.lower()
data.question1=data.question1.astype(str)
data.question2=data.question2.astype(str)
data.replace(abbr_dict,regex=True,inplace=True)


# https://www.kaggle.com/currie32/the-importance-of-cleaning-text
stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']

text = re.sub(r"what's", "", text)
text = re.sub(r"What's", "", text)
text = re.sub(r"\'s", " ", text)
text = re.sub(r"\'ve", " have ", text)
text = re.sub(r"can't", "cannot ", text)
text = re.sub(r"n't", " not ", text)
text = re.sub(r"I'm", "I am", text)
text = re.sub(r"\bm\b", " am ", text)
text = re.sub(r"\'re", " are ", text)
text = re.sub(r"\'d", " would ", text)
text = re.sub(r"\'ll", " will ", text)
# text = re.sub(r"60k", " 60000 ", text)
text = re.sub(r"\be g\b", " eg ", text)
text = re.sub(r"\bb g\b", " bg ", text)
text = re.sub(r"\0s", "0", text)
text = re.sub(r"\b9 11\b", "911", text)
text = re.sub(r"e-mail", "email", text)
text = re.sub(r"\s{2,}", " ", text)
text = re.sub(r"quikly", "quickly", text)
text = re.sub(r"\busa\b", " America ", text)
text = re.sub(r"\bUSA\b", " America ", text)
text = re.sub(r"\bu s\b", " America ", text)
text = re.sub(r"\buk\b", " England ", text)
text = re.sub(r"\bUK\b", " England ", text)
text = re.sub(r"india", "India", text)
text = re.sub(r"switzerland", "Switzerland", text)
text = re.sub(r"china", "China", text)
text = re.sub(r"chinese", "Chinese", text) 
text = re.sub(r"imrovement", "improvement", text)
text = re.sub(r"intially", "initially", text)
text = re.sub(r"quora", "Quora", text)
text = re.sub(r"\bdms\b", "direct messages ", text)  
text = re.sub(r"demonitization", "demonetization", text) 
text = re.sub(r"actived", "active", text)
text = re.sub(r"kms", " kilometers ", text)
text = re.sub(r"KMs", " kilometers ", text)
text = re.sub(r"\bcs\b", " computer science ", text) 
text = re.sub(r"\bupvotes\b", " up votes ", text)
text = re.sub(r"\biPhone\b", " phone ", text)
text = re.sub(r"\0rs ", " rs ", text) 
text = re.sub(r"calender", "calendar", text)
text = re.sub(r"ios", "operating system", text)
text = re.sub(r"gps", "GPS", text)
text = re.sub(r"gst", "GST", text)
text = re.sub(r"programing", "programming", text)
text = re.sub(r"bestfriend", "best friend", text)
text = re.sub(r"dna", "DNA", text)
text = re.sub(r"III", "3", text) 
text = re.sub(r"the US", "America", text)
text = re.sub(r"Astrology", "astrology", text)
text = re.sub(r"Method", "method", text)
text = re.sub(r"Find", "find", text) 
text = re.sub(r"banglore", "Banglore", text)
text = re.sub(r"\bJ K\b", " JK ", text)

def substitute_thousands(text):
    matches = re.finditer(r'[0-9]+(?P<thousands>\s{0,2}k\b)', text, flags=re.I)
    result = ''
    len_offset = 0
    for match in matches:
        result += '{}000'.format(text[len(result)-len_offset:match.start('thousands')])
        len_offset += 3 - (match.end('thousands') - match.start('thousands'))
    result += text[len(result)-len_offset:]
    return result
text  = substitute_thousands(text)

