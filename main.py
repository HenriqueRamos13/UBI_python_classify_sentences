import json
import nltk
import ssl
from nltk.corpus import wordnet as wn
import pycld2 as cld2
from deep_translator import GoogleTranslator
from nltk.corpus import sentiwordnet as swn
import emoji

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('sentiwordnet')

EMOJIS = [emoji.emojize(":smile:"), emoji.emojize(":expressionless:"), emoji.emojize(":disappointed:")]
# EMOJIS = ["GOOD", "NEUTRAL", "BAD"]

DATA = []

MORE_RELEVANTS = []

USERS_POPULAR = []

def getTextLanguage(text):
    isReliable, textBytesFound, details = cld2.detect(text)
    # print(wn.langs())
    language = details[0][1]
    if language == 'en':
        return {"original": "en", "nltk": "eng"}
    elif language == 'pt':
        return {"original": "pt", "nltk": "por"}
    elif language == 'es':
        return {"original": "es", "nltk": "spa"}
    elif language == 'fr':
        return {"original": "fr", "nltk": "fra"}
    elif language == 'de':
        return {"original": "de", "nltk": "deu"}
    elif language == 'it':
        return {"original": "it", "nltk": "ita"}
    elif language == 'nl':
        return {"original": "nl", "nltk": "nld"}
    elif language == 'pl':
        return {"original": "pl", "nltk": "pol"}
    else:
        return {"original": "en", "nltk": "eng"}

def defineEmoji(sentiment):
    if sentiment > 0.0:
        return EMOJIS[0]
    elif sentiment == 0.0:
        return EMOJIS[1]
    else:
        return EMOJIS[2]

def calcTheEmotion(tweet):
    tokens = nltk.word_tokenize(tweet)
    text = nltk.Text(tokens)

    language = getTextLanguage(tweet)
    # print(language)

    finalResult = []

    for w in text.vocab():
        if w.isalpha():
            wordToCalculateSentiment = w if language["original"] == "en" else GoogleTranslator(source=language["original"], target='en').translate(w)
            wordSentiment = False
            score = 0.0
            try:
                wordSentiment = swn.senti_synset(wn.synsets(wordToCalculateSentiment).pop().name())
                score = wordSentiment.pos_score() - wordSentiment.neg_score()
            except:
                pass

            
            obj = {
                    "word": w,
                    'synonyms': [],
                    'count': text.count(w),
                    'sentiment': score if wordSentiment else 0,
                    'sentiment-data': str(wordSentiment) if wordSentiment else None,
                    'emoji': emoji.demojize(defineEmoji(score))
                }
            for syn in wn.synsets(w, lang=language["nltk"])[:10]:
                word = syn.name().split('.')[0].replace('_', ' ')
                finalWord = GoogleTranslator(source="en", target=language["original"]).translate(word)
                if finalWord == w:
                    pass
                else:
                    obj["synonyms"].append({
                        "word": word if (language["nltk"] == "eng") else finalWord,
                        "original": word,
                        'definition': syn.definition(),
                    })
            # print(json.dumps(obj, indent=4))
            finalResult.append(obj)
    return finalResult
    

def printData(dataSet, beautiful=False):
    for data in dataSet:
        if beautiful:
            beautifulPrint(data)
        else:
            print(json.dumps(data, indent=4))

def beautifulPrint(obj):
    print("-" * 80)
    print("@" + obj["user"]["screen_name"] + " - " + str(obj["user"]["followers_count"]) + " - " + str(obj["id"]) + "\n\n" + obj["text"])
    print("-" * 80)
    print("\n\n\n")

def reorderForPopularity():
    USERS_POPULAR.sort(key=lambda x: x["user"]["followers_count"], reverse=True)
    USERS_POPULAR[:] = USERS_POPULAR[:30]


def getMoreRelevants():
    for data in DATA:
        if data["user"]["followers_count"] > 1000:
            MORE_RELEVANTS.append(data)

def readAndPopulateData():
    with open("data.min.json", "r") as file:
        for line in file:
            try:
                json_object = json.loads(line)
                if json_object["user"]:
                    DATA.append(json_object)
                    USERS_POPULAR.append(json_object)
            except:
                pass

def getTweetById(id):
    for data in DATA:
        if data["id"] == id:
            return data

def getAnalysis(tweet):
    result = calcTheEmotion(tweet["text"])
    
    total = 0
    for r in result:
        print(r["sentiment"])
        total += r["sentiment"]
    total = total / len(result)
    if total > 0:
        tweet["sentiment"] = "GOOD"
    elif total == 0:
        tweet["sentiment"] = "NEUTRAL"
    else:
        tweet["sentiment"] = "BAD"
    print("Sentiment: " + tweet["sentiment"])
    print("Total: " + str(total))
    return tweet

def main():
    readAndPopulateData()
    # printData()
    reorderForPopularity()
    getMoreRelevants()
    printData(MORE_RELEVANTS, True)
    calcTheEmotion("i hate soccer")
    getAnalysis(getTweetById(1000448152737140736))
            

if __name__ == "__main__":
    main()