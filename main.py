import json
import nltk
import ssl
from nltk.corpus import wordnet as wn
import pycld2 as cld2
from deep_translator import GoogleTranslator
from nltk.corpus import sentiwordnet as swn
import emoji
import spacy
import secretskeys
import tweepy
import matplotlib.pyplot as plt

try:
    auth = tweepy.OAuthHandler(secretskeys.TWITTER_API_KEY,
                               secretskeys.TWITTER_API_KEY_SECRET,
                               secretskeys.TWITTER_ACCESS_TOKEN,
                               secretskeys.TWITTER_ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
except Exception as e:
    print("Error on start twitter - " + e)
    exit()

nlp = spacy.load("en_core_web_sm")

shouldAskIdiom = False


def startNltk():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('sentiwordnet')


LOG_THINGS = False

MAX_LENGTH = 300

EMOJIS = [
    emoji.emojize(":smile:"),
    emoji.emojize(":expressionless:"),
    emoji.emojize(":disappointed:")
]

DATA = []
DATA_WITH_RELEVANT_ENTITIES = []

USERS_POPULAR = []

ENTITIES = []
ENTITIES_MORE_RELEVANTS = []


def getTextLanguage(text, userInput=False):
    isReliable, textBytesFound, details = cld2.detect(text)
    # print(wn.langs())
    language = userInput if userInput != False else details[0][1]
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
    elif language == 'ru':
        return {"original": "ru", "nltk": "rus"}
    elif language == 'jpn':
        return {"original": "jpn", "nltk": "jpn"}
    else:
        if shouldAskIdiom:
            print("\nTweet:\n" + text + "\n")
            userInput = input(
                "Language not found or not detected. Please, enter the language code or press enter to continue with english (en): "
            )
            return getTextLanguage(text, userInput)
        else:
            return {"original": "en", "nltk": "eng"}


def defineEmoji(sentiment):
    if sentiment > 0.0:
        return EMOJIS[0]
    elif sentiment == 0.0:
        return EMOJIS[1]
    else:
        return EMOJIS[2]


def getTheScoreAndSentiment(word):
    wordSentiment = False
    score = 0.0

    try:
        wordSynsets = wn.synsets(word)

        if (len(wordSynsets) <= 0):
            return score, wordSentiment

        wordSentiment = swn.senti_synset(wordSynsets.pop().name())
        score = sum([wordSentiment.pos_score() - wordSentiment.neg_score()])
        # if LOG_THINGS == True and score != 0.0:
        #     print("-" * 25 + " " + word + " " + "-" * 25)
        #     print("Sentiment - " + str(wordSentiment) + "\nScore - " +
        #           str(score))
    except Exception as e:
        if LOG_THINGS == True:
            print("Error on get sentiment - " + e)
        pass
    return score, wordSentiment


def calcTheEmotion(tweet):
    tokens = nltk.word_tokenize(tweet)
    text = nltk.Text(tokens)

    language = getTextLanguage(tweet)

    finalResult = []

    for w in text.vocab():
        if w.isalpha():
            wordToCalculateSentiment = w if language[
                "original"] == "en" else GoogleTranslator(
                    source=language["original"], target='en').translate(w)

            for word in wordToCalculateSentiment.split(" "):
                score, wordSentiment = getTheScoreAndSentiment(word)

                obj = {
                    "word": w,
                    'synonyms': [],
                    'count': text.count(w),
                    'sentiment': score if wordSentiment else 0,
                    'sentiment-data':
                    str(wordSentiment) if wordSentiment else None,
                    'emoji': emoji.demojize(defineEmoji(score))
                }

                for syn in wn.synsets(w, lang=language["nltk"])[:10]:
                    word = syn.name().split('.')[0].replace('_', ' ')
                    finalWord = GoogleTranslator(
                        source="en",
                        target=language["original"]).translate(word)
                    if finalWord == w:
                        pass
                    else:
                        obj["synonyms"].append({
                            "word":
                            word if (language["nltk"] == "eng") else finalWord,
                            "original":
                            word,
                            'definition':
                            syn.definition(),
                        })
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
    print("@" + obj["user"]["screen_name"] + " - " +
          str(obj["user"]["followers_count"]) + " - " + str(obj["id"]) +
          "\n\n" + obj["text"])
    jsonLoad = json.loads(json.dumps(obj))
    if "sentiment" in jsonLoad:
        print("Sentiment: " + obj["sentiment"])
    if "sentiment-total" in jsonLoad:
        print("Total: " + str(obj["sentiment-total"]))
    if "entities" in jsonLoad:
        print("Entities: ")
        printATweetEntities(obj["entities"])
    print("-" * 80)
    print("\n\n\n")


def reorderEntitiesForPopularity():
    ENTITIES_MORE_RELEVANTS.sort(key=lambda x: x["count"], reverse=True)


def verifyIfEntityIsInRelevantList(entity):
    for ent in ENTITIES_MORE_RELEVANTS:
        if ent["entity"] == entity["entity"]:
            return True
    return False


def getMoreRelevantEntities():
    for entitie in ENTITIES:
        if not verifyIfEntityIsInRelevantList(entitie):
            entitie["count"] = 1
            entitie["sentiment-calcs"] = []
            ENTITIES_MORE_RELEVANTS.append(entitie)
        else:
            for ent in ENTITIES_MORE_RELEVANTS:
                if ent["entity"] == entitie["entity"]:
                    ent["count"] += 1

    reorderEntitiesForPopularity()
    ENTITIES_MORE_RELEVANTS[:] = ENTITIES_MORE_RELEVANTS[:10]
    if LOG_THINGS == True:
        print("-" * 80)
        print("Entities length - " + str(len(ENTITIES)))
        print("Entities more relevants length - " +
              str(len(ENTITIES_MORE_RELEVANTS)))
        print("-" * 80)
        print("Entities more relevants - ")
        print(json.dumps(ENTITIES_MORE_RELEVANTS, indent=4))


def getTweetEntities(tweet):
    doc = nlp(tweet)
    entities = []
    for ent in doc.ents:
        if LOG_THINGS == True:
            print(ent.text, ent.label_)
        entities.append({
            "entity": ent.text,
            "type": ent.label_,
        })
    return entities


def findTweetsInTwitter(search, tweets=[]):
    if len(tweets) == 0:
        tweets = tweepy.Cursor(
            api.search_tweets, q=search,
            lang="en").items(MAX_LENGTH if MAX_LENGTH != False else 100)

    length = 0

    for tweet in tweets:
        length += 1
        if len(search) > 0:
            tweet_json = tweet._json
        else:
            tweet_json = tweet
        entities = getTweetEntities(tweet_json["text"])
        tweet_json["entities"] = entities
        ENTITIES.extend(entities)
        DATA.append(tweet_json)
        USERS_POPULAR.append(tweet_json)

        if LOG_THINGS == True:
            beautifulPrint(tweet_json)
            print("\n\nTweets found - " + str(length))


def readAndPopulateData(searchBy):
    if len(searchBy) > 0:
        findTweetsInTwitter(searchBy, [])
    else:
        objects = []
        length = 0
        with open("data.min.json", "r") as file:
            for line in file:
                try:
                    json_object = json.loads(line)
                    if json_object["user"]:
                        objects.append(json_object)
                except:
                    pass
                if MAX_LENGTH != False:
                    length += 1
                    if length == MAX_LENGTH:
                        break

        findTweetsInTwitter("", objects)


def getTweetById(id):
    for data in DATA:
        if data["id"] == id:
            return data


def printATweetEntities(entities):
    for ent in entities:
        print(ent["entity"] + " - " + ent["type"])


def getAnalysis(tweet):
    entitie = tweet["entities"][0]

    result = calcTheEmotion(tweet["text"])

    total = 0
    differentOfZero = 0
    for r in result:
        # print(r["sentiment"])
        total += r["sentiment"]
        if r["sentiment"] != 0:
            differentOfZero += 1
    if differentOfZero == 0:
        differentOfZero = 1
    total = total / differentOfZero

    indexInRELEVANT_ENTITIES = 0

    for ent in ENTITIES_MORE_RELEVANTS:
        for tweetEnt in tweet["entities"]:
            if ent["entity"] == tweetEnt["entity"]:
                if indexInRELEVANT_ENTITIES < len(ENTITIES_MORE_RELEVANTS):
                    ENTITIES_MORE_RELEVANTS[indexInRELEVANT_ENTITIES][
                        "sentiment-calcs"].append(total)
        indexInRELEVANT_ENTITIES += 1

    if total > 0:
        tweet["sentiment"] = "GOOD"
    elif total == 0:
        tweet["sentiment"] = "NEUTRAL"
    else:
        tweet["sentiment"] = "BAD"
    tweet["sentiment-total"] = total

    if LOG_THINGS == True:
        beautifulPrint(tweet)

    return tweet


def getUserInputId():
    while True:
        userInputId = input("Enter the tweet id: ")
        try:
            userInputId = int(userInputId)

            return userInputId
        except:
            print("Invalid input. Please try again.")


def getOnlyTweetsWithRelevantEntities():
    break_second_loop = False

    for tweet in DATA:
        for ent in tweet["entities"]:
            for relevantEnt in ENTITIES_MORE_RELEVANTS:
                if ent["entity"] == relevantEnt["entity"]:
                    DATA_WITH_RELEVANT_ENTITIES.append(tweet)
                    break_second_loop = True
                    break
            if break_second_loop:
                break_second_loop = False
                break

    print("DATA length - " + str(len(DATA)))
    print("DATA_WITH_RELEVANT_ENTITIES length - " +
          str(len(DATA_WITH_RELEVANT_ENTITIES)))


def getUserInput(text):
    while True:
        userInput = input(text)
        try:
            return userInput
        except:
            print("Invalid input. Please try again.")


def showGraph():
    x = []
    y = []
    for ent in ENTITIES_MORE_RELEVANTS:
        x.append(ent["entity"])
        y.append(ent["sentiment-total"])

    plt.ylim(-1, 1)
    plt.bar(x, y)
    plt.show()


def main():
    startNltk()

    # askIdiom = input(
    #     "Do you want to be notified if a idiom is not found? (y/n) (Default: n) "
    # )

    # if askIdiom == "y":
    #     shouldAskIdiom = True

    searchBy = getUserInput(
        "Search by (press enter if want to use the default data by JSON file): "
    )

    readAndPopulateData(searchBy)

    getMoreRelevantEntities()

    getOnlyTweetsWithRelevantEntities()

    for tweet in DATA_WITH_RELEVANT_ENTITIES:
        getAnalysis(tweet)

    for ent in ENTITIES_MORE_RELEVANTS:
        total = 0.0
        for calc in ent["sentiment-calcs"]:
            total += calc
        if total != 0:
            total = total / len(ent["sentiment-calcs"])
        else:
            total = 0.0
        ent["sentiment-total"] = total

        if LOG_THINGS == True:
            print(json.dumps(ent, indent=4))

    showGraph()


if __name__ == "__main__":
    main()
