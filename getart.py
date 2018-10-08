from tweepy import OAuthHandler
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications import VGG16
import wget
import tweepy
import json
import os
import pandas as pd
import numpy as np

#NYC museum information
NYMdf = pd.read_csv('NYMuseums.csv')
#Reading in the trained convolutional neural network VGG16 transfer trained on
#art pictures and not art, also setting up stuff for filtering
vgg_conv = vgg_conv = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
vgg_conv._make_predict_function()
ArtClassy = keras.models.load_model('ArtClass.h5')
ArtClassy._make_predict_function()
datagen = ImageDataGenerator(rescale=1./255)

Twitter api information, this shouldn't get loaded to github
consumer_key = os.environ.get('consumer_key')
consumer_secret = os.environ.get('consumer_secre')
access_token = os.environ.get('access_token')
access_secret = os.environ.get('access_secret')

@classmethod
def parse(cls, api, raw):
    status = cls.first_parse(api, raw)
    setattr(status, 'json', json.dumps(raw))
    return status

# Status() is the data model for a tweet
tweepy.models.Status.first_parse = tweepy.models.Status.parse
tweepy.models.Status.parse = parse
# User() is the data model for a user profil
tweepy.models.User.first_parse = tweepy.models.User.parse
tweepy.models.User.parse = parse
# You need to do it for all the models you need

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

def getart():
    wd = os.getcwd()
    twitartFP = os.path.join(wd,'static/twitart/')
    NYMfp = os.path.join(wd,'NYMuseums.csv')
    NYMdf = pd.read_csv(NYMfp)

# Removing yesterdays images
    twitterFP=[]
    for path, subdirs, files in os.walk(twitartFP):
        for name in files:
            twitterFP.append(os.path.join(path, name))
    for f in twitterFP:
        os.remove(os.path.join(twitartFP, f))
    twitterFP=[]
# Getting new images
    for i in range(0, len(NYMdf["Twitter"])):
        if str(NYMdf["Twitter"][i]) != "nan":
            sn = str(NYMdf["Twitter"][i])
            tweets = api.user_timeline(screen_name=sn,
                               count=500, include_rts=False,
                               exclude_replies=True)
            media_files = set()
            for status in tweets:
                media = status.entities.get('media', [])
                if(len(media) > 0):
                    media_files.add(media[0]['media_url'])
            tweets = []
            media_files = list(media_files)
            if len(media_files) > 10:
                for j in range(0, 10):
                    try:
                        fp = twitartFP+str(NYMdf["Twitter"][i])+"/pics/"+str(j)+".jpg"
                        wget.download(media_files[j], out=fp)
                    except OSError:
                        continue
            elif len(media_files) > 0:
                for j in range(0, len(media_files)):
                    try:
                        fp = twitartFP+str(NYMdf["Twitter"][i])+"/pics/"+str(j)+".jpg"
                        wget.download(media_files[j], out=fp)
                    except OSError:
                        continue

# Filtering out not art
    nah, musfolds, nope = next(os.walk(twitartFP))
    for g in musfolds:
        path, dirs, files = next(os.walk(twitartFP+str(g)+"/pics/"))
        nTest = len(files)
        batch_size = 1
        if nTest != 0:
            test_dir = nah+str(g)
            test_features = np.zeros(shape=(nTest, 7, 7, 512))
            test_generator = datagen.flow_from_directory(
                test_dir,
                target_size=(224, 224),
                batch_size=1,
                class_mode=None,
                shuffle=False)
# Feeding through the VGG16
            i = 0
            for inputs_batch in test_generator:
                features_batch = vgg_conv.predict(inputs_batch)
                test_features[i * batch_size : (i + 1) * batch_size] = features_batch
                i += 1
                if i * batch_size >= nTest:
                    break
# Feeding VGG16 output through art classifying layers and filtering out 'not art'
            test_features = np.reshape(test_features, (nTest, 7 * 7 * 512))
            prediction = ArtClassy.predict_classes(test_features)
            for x in range(0, len(prediction)):
                if prediction[x] == 1:
                    os.remove(os.path.join(path, files[x]))

getart()
