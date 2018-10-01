from flask import render_template, Flask, request
import pandas as pd
import numpy as np
import wget
import tweepy
import datetime
from tweepy import OAuthHandler
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img
import os
import json


#Initialize app
app = Flask(__name__, static_url_path='/static')



# Index/home page
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

# Art selection page
@app.route('/art', methods=['GET', 'POST'])
def art():

    NYMdf = pd.read_csv('C:/Users/mattm/InsightAppMM/NYMuseums.csv')

    datePicUpdate = np.genfromtxt("C:/Users/mattm/InsightAppMM/dateupdate.csv", delimiter=",", dtype=str)

    if datetime.date.today().strftime('%m/%d/%Y') in datePicUpdate:
        print("Pictures already loaded")
    else:
# Removing yesterdays images
        twitterFP=[]
        for path, subdirs, files in os.walk("C:/Users/mattm/InsightAppMM/static/twitart/"):
            for name in files:
                twitterFP.append(os.path.join(path, name))
        for f in twitterFP:
            os.remove(os.path.join("C:/Users/mattm/InsightAppMM/static/twitart/", f))
# Getting new images
        for i in range(0, len(NYMdf["Twitter"])):
            if str(NYMdf["Twitter"][i]) != "nan":
                sn = str(NYMdf["Twitter"][i])
                tweets = api.user_timeline(screen_name=sn,
                                   count=200, include_rts=False,
                                   exclude_replies=True)
                media_files = set()
                for status in tweets:
                    media = status.entities.get('media', [])
                    if(len(media) > 0):
                        media_files.add(media[0]['media_url'])
                media_files = list(media_files)
                for j in range(0, len(media_files)):
                    fp = "C:/Users/mattm/InsightAppMM/static/twitart/"+str(NYMdf["Twitter"][i])+"/pics/"+str(j)+".jpg"
                    wget.download(media_files[j], out=fp)

# Filtering out not art
        nah, musfolds, nope = next(os.walk("C:/Users/mattm/InsightAppMM/static/twitart/"))
        for g in musfolds:
            path, dirs, files = next(os.walk("C:/Users/mattm/InsightAppMM/static/twitart/"+str(g)+"/pics/"))
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
# Updating last accessed date
        datePicUpdate = np.append(datePicUpdate, datetime.date.today().strftime('%m/%d/%Y'))
        np.savetxt("dateupdate.csv", datePicUpdate, delimiter=",", fmt='%s')

# Making paths and value labels to be dynamically displayed
    imagepaths = []
    imagevalues = []
    for i in range(0, len(NYMdf["Twitter"])):
        if str(NYMdf["Twitter"][i]) != "nan":
            for j in range(0, 5):
                fp = "/static/twitart/"+NYMdf['Twitter'][i]+"/pics/"+str(j)+".jpg"
                cbval = NYMdf['Twitter'][i]+ str(j)
                if os.path.exists("C:/Users/mattm/InsightAppMM"+fp):
                    imagepaths.append(fp)
                    imagevalues.append(cbval)
    imageinfo = zip(imagepaths, imagevalues)


    return render_template('art.html', imageinfo=imageinfo)

# Museum recommendation page
@app.route('/museums', methods=['GET', 'POST'])
def museums():

# Getting results
    df = pd.DataFrame(request.form.to_dict(flat=False))
    df = df.apply(pd.to_numeric)

    museSco = []
    for i in NYMdf['Twitter']:
        if i != 'nan':
            filter_col = [col for col in df if col.startswith(str(i))]
            x = df[filter_col].mean(axis=1)
            museSco.append(x.tolist()[0])

    df = NYMdf
    df['museumScore'] = museSco
    df['museumScore'] = df['museumScore']*100
    df = df.round({'museumScore':2})
    herp = df.sort_values('museumScore', ascending = False).reset_index()
    derp = range(0, 5)


    return render_template('museums.html', herp=herp, derp=derp)

#NYC museum information
NYMdf = pd.read_csv('C:/Users/mattm/InsightAppMM/NYMuseums.csv')

#Reading in the trained convolutional neural network VGG16 transfer trained on
#art pictures and not art, also setting up stuff for filtering
from keras.applications import VGG16
vgg_conv = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
vgg_conv._make_predict_function()
ArtClassy = keras.models.load_model('ArtClass.h5')
ArtClassy._make_predict_function()
datagen = ImageDataGenerator(rescale=1./255)
#Twitter api information, this shouldn't get loaded to github
consumer_key = '5NSvgZd4QtrRF4LjPB8eHCUfe'
consumer_secret = 'PH5ALU8G3k7iXQmmTTgw0608zWNA58qS7OgCX97MjX85KMRGHD'
access_token = '1039211579538120704-5ABB2qxnEOJn0mFKTFcpLN2BnHfJfx'
access_secret = 'mV8NhgIIQbsG99WnqLvoSQ2VIb4QQw6zp0GLtqysKuq0a'


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
