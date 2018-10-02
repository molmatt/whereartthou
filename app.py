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
from keras.applications import VGG16
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
    wd = os.getcwd()
    twitartFP = os.path.join(wd,'static/twitart/')
    NYMfp = os.path.join(wd,'NYMuseums.csv')
    NYMdf = pd.read_csv(NYMfp)

    dateFP = os.path.join(wd,'dateupdate.csv')
    datePicUpdate = np.genfromtxt(dateFP, delimiter=",", dtype=str)

    if datetime.date.today().strftime('%m/%d/%Y') in datePicUpdate:
        print("Pictures already loaded")
    else:
# Removing yesterdays images
        twitterFP=[]

        for path, subdirs, files in os.walk(twitartFP):
            for name in files:
                twitterFP.append(os.path.join(path, name))
        for f in twitterFP:
            os.remove(os.path.join(twitartFP, f))
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
                media_files = list(media_files)
                for j in range(0, len(media_files)):
                    fp = twitartFP+str(NYMdf["Twitter"][i])+"/pics/"+str(j)+".jpg"
                    wget.download(media_files[j], out=fp)

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
# Updating last accessed date
        datePicUpdate = np.append(datePicUpdate, datetime.date.today().strftime('%m/%d/%Y'))
        np.savetxt("dateupdate.csv", datePicUpdate, delimiter=",", fmt='%s')

# Making paths and value labels to be dynamically displayed
    imagepaths = []
    imagevalues = []
    nah, musfolds, nope = next(os.walk(twitartFP))
    for g in musfolds:
        path, dirs, files = next(os.walk(twitartFP+str(g)+"/pics/"))
        if len(files) > 4:
            for x in range(0, 5):
                imagepaths.append('/static/twitart/'+g+'/pics/'+files[x])
                imagevalues.append(g + str(x))
        if len(files) == 4:
            for x in range(0, 4):
                imagepaths.append('/static/twitart/'+g+'/pics/'+files[x])
                imagevalues.append(g + str(x))
        if len(files) == 3:
            for x in range(0, 3):
                imagepaths.append('/static/twitart/'+g+'/pics/'+files[x])
                imagevalues.append(g + str(x))
        if len(files) == 2:
            for x in range(0, 2):
                imagepaths.append('/static/twitart/'+g+'/pics/'+files[x])
                imagevalues.append(g + str(x))
        if len(files) == 1:
            imagepaths.append('/static/twitart/'+g+'/pics/'+files[0])
            imagevalues.append(g + '0')

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
NYMdf = pd.read_csv('NYMuseums.csv')

#Reading in the trained convolutional neural network VGG16 transfer trained on
#art pictures and not art, also setting up stuff for filtering

vgg_conv = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
vgg_conv._make_predict_function()
ArtClassy = keras.models.load_model('ArtClass.h5')
ArtClassy._make_predict_function()
datagen = ImageDataGenerator(rescale=1./255)

#Twitter api information, this shouldn't get loaded to github
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
