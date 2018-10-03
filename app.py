from flask import render_template, Flask, request
import pandas as pd
import numpy as np
import datetime
import os

#Initialize app
app = Flask(__name__, static_url_path='/static')

wd = os.getcwd()
twitartFP = os.path.join(wd,'static/twitart/')
NYMfp = os.path.join(wd,'NYMuseums.csv')
# Index/home page
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

# Art selection page
@app.route('/art', methods=['GET', 'POST'])
def art():
# Making paths and value labels to be dynamically displayed
    imagepaths = []
    imagevalues = []
    nah, musfolds, nope = next(os.walk(twitartFP))
    for g in musfolds:
        path, dirs, files = next(os.walk(twitartFP+'/'+str(g)+"/pics/"))
        if len(files) > 4:
            for x in range(0, 5):
                imagepaths.append('/static/twitart/'+g+'/pics/'+files[x])
                imagevalues.append(g + str(x))
        elif len(files) > 0:
            for x in range(0, len(files)):
                imagepaths.append('/static/twitart/'+g+'/pics/'+files[x])
                imagevalues.append(g + str(x))

    hey = range(0, len(imagepaths))

    return render_template('art.html', imagepaths=imagepaths, imagevalues=imagevalues, hey=hey)

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
