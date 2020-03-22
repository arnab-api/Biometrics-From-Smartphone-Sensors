#%%
# loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import librosa
import keras
from sklearn.preprocessing import LabelEncoder
import sklearn
import itertools
from scipy.signal import find_peaks
import xgboost as xgb
import random


#%%
# Plot confusion matrix code ready
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues , mode = 'normal'):
    if(mode == 'percent'):
        cm = cm*100/cm.sum()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, with little normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        strr = str(cm[i,j])
        if(mode == 'percent'):
            strr = str(round(cm[i, j] , 2)) + '%'
        plt.text(j, i, strr ,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


#%%
# dataset preprocessing code ready
def processPeaks(signal , plot = False):

    peaks, _ = find_peaks(signal, height=0)

    if(plot == True):
        plt.plot(signal)
        print(peaks)

    peak_cnt = len(peaks)
    peak_diff = 0
    for i in range(len(peaks)):
        if(i == 0):
            peak_diff += peaks[i] - 0
        else:
            peak_diff += peaks[i] - peaks[i-1]
    
    if(peak_cnt > 0): 
        peak_diff /= len(peaks)

    return peak_cnt, peak_diff

def preprocessOneVector(feature_vec):
    ax = []
    ay = []
    az = []
    for i in range(feature_vec.shape[0]):
        ax.append(feature_vec[i]['acc_x'])
        ay.append(feature_vec[i]['acc_y'])
        az.append(feature_vec[i]['acc_z'])

    fft_ax = np.fft.fft(ax)
    fft_ay = np.fft.fft(ay)
    fft_az = np.fft.fft(az)

    # MEAN
    ax_mean = np.mean(ax)
    ay_mean = np.mean(ay)
    az_mean = np.mean(az)
    fft_ax_mean = np.mean(fft_ax)
    fft_ay_mean = np.mean(fft_ay)
    fft_az_mean = np.mean(fft_az)

    # MEDIAN
    ax_median = np.median(ax)
    ay_median = np.median(ay)
    az_median = np.median(az)
    fft_ax_median = np.median(fft_ax)
    fft_ay_median = np.median(fft_ay)
    fft_az_median = np.median(fft_az)

    # MAGNITUDE
    mag = 0
    for i in range(feature_vec.shape[0]):
        val = 0
        val += feature_vec[i]['acc_x'] * feature_vec[i]['acc_x'] 
        val += feature_vec[i]['acc_y'] * feature_vec[i]['acc_y'] 
        val += feature_vec[i]['acc_z'] * feature_vec[i]['acc_z'] 
        mag += math.sqrt(val)
    mag /= feature_vec.shape[0]

    # CROSS-CORRELATION
    Corr_xz = np.correlate(ax , az)
    Corr_yz = np.correlate(ay , az)


    # AVG DIFF FROM MEAN
    avg_diff_x = 0
    avg_diff_y = 0
    avg_diff_z = 0
    for i in range(feature_vec.shape[0]):
        avg_diff_x += abs(ax_mean - ax[i])
        avg_diff_y += abs(ay_mean - ay[i])
        avg_diff_z += abs(az_mean - az[i])

    avg_diff_x /= feature_vec.shape[0]
    avg_diff_y /= feature_vec.shape[0]
    avg_diff_z /= feature_vec.shape[0]

    # SPECTRAL CENTROID
    spc_x = librosa.feature.spectral_centroid(np.array(ax))[0][0]
    spc_y = librosa.feature.spectral_centroid(np.array(ay))[0][0]
    spc_z = librosa.feature.spectral_centroid(np.array(az))[0][0]

    # PEAKS
    x_peak_cnt , x_peak_diff = processPeaks(ax)
    y_peak_cnt , y_peak_diff = processPeaks(ay)
    z_peak_cnt , z_peak_diff = processPeaks(az)


    feature_processed = [
        ax_mean, ay_mean, az_mean,
        fft_ax_mean.real, fft_ax_mean.imag, fft_ay_mean.real, fft_ay_mean.imag, 
        fft_az_mean.real, fft_az_mean.imag,

        ax_median, ay_median, az_median,
        fft_ax_median.real, fft_ax_median.imag, fft_ay_median.real, fft_ay_median.imag, 
        fft_az_median.real, fft_az_median.imag,

        mag,

        Corr_xz, Corr_yz,

        avg_diff_x, avg_diff_y, avg_diff_z,

        spc_x, spc_y , spc_z,

        # x_peak_cnt, x_peak_diff, y_peak_cnt, y_peak_diff, z_peak_cnt, z_peak_diff 
    ]

    return feature_processed

#%%
# pca2D visalisation code ready
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualizePCA2D(X , y , colors):

    pca = TSNE(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['principal component 1', 'principal component 2'])
    # principalDf.head(5)

    #%%
    targetDf = pd.DataFrame(data = np.array(y) , columns = ['target'])
    # targetDf.head()

    #%%
    finalDf = pd.concat([principalDf, targetDf], axis = 1)
    print(finalDf.head(5))

    #%%
    # Visualize data
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 Component PCA', fontsize = 20)


    targets = list(set(y))

    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()

#%%
# pca 3D visualisation code ready
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
from sklearn.manifold import TSNE


def visualizePCA3D(X,y):
    np.random.seed(5)

    centers = [[1, 1], [-1, -1], [1, -1]]

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = TSNE(n_components=3)
    # pca.fit(X)
    X = pca.fit_transform(X)

    le = LabelEncoder()
    y_le = le.fit_transform(y)

    arr = list(set(zip(y , y_le)))
    print(arr)

    print(X[0:5])
    for name, label in arr:
        ax.text3D(X[y == label, 0].mean(),
                X[y == label, 1].mean() + 1.5,
                X[y == label, 2].mean(), name,
                horizontalalignment='center',
                bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results


    yy = np.choose(y_le, list(set(y_le))).astype(np.float)

    print(X.shape, y.shape, y_le.shape, yy.shape)
    ax.scatter(
        X[:, 0], X[:, 1], X[:, 2], 
        c=yy, cmap='tab10',
        edgecolor='k'
    )

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()

#%%
#JSON dataset loading code ready

import json
def loadJSONDataset(file_name):
    with open(file_name) as f:
        data = json.load(f)
    arr = np.array(json.loads(data[0]['vector']))
    print(arr)
    print(type(arr) , arr.shape)

    data_acc = []
    for reading in data:
        if(reading["sensor"] != "Accelerometer"):
            continue
        arr = np.array(json.loads(reading['vector']))
        data_acc.append({
            'acc_x': arr[0],
            'acc_y': arr[1],
            'acc_z': arr[2]
        })
    return np.array(data_acc)

def segmentaizeData(data_acc):
    data_seg = []
    st = 0
    while(st + 100 < len(data_acc)):
        nd = st + 100
        data_seg.append(data_acc[st:nd])
        st = st + 50
    data_seg = np.array(data_seg)
    return data_seg

def extractFeatureFromSegmentizeData(data_seg):
    data_feature = []
    for i in range(data_seg.shape[0]):
        feature_clean = preprocessOneVector(data_seg[i])
        data_feature.append(feature_clean)
    data_feature = np.array(data_feature)
    return data_feature

#%%
# New dataset loading code ready
def loadNewData(user_data):
    X = []
    y = []
    for (user_name, st, nd) in user_data:
        for i in range(st,nd+1):
            file_name = 'new_data/'+user_name+str(i)+'.json'
            data = loadJSONDataset(file_name)
            data_seg = segmentaizeData(data)
            data_feature = extractFeatureFromSegmentizeData(data_seg)

            print(user_name , i , "::", nd , data_feature.shape)
            for i in range(len(data_feature)):
                X.append(data_feature[i])
                y.append(user_name)
    
    zz = list(zip(X,y))
    random.shuffle(zz)
    X,y = zip(*zz)

    return np.array(X) , np.array(y)

#%%

#loading new dataset
user_data = [
    ('Raz', 1,  2),
    ('Tipu',1 , 2),
    ('Sajib',1 , 2),
    ('Tuhin', 1, 2),
    ('ShafiqVai', 1, 2)
]

X_train, y_train = loadNewData(user_data)
print('loaded train dataset :: ' , X_train.shape , y_train.shape)
#%%
test_data = [
    ('Raz', 3,  3),
    ('Tipu',3 , 3),
    ('Sajib',3 , 3),
    ('Tuhin', 3, 3),
    ('ShafiqVai', 3, 3)
]

X_test, y_test = loadNewData(test_data)
print('loaded test dataset :: ' , X_test.shape , y_test.shape)


#%%
visualizePCA2D(X_train , y_train , colors = ['r' , 'g' , 'b' , 'y' , 'w'])

#%%
###### TRIPLET CODING START ######

#%%
# SCATTER DATA
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context('notebook', font_scale=1.5,
                rc={"lines.linewidth": 2.5})

from sklearn.manifold import TSNE

def scatter(x, labels, subtitle=None):
    # We choose a color palette with seaborn.
    pca = TSNE(n_components=2)
    x = pca.fit_transform(X)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    
    print(x.shape , labels.shape)

    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
        
    if subtitle != None:
        plt.suptitle(subtitle)
        
    plt.show()




#%%
print(X_train.shape , y_train.shape)
scatter(X_train , y_train)

#%%
