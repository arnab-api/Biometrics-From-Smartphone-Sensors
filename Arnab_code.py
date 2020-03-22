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
# dataset load code ready
def load_dataset(arr):

    X = []
    y = []

    total_datapoints = 0
    for val in arr:
        keyword = val['keyword']
        limit = val['limit']
        
        feature = []

        for l in range(limit):
            add = str(l + 1)
            df = pd.read_csv("dataset/"+keyword+add+".csv")

            total_datapoints += len(df)

            acc_x = df['ACCELEROMETER X (m/s²)']
            acc_y = df['ACCELEROMETER Y (m/s²)']
            acc_z = df['ACCELEROMETER Z (m/s²)']
            nv = df['NV']
            avg = df['AVG']
            sd = df['SD']


            for i in range(len(df)):
                feature.append(
                    {
                        'acc_x': acc_x[i],
                        'acc_y': acc_y[i],
                        'acc_z': acc_z[i],
                        'nv': nv[i],
                        'avg': avg[i],
                        'sd': sd[i]
                    }
                )
        st = 0
        while(st + 100  < len(feature)):
            nd = st + 100
            X.append(feature[st:nd])
            st += 50
            y.append(keyword)

    print("Total datapoints: " , total_datapoints)

    return X,y



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
print("loading dataset")
arr = [
    {'keyword': 'razzak' , 'limit': 3},
    {'keyword': 'saifulvai' , 'limit': 2},
    {'keyword': 'sajib' , 'limit': 3}
]
X,y = load_dataset(arr)
X = np.array(X)
print("dataset loaded :: " , X.shape , len(y))
y[0:5]


#%%
# shuffling dataset
zz = list(zip(X,y))
random.shuffle(zz)
X,y = zip(*zz)
X = np.array(X)
print("dataset loaded :: " , X.shape , len(y))
y[0:5]

#%%
# preprocess dataset
print("preprocessing dataset")
X_clean = []
for ff in range(X.shape[0]):
    feature_clean = preprocessOneVector(X[ff])
    X_clean.append(feature_clean)

le = LabelEncoder()
y_cat = le.fit_transform(y)
y_cat = keras.utils.to_categorical(y_cat)

X_clean = np.array(X_clean)
print("X: " , X_clean.shape)
print("Y: " , y_cat.shape)


#%%
# visualize loaded dataset
visualizePCA2D(X = X_clean ,y =  y ,colors = ['r' , 'g' , 'b'])


#%%
#TRAIN TEST DATA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_cat, test_size=0.33, random_state=31)

#%%
# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=12)
clf.fit(X_train , y_train)

#%%
pred = clf.predict(X_test)
score = sklearn.metrics.accuracy_score(y_test, pred)
print("accuracy: {}%".format(score*100))
print(y_test[0:5] , pred[0:5])

#%%
cm = sklearn.metrics.confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1), labels=[0, 1 , 2])
plot_confusion_matrix(cm, classes=['razzak' , 'saiful' , 'sajib'])


#%%
# XGBOOST
from xgboost import XGBClassifier
param = {
    'max_depth': 5,  
    'eta': 0.01, 
    'silent': 1, 
    'objective': 'multi:softprob',  
    'num_class': 3
}  
num_round = 100 
bst = XGBClassifier(max_depth = 7)
bst.fit(X_train , y_train.argmax(axis=1))

#%%
pred_xgb = bst.predict(X_test)
score = sklearn.metrics.accuracy_score(y_test, keras.utils.to_categorical(pred_xgb))
print("accuracy: {}%".format(score*100))

#%%
cm = sklearn.metrics.confusion_matrix(y_test.argmax(axis=1), pred_xgb, labels=[0, 1 , 2])
plot_confusion_matrix(cm, classes=['razzak' , 'saiful' , 'sajib'])


#%%
prob_xgb = bst.predict_proba(X_test)

for i in range(len(y_test)):
    print(prob_xgb[i] , prob_xgb[i].argmax() , y_test[i].argmax())




#%%
# loading JOSN data code ready

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


arnab = loadJSONDataset('walk_arnab.json')
print(arnab.shape)
arnab_seg = segmentaizeData(arnab)
print(arnab_seg.shape)
arnab_feature = extractFeatureFromSegmentizeData(arnab_seg)
print(arnab_feature.shape)

#%%
bst.predict_proba(arnab_feature)

#%%
arnab_label = []
for i in range(arnab_feature.shape[0]):
    if(i < 40):
        arnab_label.append(1)
    else:
        arnab_label.append(0)

A_train, A_test, b_train, b_test = train_test_split(arnab_feature, arnab_label, test_size=0.10, random_state=31)

#%%
print(A_train.shape , len(b_train))
print(A_test.shape , len(b_test))

#%%
from xgboost import XGBClassifier
param = {
    'max_depth': 5,  
    'eta': 0.01, 
    'silent': 1, 
    'objective': 'multi:softprob',  
    'num_class': 3
}  
num_round = 100 
arnab_bst = XGBClassifier(max_depth = 7)
arnab_bst.fit(A_train , b_train)


#%%
arnab_bst.predict_proba(X_test)

#%%

razzak = loadJSONDataset('walk_razzak.json')
razzak_seg = segmentaizeData(razzak)
razzak_feature = extractFeatureFromSegmentizeData(razzak_seg)

print(razzak_feature.shape)
#%%

arnab_bst.predict_proba(razzak_feature[0:50])


# #%%
arnab2 = loadJSONDataset('walk_arnab_2.json')
arnab2_seg = segmentaizeData(arnab2)
arnab2_feature = extractFeatureFromSegmentizeData(arnab2_seg)

print(arnab2_feature.shape)

#%%
arnab_bst.predict_proba(razzak_feature[0:50])

#%%
tuhin = loadJSONDataset('walk_tuhin.json')
tuhin_seg = segmentaizeData(tuhin)
tuhin_feature = extractFeatureFromSegmentizeData(tuhin_seg)

print(tuhin_feature.shape)

#%%
def appendData(X, y, addX, keyword):

    X = list(X)
    y = list(y)

    for i in range(len(addX)):
        X.append(addX[i])
        y.append(keyword)

    return X,y

X = []
y = []
X, y = appendData(X, y, razzak_feature, 'razzak')
# X, y = appendData(X, y, arnab2_feature, 'arnab2')
X, y = appendData(X, y, arnab_feature, 'arnab')
X, y = appendData(X, y, tuhin_feature, 'tuhin')


X = np.array(X)
y = np.array(y)

print("Tuhin:", tuhin_feature.shape)
print("Arnab:", arnab_feature.shape)
# print("Arnab2:", arnab2_feature.shape)
print("Razzak:", razzak_feature.shape)
print(X.shape, y.shape)

#%%
zz = list(zip(X , y))
random.shuffle(zz)
X, y = zip(*zz)

X = np.array(X)
y = np.array(y)
y[0:10]
#%%
visualizePCA2D(X,y,['r','y','w'])

#%%
from xgboost import XGBClassifier

bst = XGBClassifier(max_depth = 7)
bst.fit(X , y)


#%%
pred = bst.predict(arnab2_feature)
prob = bst.predict_proba(arnab2_feature)

for i in range(len(pred)):
    print(pred[i] , prob[i])
#%%

#%%

le = LabelEncoder()
y_le = le.fit_transform(y)
visualizePCA3D(X,y_le)


#%%
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
# XGBOOST
from xgboost import XGBClassifier
param = {
    'max_depth': 5,  
    'eta': 0.01, 
    'silent': 1, 
    'objective': 'multi:softprob',  
    'num_class': 5
}  
num_round = 100 
bst = XGBClassifier(max_depth = 7)
bst.fit(X_train , y_train)

#%%
pred_xgb = bst.predict(X_test)
score = sklearn.metrics.accuracy_score(y_test, pred_xgb)
print("accuracy: {}%".format(score*100))

#%%
list(set(y_train))
#%%
cm = sklearn.metrics.confusion_matrix(y_test, pred_xgb, labels=list(set(y_train)))
plot_confusion_matrix(cm, classes=list(set(y_train)))


#%%
prob_xgb = bst.predict_proba(X_test)

for i in range(len(y_test)):
    print(prob_xgb[i] , prob_xgb[i].argmax() , y_test[i].argmax())

#%%
