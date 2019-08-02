import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from math import sqrt
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 10000)



''' 
    modified read_csv
    example: 
    data = read_csv('falldown_hold', 'falldown_hold1')
'''
def read_csv(directory_name , fileName):
    '''
    Read the file from a directory given directory name and file name, we collected all the data in one directory 
    '''
    read_file = 'sensor data/' + directory_name + '/' + fileName + '.csv'
    df = pd.read_csv(read_file)                                           # Create a DataFrame for return value 
    del df['Unnamed: 7']                                                  # delete unknown columns to make DataFrame clean  
    df = df[ (df['time'] >= 3) & (df['time'] <= 6) ]                      # Only included time from 5s to 15s
    df['aT'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)           # get total acceleration
    return df                                                             # return the DataFrame


'''
    clean a serire of data
    example:
    data['ax']=Butterworth_filter(data['ax'])
'''
def Butterworth_filter(data):
    '''
    Low-pass: keep the low frequencies; discard the high.
    High-pass: keep the high frequencies; discard the low.
    '''
    b, a = signal.butter(3, 0.05, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, data)
    return low_passed

'''
    clean a data frame
    example:
    data = Butterworth_filter_forplot(data)
'''
def Butterworth_filter_forplot(data):
    '''
    Given a dataFrame , apply Butterworth filter for each column 
    '''
    data = data.apply(Butterworth_filter , axis = 0)
    return data


# citation: https://github.com/philip-L/CMPT353-1/blob/master/analysis2.py
# Figure out what happening here, write our own FFT function and report must mention why we do this 
def Butterworth_filter_and_FFT(data):
    # Using the Butterworth filter
    data_bw = data.apply(Butterworth_filter , axis = 0)
    data_bw = data_bw.reset_index(drop = True)
    # del data_bw['time']
    data = data.reset_index(drop = True)
    # FFT of the data after the Butterworth filter
    data_FT = data_bw.apply(np.fft.fft , axis = 0)      
    data_FT = data_FT.apply(np.fft.fftshift , axis = 0)
    data_FT = data_FT.abs()
    
    # Determine the sampling frequency
    Fs = round(len(data) / data.at[len(data)-1, 'time']) #samples per second
    data_FT['freq'] = np.linspace(-Fs/2, Fs/2, num = len(data))
    
    # Find the largest peak at a frequency greater than 0 to determine the average steps per second
    temp_FT = data_FT[data_FT['freq'] > 0.1]
    ind = temp_FT['aT'].nlargest(n = 1)
    max_ind = ind.idxmax()
    avg_freq = data_FT.at[max_ind , 'freq']
    
    #Transform the data to fit a normal distribution
    max_val = data_FT['aT'].nlargest(n = 1)
    max_val_ind = max_val.idxmax()
    data_FT.at[max_val_ind , 'aT'] = temp_FT['aT'].max()
    
    return data_FT , avg_freq


#Butterworth_filter_and_FFT(read_csv('sensor data' , '上楼梯口袋1'))


'''
    pass a data get summary,
    stat_summary=
    [ax_mean, ax_std, ax_min, ax_25, ax_50, ax_75, ax_max,
    ay_mean, ay_std, ay_min, ay_25, ay_50, ay_75, ay_max,
    az_mean, az_std, az_min, az_25, az_50, az_75, az_max,
    wx_mean, wx_std, wx_min, wx_25, wx_50, wx_75, wx_max,
    wy_mean, wy_std, wy_min, wy_25, wy_50, wy_75, wy_max,
    wz_mean, wz_std, wz_min, wz_25, wz_50, wz_75, wz_max,
    aT_mean, aT_std, aT_min, aT_25, aT_50, aT_75, aT_max]
    example:
    summary=get_basic_feature(read_csv('downstairs_hold' , 'downstairs_hold1'))
'''
def get_basic_feature(data):
    # The parameter will be the original dataFrame after some data cleaning
    '''
    ax , ay , az , wx , wy , wz , aT
    mean        0.379203
    std         2.659466
    min       -11.236750
    25%        -0.963552
    50%         0.422153
    75%         1.849594
    max         9.068970
    Get the basic statistical feature for each direction of acceleration and gyrpscope
    .describe will give us mean, std, min, 25%, 50%, 75%, max value. All of these are basic feature we need give it to Machine Learning
    '''
    stat_summary = []
    ax_stat_summary = data['ax'].describe()      # .describe get the basic feature 
    for i in range(1 , 8):
        stat_summary.append(ax_stat_summary[i])  # each feature append it to the list  
    ay_stat_summary = data['ay'].describe()
    for i in range(1 , 8):
        stat_summary.append(ay_stat_summary[i])
    az_stat_summary = data['az'].describe()
    for i in range(1 , 8):
        stat_summary.append(az_stat_summary[i]) 
    wx_stat_summary = data['wx'].describe()
    for i in range(1 , 8):
        stat_summary.append(wx_stat_summary[i]) 
    wy_stat_summary = data['wy'].describe()
    for i in range(1 , 8):
        stat_summary.append(wy_stat_summary[i]) 
    wz_stat_summary = data['wz'].describe()
    for i in range(1 , 8):
        stat_summary.append(wz_stat_summary[i]) 
    aT_stat_summary = data['aT'].describe()
    for i in range(1 , 8):
        stat_summary.append(aT_stat_summary[i]) 

    return stat_summary # return a large list that given a dataFrame, return all the basic feature


#get_basic_feature(Butterworth_filter_forplot(read_csv('sensor data' , '上楼梯口袋1')))




'''
    a new feature we thought other than max, mean, min...
    however after testing this feature is useless.
'''
def get_acceleration_slope_max(data_col):
    data_shift = data_col.shift(periods = -1 , fill_value = 0)
    data_difference = abs(data_col - data_shift) 
    data_slope = abs(data_difference / data_col)
    data_slope = data_slope[:-1]
    return data_slope.max()

#get_acceleration_slope_max(Butterworth_filter_forplot(read_csv('sensor data' , '走路口袋10'))['aT'])



'''
    final function to get basic feature by using butterworth
    example:
    data_feature_bw=get_basic_feature_butterworth(read_csv('downstairs_hold', 'downstairs_hold1'))
'''
def get_basic_feature_butterworth(data):
    data_bw = Butterworth_filter_forplot(data)
    data_feature = get_basic_feature(data_bw)
    '''
    ax_slope_max = get_acceleration_slope_max(data['ax'])
    ay_slope_max = get_acceleration_slope_max(data['ay'])
    az_slope_max = get_acceleration_slope_max(data['az'])
    wx_slope_max = get_acceleration_slope_max(data['wx'])
    wy_slope_max = get_acceleration_slope_max(data['wy'])
    wz_slope_max = get_acceleration_slope_max(data['wz'])
    aT_slope_max = get_acceleration_slope_max(data['aT'])

    data_feature.append(ax_slope_max)
    data_feature.append(ay_slope_max)
    data_feature.append(az_slope_max)
    data_feature.append(wx_slope_max)
    data_feature.append(wy_slope_max)
    data_feature.append(wz_slope_max)
    data_feature.append(aT_slope_max)
    '''
    return data_feature


'''
    automic get all the data and data category, combine them to df.
'''
def get_feature_dataFrame():
    feature_list = []
    for i in range(1 , 16):
        data = read_csv('downstairs_hold' , 'downstairs_hold' + str(i))
        data_feature = get_basic_feature_butterworth(data)
        data_feature.append('downstairs_hold')
        feature_list.append(data_feature)

    for i in range(1 , 16):
        data = read_csv('downstairs_inpocket' , 'downstairs_inpocket' + str(i))
        data_feature = get_basic_feature_butterworth(data)
        data_feature.append('downstairs_inpocket')
        feature_list.append(data_feature)

    for i in range(1 , 16):
        data = read_csv('upstairs_inpocket' , 'upstairs_inpocket' + str(i))
        data_feature = get_basic_feature_butterworth(data)
        data_feature.append('unstairs_inpocket')
        feature_list.append(data_feature)

    for i in range(1 , 16):
        data = read_csv('upstairs_hold' , 'upstairs_hold' + str(i))
        data_feature = get_basic_feature_butterworth(data)
        data_feature.append('upstairs_hold')
        feature_list.append(data_feature)

    for i in range(1 , 16):
        data = read_csv('walk_hold' , 'walk_hold' + str(i))
        data_feature = get_basic_feature_butterworth(data)
        data_feature.append('walk_hold')
        feature_list.append(data_feature)

    for i in range(1 , 16):
        data = read_csv('walk_inpocket' , 'walk_inpocket' + str(i))
        data_feature = get_basic_feature_butterworth(data)
        data_feature.append('walk_inpocket')
        feature_list.append(data_feature)

    for i in range(1 , 16):
        data = read_csv('falldown_hold' , 'falldown_hold' + str(i))
        data_feature = get_basic_feature_butterworth(data)
        data_feature.append('falldown_hold')
        feature_list.append(data_feature)   

    '''
    ax , ay , az , wx , wy , wz , aT
    mean        0.379203
    std         2.659466
    min       -11.236750
    25%        -0.963552
    50%         0.422153
    75%         1.849594
    max         9.068970
    '''
    '''
    'ax_slope_max' , 'ay_slope_max' , 'az_slope_max' , 'wx_slope_max' , 'wy_slope_max' , 
                   'wz_slope_max' , 'aT_slope_max' , 'category'
    '''
    column_name = ['ax_mean' , 'ax_std' , 'ax_min' , 'ax_25' , 'ax_50' , 'ax_75' , 'ax_max',
                   'ay_mean' , 'ay_std' , 'ay_min' , 'ay_25' , 'ay_50' , 'ay_75' , 'ay_max',
                   'az_mean' , 'az_std' , 'az_min' , 'az_25' , 'az_50' , 'az_75' , 'az_max',
                   'wx_mean' , 'wx_std' , 'wx_min' , 'wx_25' , 'wx_50' , 'wx_75' , 'wx_max',
                   'wy_mean' , 'wy_std' , 'wy_min' , 'wy_25' , 'wy_50' , 'wy_75' , 'wy_max',
                   'wz_mean' , 'wz_std' , 'wz_min' , 'wz_25' , 'wz_50' , 'wz_75' , 'wz_max',
                   'aT_mean' , 'aT_std' , 'aT_min' , 'aT_25' , 'aT_50' , 'aT_75' , 'aT_max',
                   'catogary']
    df = pd.DataFrame(feature_list , columns = column_name)
    df.to_csv('feature_df.csv')
    return df

'''
    automatic read all data and get X.
    X=
    [[downstairs_hold1_features], [downstairs_inpocket1_features], [upstairs_hold1_features],
    [upstairs_inpocket1_features], [walk_hold1_features], [walk_inpocket1_features], [falldown_hold1_features]
    [downstairs_hold2_features], ...
    [falldown_hold15_features]]
'''
def get_X():
    X = []
    for i in range(1 , 16):
        X.append(get_basic_feature_butterworth(read_csv('downstairs_hold' , 'downstairs_hold' + str(i))))
        X.append(get_basic_feature_butterworth(read_csv('downstairs_inpocket' , 'downstairs_inpocket' + str(i))))
        X.append(get_basic_feature_butterworth(read_csv('upstairs_hold' , 'upstairs_hold' + str(i))))      
        X.append(get_basic_feature_butterworth(read_csv('upstairs_inpocket' , 'upstairs_inpocket' + str(i))))
        X.append(get_basic_feature_butterworth(read_csv('walk_hold' , 'walk_hold' + str(i))))
        X.append(get_basic_feature_butterworth(read_csv('walk_inpocket' , 'walk_inpocket' + str(i))))
        X.append(get_basic_feature(read_csv('falldown_hold' , 'falldown_hold' + str(i))))
        X.append(get_basic_feature(read_csv('falldown_inpocket' , 'falldown_inpocket' + str(i))))
    return X  

def get_X_with_butt():
    X = []
    for i in range(1 , 16):
        X.append(get_basic_feature_butterworth(read_csv('downstairs_hold' , 'downstairs_hold' + str(i))))
        X.append(get_basic_feature_butterworth(read_csv('downstairs_inpocket' , 'downstairs_inpocket' + str(i))))
        X.append(get_basic_feature_butterworth(read_csv('upstairs_hold' , 'upstairs_hold' + str(i))))      
        X.append(get_basic_feature_butterworth(read_csv('upstairs_inpocket' , 'upstairs_inpocket' + str(i))))
        X.append(get_basic_feature_butterworth(read_csv('walk_hold' , 'walk_hold' + str(i))))
        X.append(get_basic_feature_butterworth(read_csv('walk_inpocket' , 'walk_inpocket' + str(i))))
        X.append(get_basic_feature_butterworth(read_csv('falldown_hold' , 'falldown_hold' + str(i))))
        X.append(get_basic_feature(read_csv('falldown_inpocket' , 'falldown_inpocket' + str(i))))
    return X 

def get_X_orig():
    X = []
    for i in range(1 , 16):
        X.append(get_basic_feature(read_csv('downstairs_hold' , 'downstairs_hold' + str(i))))
        X.append(get_basic_feature(read_csv('downstairs_inpocket' , 'downstairs_inpocket' + str(i))))
        X.append(get_basic_feature(read_csv('upstairs_hold' , 'upstairs_hold' + str(i))))      
        X.append(get_basic_feature(read_csv('upstairs_inpocket' , 'upstairs_inpocket' + str(i))))
        X.append(get_basic_feature(read_csv('walk_hold' , 'walk_hold' + str(i))))
        X.append(get_basic_feature(read_csv('walk_inpocket' , 'walk_inpocket' + str(i))))
        X.append(get_basic_feature(read_csv('falldown_hold' , 'falldown_hold' + str(i))))
        X.append(get_basic_feature(read_csv('falldown_inpocket' , 'falldown_inpocket' + str(i))))
    return X 

def get_y():
    y = []
    for i in range(1, 16):
        y.append('downstairs_hold')
        y.append('downstairs_inpocket')
        y.append('upstairs_hold')
        y.append('upstairs_inpocket')
        y.append('walk_hold')
        y.append('walk_inpocket')
        y.append('falldown_hold')
        y.append('falldown_inpocket')
    return y


#get_feature_dataFrame()


def build_test_data(directory_name , fileName):
    '''
    Some time we don't just want the predict score, we want to know given an input data, what will the Machine Learning
    exactly give us. So this piece of code is build the test data. Also we collect some test data. 
    '''
    test_data = pd.read_csv(directory_name + '/' + fileName + '.csv')
    
    del test_data['Unnamed: 7']                                                             # Delete unknown columns to make DataFrame clean  
    test_data = test_data[ (test_data['time'] >= 3) & (test_data['time'] <= 6) ]            # Only included time from 5s to 15s
    test_data['aT'] = np.sqrt(test_data['ax']**2 + test_data['ay']**2 + test_data['az']**2) # Get total acceleration
    feature_list = []
    feature = get_basic_feature(test_data)
    feature_list.append(feature)
    column_name = ['ax_mean' , 'ax_std' , 'ax_min' , 'ax_25' , 'ax_50' , 'ax_75' , 'ax_max',
               'ay_mean' , 'ay_std' , 'ay_min' , 'ay_25' , 'ay_50' , 'ay_75' , 'ay_max',
               'az_mean' , 'az_std' , 'az_min' , 'az_25' , 'az_50' , 'az_75' , 'az_max',
               'wx_mean' , 'wx_std' , 'wx_min' , 'wx_25' , 'wx_50' , 'wx_75' , 'wx_max',
               'wy_mean' , 'wy_std' , 'wy_min' , 'wy_25' , 'wy_50' , 'wy_75' , 'wy_max',
               'wz_mean' , 'wz_std' , 'wz_min' , 'wz_25' , 'wz_50' , 'wz_75' , 'wz_max',
               'aT_mean' , 'aT_std' , 'aT_min' , 'aT_25' , 'aT_50' , 'aT_75' , 'aT_max',]
               
    return feature_list


#build_test_data('Test predict data' , 'downstairs_hold_test')


OUTPUT_TEMPLATE = (
    'Bayesian classifier: {bayes_rgb:.3g} {bayes_lab:.3g}\n'
    'kNN classifier:      {knn_rgb:.3g} {knn_lab:.3g}\n'
    'SVM classifier:      {svm_rgb:.3g} {svm_lab:.3g}\n'
)

def ML_tools():
    # X = get_X_orig()
    # X = get_X_with_butt()
    X = get_X()
    y = get_y()
    X_train , X_valid , y_train , y_valid = train_test_split(
        X , y 
    )
    

    bayes_model = GaussianNB()
    bayes_model.fit(X_train , y_train)
    score=bayes_model.score(X_valid, y_valid)
    return score
    # print(score)
    # X_test = build_test_data()
    # print(bayes_model.predict(X_test))
# ave=0
# for i in range(30):
#     ave+=ML_tools()
# ave/=30
# print(ave)



