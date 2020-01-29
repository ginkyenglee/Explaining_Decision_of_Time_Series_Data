import pandas as pd
import numpy as np

def readucr(filename):
    if "UWave" in filename:
        data = np.loadtxt(filename)
    else:
        data = np.loadtxt(filename, diameter=',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y
        
def load_data(data_name):
    print("Load data {}".format(data_name))
    if "EEG" in data_name:
        from scipy.io import arff
        from sklearn.model_selection import train_test_split

        data = arff.loadarff('./data/EEG-Eye-State-Dataset.arff')
        df = pd.DataFrame(data[0]).values
        X, y = df[:, 0:14], df[:,-1]

        reshapedX = []
        reshapedY = []
        for i in range(len(X)):
            if i < len(X)-117:
                reshapedX.append(X[i:i+117, :])
                reshapedY.append(y[i+117-1])

        reshapedX = np.array(reshapedX, dtype=np.float64)
        reshapedY = np.array(reshapedY, dtype=np.int64)
        reshapedY = reshapedY.reshape(reshapedY.shape[0],1)
        
        trainx, testx, trainy, testy = train_test_split(reshapedX, reshapedY, test_size=0.33, random_state=321)
        
    elif "Occumpancy" in data_name:
        from sklearn.model_selection import train_test_split
        
        data = pd.read_csv('./data/Occumpancy.txt')
        df = data.values
        X, y = df[:,2:-1], df[:,-1]

        reshapedX = []
        reshapedY = []
        for i in range(len(X)):
            if i < len(X)-117:
                reshapedX.append(X[i:i+117, :])
                reshapedY.append(y[i+117-1])

        reshapedX = np.array(reshapedX, dtype=np.float64)
        reshapedY = np.array(reshapedY, dtype=np.int64)
        reshapedY = reshapedY.reshape(reshapedY.shape[0],1)
        
        trainx, testx, trainy, testy = train_test_split(reshapedX, reshapedY, test_size=0.33, random_state=321)

    elif "Gas_sensor" in data_name:
        from sklearn.model_selection import train_test_split

        data = np.loadtxt('./data/Gas_sensor.dat', skiprows=1)

        df = pd.DataFrame(data).values
        X, y = data[:10000,1:], data[:10000,1]

        reshapedX = []
        reshapedY = []
        for i in range(len(X)):
            if i < len(X)-256:
                reshapedX.append(X[i:i+256, :])
                reshapedY.append(y[i+256-1])

        reshapedX = np.array(reshapedX, dtype=np.float64)
        reshapedY = np.array(reshapedY, dtype=np.int64)
        reshapedY = reshapedY.reshape(reshapedY.shape[0],1)

        trainx, testx, trainy, testy = train_test_split(reshapedX, reshapedY, test_size=0.33, random_state=321)

    elif "HAR" in data_name:
        from pandas import read_csv
        from numpy import dstack


        # load a single file as a numpy array
        def load_file(filepath):
            dataframe = read_csv(filepath, header=None, delim_whitespace=True)
            return dataframe.values

        # load a list of files, such as x, y, z data for a given variable
        def load_group(filenames, prefix=''):
            loaded = list()
            for name in filenames:
                data = load_file(prefix + name)
                loaded.append(data)
            # stack group so that features are the 3rd dimension
            loaded = dstack(loaded)
            return loaded

        # load a dataset group, such as train or test
        def load_dataset(group, prefix=''):
            filepath = prefix + group + '/Inertial Signals/'
            # load all 9 files as a single array
            filenames = list()
            # total acceleration
            filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
            # body acceleration
            filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
            # body gyroscope
            filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
            # load input data
            X = load_group(filenames, filepath)
            # load class output
            y = load_file(prefix + group + '/y_'+group+'.txt')
            return X, y
        
        data_path ="./data/HAR/"

        # load all train
        trainx, trainy = load_dataset('train', data_path)
        
        # load all test
        testx, testy = load_dataset('test', data_path)
        
    elif "UWaveGesture" in data_name:
        trainx, trainy = readucr('./data/'+data_name+'/'+data_name+'_TEST.txt')
        testx, testy = readucr('./data/'+data_name+'/'+data_name+'_TRAIN.txt')

    batch_size = min(int(trainx.shape[0]/10), 64)
    print ("batch size:{}".format(batch_size))    
    print("train data {},{}".format(trainx.shape, trainy.shape))
    print("test data {},{}".format(testx.shape, testy.shape)) 
    return trainx, testx,trainy,testy,batch_size



def class_breakdown(data):
    # convert the numpy array into a dataframe
    df = pd.DataFrame(data)
    # group data by the class value and calculate the number of rows
    counts = df.groupby(0).size()
    # retrieve raw rows
    counts = counts.values
    # summarize
    for i in range(len(counts)):
        percent = counts[i] / len(df) * 100
        print('Class=%d, total=%d, percentage=%.3f' % (i+1, counts[i], percent))