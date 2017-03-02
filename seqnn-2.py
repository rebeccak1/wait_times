import numpy
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import seq2seq
from matplotlib.dates import date2num

# fix random seed for reproducibility
numpy.random.seed(7)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, look_forward=1):
    dataX, dataY = [], []
    #for i in range(len(dataset)-look_back-1):
    for i in range(len(dataset)-look_back-look_forward):
	a = dataset[i:(i+look_back), 0]
	dataX.append(a)
	b = dataset[(i+look_back):(i+look_back+look_forward), 0]
	dataY.append(b)
    print i

    return numpy.array(dataX), numpy.array(dataY)


if __name__ == '__main__':
    dateparse = lambda dates: datetime.datetime.strptime(dates, '%Y-%m-%d\t%H:%M:%S')

    df = pd.read_csv('data-1-17.txt', parse_dates=0, index_col=0,date_parser=dateparse, delimiter=',')
    dataset = df['Waits']
    dataset = df.values
    dataset = dataset.astype('float32')
    dates = df.index
    #s = pd.Series(dataset, index=df.index)
    
    print dataset
    print dataset.shape

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print train.shape, test.shape, 'split sizes'
    
    '''
    train_dates, test_dates = numpy.zeros([len(dates)]), numpy.zeros([len(dates)])
    #train_dates[:], test_dates[:] = numpy.nan, numpy.nan

    train_dates[0:train_size], test_dates[train_size:len(dataset)] = numpy.array(dates[0:train_size].to_pydatetime()), numpy.array(dates[train_size:len(dataset)].to_pydatetime(), dtype=numpy.datetime64)
    '''

    # reshape into X=t and Y=t+1
    look_back = 288
    look_forward = 96
    

    '''
    input_list = [numpy.expand_dims(numpy.atleast_2d(train[i:look_back+i,0]),axis=0) for i in xrange(len(train) - look_back - look_forward)]
    trainX = numpy.concatenate(input_list, axis=0)
    target_list = [numpy.atleast_2d(train[i+look_back:i+look_back+look_forward,0]) for i in xrange(len(train)-look_back-look_forward)]
    trainY = numpy.concatenate(target_list,axis=0)

    input_list = [numpy.expand_dims(numpy.atleast_2d(test[i:look_back+i,0]),axis=0) for i in xrange(len(test)-look_back - look_forward)]
    testX = numpy.concatenate(input_list, axis=0)
    target_list = [numpy.atleast_2d(test[i+look_back:i+look_back+look_forward,0]) for i in xrange(len(test)-look_back-look_forward)]
    testY = numpy.concatenate(target_list,axis=0)
    '''


    trainX, trainY = create_dataset(train, look_back, look_forward)
    testX, testY = create_dataset(test, look_back, look_forward)

    print trainX.shape, trainY.shape, 'train X shape train Y shape'

    #reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    trainY = numpy.reshape(trainY, (trainY.shape[0], look_forward))
    testY = numpy.reshape(testY, (testY.shape[0], look_forward))

    print trainX.shape, trainY.shape, "training shape" 
    print testX.shape, testY.shape, "testing shape"

    #create and fit the LSTM network
    model = Sequential()
    hidden = 128

    model.add(LSTM(input_dim=look_back, output_dim=hidden, activation='sigmoid'))
    model.add(Dense(input_dim=hidden, output_dim=look_forward))
    model.add(Activation('linear'))
    

    #compile model
    #model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    hist = model.fit(trainX, trainY, nb_epoch=200, batch_size=128, verbose=2)
    print (hist.history)

    # evaluate the model
    scores = model.evaluate(trainX, trainY, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    '''
# serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    '''

    #make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    print "train predict shape {}".format(trainPredict.shape)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    print "test predict shape {}".format(testPredict.shape)
    print "testY shape {}".format(testY.shape)

    # calculate root mean squared error

    print trainY.shape, trainPredict.shape
    print trainY[:,0].shape, trainPredict[:,0].shape
    trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))


    '''
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    '''

    trainPredict1 =trainPredict[:,0]
    trainPredict11 = trainPredict1.reshape(trainPredict.shape[0],1)
    print trainPredict11.shape

    trainPredict2 = trainPredict[:,1]
    trainPredict12 = trainPredict2.reshape(trainPredict.shape[0],1)
    print trainPredict12.shape

    # shift train predictions for plotting
    trainPredictPlot1 = numpy.empty_like(dataset)
    trainPredictPlot1[:, :] = numpy.nan
    trainPredictPlot1[look_back:len(trainPredict)+look_back, :] = trainPredict11
    
    #shift train t+2 predictions for plotting
    trainPredictPlot2 = numpy.empty_like(dataset)
    trainPredictPlot2[:, :] = numpy.nan
    trainPredictPlot2[look_back:len(trainPredict)+look_back, :] = trainPredict12
    #############################################
    testPredict1 = testPredict[:,0]
    print testPredict1.shape, 'testpredict1 shape'
    testPredict11 = testPredict1.reshape(testPredict.shape[0],1)
    print testPredict11.shape, 'testpredict11 shape'
    testPredict2 = testPredict[:,1]
    testPredict12 = testPredict2.reshape(testPredict.shape[0],1)
    print testPredict1.shape, 'test predict shape', train_size, 'train_size'
    print len(trainPredict)+look_back*2+1, look_back+train_size
    
    # shift test predictions for plotting
    testPredictPlot1 = numpy.empty_like(dataset)
    testPredictPlot1[:, :] = numpy.nan
    #testPredictPlot1[len(trainPredict)+(look_back*2)+1:len(trainPredict)+(look_back*2)+1+len(testPredict), :] = testPredict11
    testPredictPlot1[look_back+train_size:look_back+train_size+len(testPredict), :] = testPredict11
    
    # shift test predictions for plotting
    testPredictPlot2 = numpy.empty_like(dataset)
    testPredictPlot2[:, :] = numpy.nan
    testPredictPlot2[len(trainPredict)+(look_back*2)+1:len(trainPredict)+(look_back*2)+1+len(testPredict), :] = testPredict12
    
    # plot baseline and predictions
    testdata = numpy.empty_like(dataset)
    testdata[:,:] = numpy.nan
    testdata[train_size:len(dataset),:] = scaler.inverse_transform(dataset[train_size:len(dataset),:])

    traindata = numpy.empty_like(dataset)
    traindata[:,:] = numpy.nan
    traindata[0:train_size,:] = scaler.inverse_transform(dataset[0:train_size,:])
    print len(trainPredict), train_size, 'predictsize, train size'

    l1, = plt.plot_date(dates[train_size:len(dataset)],testdata[train_size:len(dataset)],'b-', c='seagreen')
    l4, = plt.plot_date(dates[train_size:len(dataset)],testPredictPlot1[train_size:len(dataset)], 'b-',c='mediumspringgreen')

    l6, = plt.plot_date(dates[0:train_size],traindata[0:train_size],'b-',c='mediumblue')
    l2, = plt.plot_date(dates[0:train_size],trainPredictPlot1[0:train_size],'b-',c='dodgerblue')

    print dates[train_size+4436+96:96*2+train_size+4436].shape, trainPredict[-1,:].shape
    l7, = plt.plot_date(dates[train_size+4436+96:train_size+4436+96*2],trainPredict[-1,:],'b-',c='red')
    #l3, = plt.plot(trainPredictPlot2)

    l5, = plt.plot_date(dates[train_size:len(dataset)],testPredictPlot2[train_size:len(dataset)], 'b-',c='orange')
    #plt.legend([l1,l2,l5],['Data','trainPredict 1','train Predict 2', 'test predict 1', 'test predict 2'])
    
    plt.legend([l1,l6,l2,l4,l5,l7],['Test Data','Train Data', 'Train Predict 1','Test Predict 1', 'Test Predict 2'])
    
    plt.show()
