from keras.models import load_model
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import linregress
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import matplotlib.dates as mdates
import datetime

#df = pd.read_csv('Pgen_tmx2019_hrs.csv', engine='python' )
df = pd.read_csv('Pgen_tmx2019_min.csv', engine='python' )
#df = pd.read_csv("household_power_consumption_days.csv")
#df = pd.read_csv("feeds(7).csv")

df.head()
#features_considered = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'sub_metering_4']
features_considered = ['PVP. Gen [W]', 'C. Gen [A]', 'V_AC[V]', 'Direct [W/m^2]', 'Global [W/m^2]', 'Diffuse [W/m^2]', 'Temp. [oC]', 'Hum. [%]', 'Wind Vel. [m/s]', 'Press. [mbar]']

features = df[features_considered]
features.index = df['datetime']
#features.index = df['Date Time']
labels =['2019-01-01 01:00:00', '2019-02-01 01:00:00', '2019-03-01 01:00:00', '2019-04-01 01:00:00', '2019-05-01 01:00:00', '2019-06-01 01:00:00', '2019-07-01 01:00:00', '2019-08-01 01:00:00', '2019-09-01 01:00:00', '2019-10-01 01:00:00', '2019-11-01 01:00:00', '2019-12-01 01:00:00']
meses=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
features.head()
features.plot(subplots=True, legend=False)
[ax.legend(loc=1) for ax in plt.gcf().axes]

#ax.set_xlim('2019-01-01 00:00:00', '2020-01-01 00:00:00')
#plt.xticks(meses, labels, rotation=40)
plt.show()

#fi = 'household_power_consumption_days.csv'
fi = 'Pgen_tmx2019_min.csv'
#fi = 'Pgen_tmx2019_hrs.csv'
#fi = 'feeds(7).csv'
raw = pd.read_csv(fi, delimiter=',', engine='python' )
raw = raw.drop('datetime', axis=1)
#raw = raw.drop('Date Time', axis=1)
y=raw['PVP. Gen [W]'].to_numpy()
x=np.arange(56836)
pol = np.polyfit(x,y,10)
p3=np.polyval(pol,x)
print("raw shape:")
print (raw.shape)
print("y shape:")
print (y.shape)
print("x shape:")
print (x.shape)
print("pol_reg shape:")
print (p3.shape)

#plt.plot(p3, label='Pol.Reg.',linewidth=1)
#plt.show()

scaler = MinMaxScaler(feature_range=(-1, 1))
raw = scaler.fit_transform(raw)

time_shift = 1 #shift is the number of steps we are predicting ahead
n_rows = raw.shape[0] #n_rows is the number of time steps of our sequence
n_feats = raw.shape[1]
train_size = int(n_rows * 0.8)

train_data = raw[:train_size, :] #first train_size steps, all 5 features
test_data = raw[train_size:, :] #I'll use the beginning of the data as state adjuster

x_train = train_data[:-time_shift, :] #the entire train data, except the last shift steps
x_test = test_data[:-time_shift,:] #the entire test data, except the last shift steps
x_predict = raw[:-time_shift,:] #the entire raw data, except the last shift steps

y_train = train_data[time_shift:, :]
y_test = test_data[time_shift:,:]
y_predict_true = raw[time_shift:,:]

x_train = x_train.reshape(1, x_train.shape[0], x_train.shape[1]) #ok shape (1,steps,5) - 1 sequence, many steps, 5 features
y_train = y_train.reshape(1, y_train.shape[0], y_train.shape[1])
x_test = x_test.reshape(1, x_test.shape[0], x_test.shape[1])
y_test = y_test.reshape(1, y_test.shape[0], y_test.shape[1])
x_predict = x_predict.reshape(1, x_predict.shape[0], x_predict.shape[1])
y_predict_true = y_predict_true.reshape(1, y_predict_true.shape[0], y_predict_true.shape[1])

print("\nx_train:")
print (x_train.shape)
print("y_train")
print (y_train.shape)
print("x_test")
print (x_test.shape)
print("y_test")
print (y_test.shape)

#model = tf.keras.models.load_model('CNN_LSTM5.h5')
model_A = tf.keras.models.load_model('CNN_LSTM_PV3.h5')
model_B = tf.keras.models.load_model('CNN_PV.h5')
model_C = tf.keras.models.load_model('LSTM_PV.h5')
#model = tf.keras.models.load_model('CNN_PV.h5')
#model = tf.keras.models.load_model('LSTM.h5')
#model = tf.keras.models.load_model('LSTM_solar.h5')
y_predict_model = model_A.predict(x_predict)
y_predict_model2 = model_A.predict(x_test)
y_predict_model3 = model_A.predict(x_train)

p2 = model_C.predict(x_predict)
p2_2 = model_C.predict(x_test)
p2_3 = model_C.predict(x_train)

pB = model_B.predict(x_predict)
pB_2 = model_B.predict(x_test)
pB_3 = model_B.predict(x_train)

print("\ny_predict_true:")
print (y_predict_true.shape)
print("y_predict_model_global: ")
print (y_predict_model.shape)
print("y_predict_model_validation: ")
print (y_predict_model2.shape)
print("y_predict_model_train: ")
print (y_predict_model3.shape)


def plot(true, predicted, divider, predicted2, predicted3):

    predict_plot3 = scaler.inverse_transform(predicted2[0])
    predict_plot2 = scaler.inverse_transform(predicted3[0])
    predict_plot = scaler.inverse_transform(predicted[0])
    true_plot = scaler.inverse_transform(true[0])

    predict_plot = predict_plot[:,0]
    predict_plot2 = predict_plot2[:,0]
    predict_plot3 = predict_plot3[:,0]
    true_plot = true_plot[:,0]

    plt.figure(figsize=(16,6))

    #plt.plot(true_plot, label='True',linewidth=1)
    plt.plot(true_plot, label='True PVPG',linewidth=1)
    plt.plot(predict_plot,  label='CNN_LSTM_5',color='y',linewidth=1)
    plt.plot(predict_plot2, label='CNN_LSTM_2',linewidth=1)
    plt.plot(predict_plot3, label='LSTM',linewidth=1)
    if divider > 0:
        maxVal = max(true_plot.max(),predict_plot.max())
        minVal = min(true_plot.min(),predict_plot.min())

        plt.plot([divider,divider],[minVal,maxVal],label='train/test limit',color='k')
        #plt.plot([divider,divider],[minVal,maxVal],label='lim entr/val',color='k')


    plt.ylabel('Power generated [W]')
    plt.xlabel('Time [/10min]')
    plt.legend()
    plt.show()
def plot2(true, predicted, divider, predicted2, predicted3):

    predict_plot3 = scaler.inverse_transform(predicted2[0])
    predict_plot2 = scaler.inverse_transform(predicted3[0])
    predict_plot = scaler.inverse_transform(predicted[0])
    true_plot = scaler.inverse_transform(true[0])

    predict_plot = predict_plot[:,0]
    predict_plot2 = predict_plot2[:,0]
    predict_plot3 = predict_plot3[:,0]
    true_plot = true_plot[:,0]

    plt.figure(figsize=(16,6))
    plt.plot(true_plot, label='True',linewidth=1)
    plt.plot(predict_plot,  label='CNN_LSTM_5',color='y',linewidth=1)
    plt.plot(predict_plot2, label='CNN_LSTM_2',linewidth=1)
    plt.plot(predict_plot3, label='LSTM',linewidth=1)

    if divider > 0:
        maxVal = max(true_plot.max(),predict_plot.max())
        minVal = min(true_plot.min(),predict_plot.min())

        #plt.plot([divider,divider],[minVal,maxVal],label='train/test limit',color='k')

    plt.legend()
    plt.show()

test_size = n_rows - train_size
print("test length: " + str(test_size))

#print("-------------------------------MSE------------------------------------------------")
mse = np.square(np.subtract(y_predict_true,y_predict_model)).mean()
mse2 = np.square(np.subtract(y_test,y_predict_model2)).mean()
mse3 = np.square(np.subtract(y_train,y_predict_model3)).mean()

mse_B = np.square(np.subtract(y_predict_true,p2)).mean()
mse2_B = np.square(np.subtract(y_test,p2_2)).mean()
mse3_B = np.square(np.subtract(y_train,p2_3)).mean()

mse_C = np.square(np.subtract(y_predict_true,pB)).mean()
mse2_C = np.square(np.subtract(y_test,pB_2)).mean()
mse3_C = np.square(np.subtract(y_train,pB_3)).mean()
#print("-------------------------------RMSE---------------------------------------------")
rmse = np.sqrt(mse)
rmse2 = np.sqrt(mse2)
rmse3 = np.sqrt(mse3)

rmse_B = np.sqrt(mse_B)
rmse2_B = np.sqrt(mse2_B)
rmse3_B = np.sqrt(mse3_B)

rmse_C = np.sqrt(mse_C)
rmse2_C = np.sqrt(mse2_C)
rmse3_C = np.sqrt(mse3_C)
#print("-------------------------------MAE------------------------------------------------")
mae = np.abs(np.subtract(y_predict_true,y_predict_model)).mean()
mae2 = np.abs(np.subtract(y_test,y_predict_model2)).mean()
mae3 = np.abs(np.subtract(y_train,y_predict_model3)).mean()

mae_B = np.abs(np.subtract(y_predict_true,p2)).mean()
mae2_B = np.abs(np.subtract(y_test,p2_2)).mean()
mae3_B = np.abs(np.subtract(y_train,p2_3)).mean()

mae_C = np.abs(np.subtract(y_predict_true,pB)).mean()
mae2_C = np.abs(np.subtract(y_test,pB_2)).mean()
mae3_C = np.abs(np.subtract(y_train,pB_3)).mean()
#print("--------------------------------MSE-----------------------------------------------")
print("MSE metrics for CNN_LSTM_5 model:")
print("MSE validation: " + str(mse2))
print("MSE train: " + str(mse3))
print("MSE global: " + str(mse))
print("-------------------------------------------------------------------------------")
print("MSE metrics for CNN_LSTM_2 model:")
print("MSE validation: " + str(mse2_C))
print("MSE train: " + str(mse3_C))
print("MSE global: " + str(mse_C))
print("-------------------------------------------------------------------------------")
print("MSE metrics for LSTM model:")
print("MSE validation: " + str(mse2_B))
print("MSE train: " + str(mse3_B))
print("MSE global: " + str(mse_B))
print("-------------------------------------------------------------------------------")

#print("--------------------------------RMSE-----------------------------------------------")
print("RMSE metrics for CNN_LSTM_5 model:")
print("RMSE validation: " + str(rmse2))
print("RMSE train: " + str(rmse3))
print("RMSE global: " + str(rmse))
print("-------------------------------------------------------------------------------")
print("RMSE metrics for CNN_LSTM_2 model:")
print("RMSE validation: " + str(rmse2_C))
print("RMSE train: " + str(rmse3_C))
print("RMSE global: " + str(rmse_C))
print("-------------------------------------------------------------------------------")
print("RMSE metrics for LSTM model:")
print("RMSE validation: " + str(rmse2_B))
print("RMSE train: " + str(rmse3_B))
print("RMSE global: " + str(rmse_B))
print("-------------------------------------------------------------------------------")

#print("--------------------------------MAE-----------------------------------------------")
print("MAE metrics for CNN_LSTM_5 model:")
print("MAE validation: " + str(mae2))
print("MAE train: " + str(mae3))
print("MAE global: " + str(mae))
print("-------------------------------------------------------------------------------")
print("MAE metrics for CNN_LSTM_2 model:")
print("MAE validation: " + str(mae2_C))
print("MAE train: " + str(mae3_C))
print("MAE global: " + str(mae_C))
print("-------------------------------------------------------------------------------")
print("MAE metrics for LSTM model:")
print("MAE validation: " + str(mae2_B))
print("MAE train: " + str(mae3_B))
print("MAE global: " + str(mae_B))
print("-------------------------------------------------------------------------------")

plot(y_predict_true,y_predict_model,train_size,p2,pB)
plot(y_predict_true[:,-2*test_size:],y_predict_model[:,-2*test_size:],test_size,p2[:,-2*test_size:],pB[:,-2*test_size:])
plot2(y_test,y_predict_model2,test_size,p2_2,pB_2)
