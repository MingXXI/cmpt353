#plot_acceleration_FFT(Butterworth_filter_and_FFT(read_csv('sensor data' , '走路口袋1')))
def plot_acceleration_FFT(data_FT):
    plt.figure(figsize = (20, 15))
    plt.plot(data_FT['ax'] , 'r.' , alpha = 0.5)
    plt.title("FFT for total acceleration")

#plot_acceleration(Butterworth_filter_forplot(read_csv('falldown_hold' , 'falldown_hold1')))
#plot_acceleration(read_csv('falldown_hold' , 'falldown_hold1'))
#plot_acceleration(Butterworth_filter_forplot(read_csv('sensor data' , '上楼梯手持8')))
def plot_acceleration(data):
    '''
    General function for plot the accleration. Subplot 4 graph, x axis y axis z axis and total accleration
    '''
    plt.figure(figsize = (30, 30))
    plt.subplot(4 , 1 , 1)
    plt.plot(data['time'] , data['ax'] , 'r.' , alpha = 0.5)
    plt.title('X axis acceleration')
    plt.xlabel('Time(s)')
    plt.ylabel('Acceleration(m/s^2)')
    plt.subplot(4 , 1 , 2)
    plt.plot(data['time'] , data['ay'] , 'g.' , alpha = 0.5)
    plt.title('Y axis acceleration')
    plt.xlabel('Time(s)')
    plt.ylabel('Acceleration(m/s^2)')
    plt.subplot(4 , 1 , 3)
    plt.plot(data['time'] , data['az'] , 'b.' , alpha = 0.5)
    plt.title('Z axis acceleration')
    plt.xlabel('Time(s)')
    plt.ylabel('Acceleration(m/s^2)')
    plt.subplot(4 , 1 , 4)
    plt.plot(data['time'] , data['aT'] , 'k.' , alpha = 0.5)
    plt.title('Total acceleration')
    plt.xlabel('Time(s)')
    plt.ylabel('Acceleration(m/s^2)')


#plot_gyroscope(Butterworth_filter_forplot(read_csv('sensor data' , '摔倒手持4')))
def plot_gyroscope(data):
    '''
    General function for plot the Gyroscope. Subplot 3 graph, x axis y axis z axis
    '''
    plt.figure(figsize = (30, 30))
    plt.subplot(3 , 1 , 1)
    plt.plot(data['time'] , data['wx'] , 'r.' , alpha = 0.5)
    plt.title('X axis gyroscope')
    plt.xlabel('Time(s)')
    plt.ylabel('Gyroscope(rad/s)')
    plt.subplot(3 , 1 , 2)
    plt.plot(data['time'] , data['wy'] , 'g.' , alpha = 0.5)
    plt.title('Y axis gyroscope')
    plt.xlabel('Time(s)')
    plt.ylabel('Gyroscope(rad/s)')
    plt.subplot(3 , 1 , 3)
    plt.plot(data['time'] , data['wz'] , 'b.' , alpha = 0.5)
    plt.title('Z axis gyroscope')
    plt.xlabel('Time(s)')
    plt.ylabel('Gyroscope(rad/s)')
