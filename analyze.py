from Cleaning_Data import *
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

x = range(49)
y = get_X()

'''
	difference between use Butterworth or not plot
'''
def original_data(directory_name , fileName):
	data=read_csv(directory_name , fileName)
	plt.figure(figsize = (15, 15))
	plt.subplot(7 , 1 , 1)
	plt.plot(data['time'] , data['ax'] , 'b.' , alpha = 0.5)
	plt.title('X axis acceleration')
	plt.xlabel('Time(s)')
	plt.ylabel('Acceleration(m/s^2)')
	plt.subplot(7 , 1 , 2)
	plt.plot(data['time'] , data['ay'] , 'b.' , alpha = 0.5)
	plt.title('Y axis acceleration')
	plt.xlabel('Time(s)')
	plt.ylabel('Acceleration(m/s^2)')
	plt.subplot(7 , 1 , 3)
	plt.plot(data['time'] , data['az'] , 'b.' , alpha = 0.5)
	plt.title('Z axis acceleration')
	plt.xlabel('Time(s)')
	plt.ylabel('Acceleration(m/s^2)')
	plt.subplot(7 , 1 , 4)
	plt.plot(data['time'] , data['wx'] , 'b.' , alpha = 0.5)
	plt.title('X axis gyroscope')
	plt.xlabel('Time(s)')
	plt.ylabel('Gyroscope(rad/s)')
	plt.subplot(7 , 1 , 5)
	plt.plot(data['time'] , data['wy'] , 'b.' , alpha = 0.5)
	plt.title('Y axis gyroscope')
	plt.xlabel('Time(s)')
	plt.ylabel('Gyroscope(rad/s)')
	plt.subplot(7 , 1 , 6)
	plt.plot(data['time'] , data['wz'] , 'b.' , alpha = 0.5)
	plt.title('Z axis gyroscope')
	plt.xlabel('Time(s)')
	plt.ylabel('Gyroscope(rad/s)')
	plt.subplot(7 , 1 , 7)
	plt.plot(data['time'] , data['aT'] , 'b.' , alpha = 0.5)
	plt.title('Total acceleration')
	plt.xlabel('Time(s)')
	plt.ylabel('Acceleration(m/s^2)')
	plt.savefig(fileName+'.png')

def Original_vs_butterworth():
	i='6'		#change data set.

	#compare downstair_hold
	orig=get_basic_feature(read_csv('downstairs_hold' , 'downstairs_hold'+i))
	butt=get_basic_feature_butterworth(read_csv('downstairs_hold' , 'downstairs_hold'+i))
	plt.figure(figsize = (15, 10))
	plt.subplot(2, 1, 1)
	plt.plot(x, orig, 'b-' , alpha = 0.5)
	plt.title('downstairs_hold_original'+i)
	plt.subplot(2 , 1 , 2)
	plt.plot(x, butt, 'b-' , alpha = 0.5)
	plt.title('downstairs_hold_butterworth'+i)
	plt.savefig('Butterworth_dh'+i+'.png')

	#compare downstairs_inpocket
	orig=get_basic_feature(read_csv('downstairs_inpocket' , 'downstairs_inpocket'+i))
	butt=get_basic_feature_butterworth(read_csv('downstairs_inpocket' , 'downstairs_inpocket'+i))
	plt.figure(figsize = (15, 10))
	plt.subplot(2, 1, 1)
	plt.plot(x, orig, 'b-' , alpha = 0.5)
	plt.title('downstairs_inpocket_original'+i)
	plt.subplot(2, 1, 2)
	plt.plot(x, butt, 'b-' , alpha = 0.5)
	plt.title('downstairs_inpocket_butterworth'+i)

	#compare upstairs_hold
	orig=get_basic_feature(read_csv('upstairs_hold' , 'upstairs_hold'+i))
	butt=get_basic_feature_butterworth(read_csv('upstairs_hold' , 'upstairs_hold'+i))
	plt.figure(figsize = (15, 10))
	plt.subplot(2, 1, 1)
	plt.plot(x, orig, 'b-' , alpha = 0.5)
	plt.title('upstairs_hold_original'+i)
	plt.subplot(2, 1, 2)
	plt.plot(x, butt, 'b-' , alpha = 0.5)
	plt.title('upstairs_hold_butterworth'+i)

	#compare upstairs_inpocket
	orig=get_basic_feature(read_csv('upstairs_inpocket' , 'upstairs_inpocket'+i))
	butt=get_basic_feature_butterworth(read_csv('upstairs_inpocket' , 'upstairs_inpocket'+i))
	plt.figure(figsize = (15, 10))
	plt.subplot(2, 1, 1)
	plt.plot(x, orig, 'b-' , alpha = 0.5)
	plt.title('upstairs_inpocket_original'+i)
	plt.subplot(2, 1, 2)
	plt.plot(x, butt, 'b-' , alpha = 0.5)
	plt.title('upstairs_inpocket_butterworth'+i)

	#compare walk_hold
	orig=get_basic_feature(read_csv('walk_hold' , 'walk_hold'+i))
	butt=get_basic_feature_butterworth(read_csv('walk_hold' , 'walk_hold'+i))
	plt.figure(figsize = (15, 10))
	plt.subplot(2, 1, 1)
	plt.plot(x, orig, 'b-' , alpha = 0.5)
	plt.title('walk_hold_original'+i)
	plt.subplot(2, 1, 2)
	plt.plot(x, butt, 'b-' , alpha = 0.5)
	plt.title('walk_hold_butterworth'+i)

	#compare walk_inpocket
	orig=get_basic_feature(read_csv('walk_inpocket' , 'walk_inpocket'+i))
	butt=get_basic_feature_butterworth(read_csv('walk_inpocket' , 'walk_inpocket'+i))
	plt.figure(figsize = (15, 10))
	plt.subplot(2, 1, 1)
	plt.plot(x, orig, 'b-' , alpha = 0.5)
	plt.title('walk_inpocket_original'+i)
	plt.subplot(2, 1, 2)
	plt.plot(x, butt, 'b-' , alpha = 0.5)
	plt.title('walk_inpocket_butterworth'+i)

	#compare falldown_hold
	orig=get_basic_feature(read_csv('falldown_hold' , 'falldown_hold'+i))
	butt=get_basic_feature_butterworth(read_csv('falldown_hold' , 'falldown_hold'+i))
	plt.figure(figsize = (15, 10))
	plt.subplot(2, 1, 1)
	plt.plot(x, orig, 'b-' , alpha = 0.5)
	plt.title('falldown_hold_original'+i)
	plt.subplot(2, 1, 2)
	plt.plot(x, butt, 'b-' , alpha = 0.5)
	plt.title('falldown_hold_butterworth'+i)
	plt.savefig('Butterworth_fh'+i+'.png')

	#compare falldown_hold
	orig=get_basic_feature(read_csv('falldown_inpocket' , 'falldown_inpocket'+i))
	butt=get_basic_feature_butterworth(read_csv('falldown_inpocket' , 'falldown_inpocket'+i))
	plt.figure(figsize = (15, 10))
	plt.subplot(2, 1, 1)
	plt.plot(x, orig, 'b-' , alpha = 0.5)
	plt.title('falldown_inpocket_original'+i)
	plt.subplot(2, 1, 2)
	plt.plot(x, butt, 'b-' , alpha = 0.5)
	plt.title('falldown_inpocket_butterworth'+i)

#compair different catogary set: subplot
def plot_feature():
	i=7
	n=8*i
	# plt.plot(x[:-8], y[0][:-8], 'b-' , alpha = 0.5)
	plt.figure(figsize = (15, 15))
	plt.subplot(8 , 1 , 1)
	plt.plot(x, y[0+n], 'b-' , alpha = 0.5)
	plt.title('downstairs_hold')
	plt.subplot(8 , 1 , 2)
	plt.plot(x, y[1+n], 'b-' , alpha = 0.5)
	plt.title('downstairs_inpocket')
	plt.subplot(8 , 1 , 3)
	plt.plot(x, y[2+n], 'b-' , alpha = 0.5)
	plt.title('upstairs_hold')
	plt.subplot(8 , 1 , 4)
	plt.plot(x, y[3+n], 'b-' , alpha = 0.5)
	plt.title('upstairs_inpocket')
	plt.subplot(8 , 1 , 5)
	plt.plot(x, y[4+n], 'b-' , alpha = 0.5)
	plt.title('walk_hold')
	plt.subplot(8 , 1 , 6)
	plt.plot(x, y[5+n], 'b-' , alpha = 0.5)
	plt.title('walk_inpocket')
	plt.subplot(8 , 1 , 7)
	plt.plot(x, y[6+n], 'b-' , alpha = 0.5)
	plt.title('falldown_hold')
	plt.subplot(8 , 1 , 8)
	plt.plot(x, y[7+n], 'b-' , alpha = 0.5)
	plt.title('falldown_inpocket')
	plt.savefig('different_catogary_set'+str(i+1)+'.png')

#compair different catogary set: plot together
def plot_compare_feature():
	i=7
	n=8*i
	plt.figure(figsize = (15, 15))
	plt.plot(x, y[0+n], 'r-' , alpha = 0.5, label='downstairs_hold'+str(i+1))	 #red downstairs_hold
	plt.plot(x, y[1+n], 'g-' , alpha = 0.5, label='downstairs_inpocket'+str(i+1)) #green downstairs_inpocket
	plt.plot(x, y[2+n], 'b-' , alpha = 0.5, label='upstairs_hold'+str(i+1))	  #blue upstairs_hold
	plt.plot(x, y[3+n], 'y-' , alpha = 0.5, label='upstairs_inpocket'+str(i+1))   #yellow upstairs_inpocket
	plt.plot(x, y[4+n], 'k-' , alpha = 0.5, label='walk_hold'+str(i+1))	       #black walk_hold
	plt.plot(x, y[5+n], 'c-' , alpha = 0.5, label='walk_inpocket'+str(i+1))	#cyan walk_inpocket
	# plt.plot(x, y[6+n], 'm-' , alpha = 0.5, label='falldown_hold'+str(i+1))       #magenta falldown_hold
	# plt.plot(x, y[7+n], 'm--' , alpha = 0.5, label='falldown_inpocket'+str(i+1))       #magenta -- falldown_inpocket
	plt.legend()
	plt.savefig('compare_different_catogary_set'+str(i+1)+'.png')


def same_catogary():
	# downstairs_hold
	n1=1
	n2=2
	n3=3
	s1=7*(n1-1)
	s2=7*(n2-1)
	s3=7*(n3-1)
	plt.figure(figsize = (30, 30))
	plt.subplot(3 , 1 , 1)
	plt.plot(x, y[s1], 'b-' , alpha = 0.5)
	plt.title('downstairs_hold'+str(n1))
	plt.subplot(3 , 1 , 2)
	plt.plot(x, y[s2], 'b-' , alpha = 0.5)
	plt.title('downstairs_hold'+str(n2))
	plt.subplot(3 , 1 , 3)
	plt.plot(x, y[s3], 'b-' , alpha = 0.5)
	plt.title('downstairs_hold'+str(n3))

	plt.figure(figsize = (30, 30))
	plt.plot(x, y[s1], 'r-' , alpha = 0.5)
	plt.title('downstairs_hold'+str(n1))
	plt.plot(x, y[s2], 'g-' , alpha = 0.5)
	plt.title('downstairs_hold'+str(n2))
	plt.plot(x, y[s3], 'b-' , alpha = 0.5)
	plt.title('downstairs_hold'+str(n3))

	# downstairs_inpocket
	n1=1
	n2=2
	n3=3
	s1=7*(n1-1)+1
	s2=7*(n2-1)+1
	s3=7*(n3-1)+1
	plt.figure(figsize = (30, 30))
	plt.subplot(3 , 1 , 1)
	plt.plot(x, y[s1], 'b-' , alpha = 0.5)
	plt.title('downstairs_inpocket'+str(n1))
	plt.subplot(3 , 1 , 2)
	plt.plot(x, y[s2], 'b-' , alpha = 0.5)
	plt.title('downstairs_inpocket'+str(n2))
	plt.subplot(3 , 1 , 3)
	plt.plot(x, y[s3], 'b-' , alpha = 0.5)
	plt.title('downstairs_inpocket'+str(n3))

	# upstairs_hold
	n1=1
	n2=2
	n3=3
	s1=7*(n1-1)+2
	s2=7*(n2-1)+2
	s3=7*(n3-1)+2
	plt.figure(figsize = (30, 30))
	plt.subplot(3 , 1 , 1)
	plt.plot(x, y[s1], 'b-' , alpha = 0.5)
	plt.title('upstairs_hold'+str(n1))
	plt.subplot(3 , 1 , 2)
	plt.plot(x, y[s2], 'b-' , alpha = 0.5)
	plt.title('upstairs_hold'+str(n2))
	plt.subplot(3 , 1 , 3)
	plt.plot(x, y[s3], 'b-' , alpha = 0.5)
	plt.title('upstairs_hold'+str(n3))

	# upstairs_inpocket
	n1=1
	n2=2
	n3=3
	s1=7*(n1-1)+3
	s2=7*(n2-1)+3
	s3=7*(n3-1)+3
	plt.figure(figsize = (30, 30))
	plt.subplot(3 , 1 , 1)
	plt.plot(x, y[s1], 'b-' , alpha = 0.5)
	plt.title('upstairs_inpocket'+str(n1))
	plt.subplot(3 , 1 , 2)
	plt.plot(x, y[s2], 'b-' , alpha = 0.5)
	plt.title('upstairs_inpocket'+str(n2))
	plt.subplot(3 , 1 , 3)
	plt.plot(x, y[s3], 'b-' , alpha = 0.5)
	plt.title('upstairs_inpocket'+str(n3))

	# walk_hold
	n1=1
	n2=2
	n3=3
	s1=7*(n1-1)+4
	s2=7*(n2-1)+4
	s3=7*(n3-1)+4
	plt.figure(figsize = (30, 30))
	plt.subplot(3 , 1 , 1)
	plt.plot(x, y[s1], 'b-' , alpha = 0.5)
	plt.title('walk_hold'+str(n1))
	plt.subplot(3 , 1 , 2)
	plt.plot(x, y[s2], 'b-' , alpha = 0.5)
	plt.title('walk_hold'+str(n2))
	plt.subplot(3 , 1 , 3)
	plt.plot(x, y[s3], 'b-' , alpha = 0.5)
	plt.title('walk_hold'+str(n3))

	# walk_inpocket
	n1=1
	n2=2
	n3=3
	s1=7*(n1-1)+5
	s2=7*(n2-1)+5
	s3=7*(n3-1)+5
	plt.figure(figsize = (30, 30))
	plt.subplot(3 , 1 , 1)
	plt.plot(x, y[s1], 'b-' , alpha = 0.5)
	plt.title('walk_inpocket'+str(n1))
	plt.subplot(3 , 1 , 2)
	plt.plot(x, y[s2], 'b-' , alpha = 0.5)
	plt.title('walk_inpocket'+str(n2))
	plt.subplot(3 , 1 , 3)
	plt.plot(x, y[s3], 'b-' , alpha = 0.5)
	plt.title('walk_inpocket'+str(n3))

	# falldown_hold
	n1=1
	n2=2
	n3=3
	s1=7*(n1-1)+6
	s2=7*(n2-1)+6
	s3=7*(n3-1)+6
	plt.figure(figsize = (30, 30))
	plt.subplot(3 , 1 , 1)
	plt.plot(x, y[s1], 'b-' , alpha = 0.5)
	plt.title('falldown_hold'+str(n1))
	plt.subplot(3 , 1 , 2)
	plt.plot(x, y[s2], 'b-' , alpha = 0.5)
	plt.title('falldown_hold'+str(n2))
	plt.subplot(3 , 1 , 3)
	plt.plot(x, y[s3], 'b-' , alpha = 0.5)
	plt.title('falldown_hold'+str(n3))
	plt.show()


# Original_vs_butterworth()
# original_data('falldown_hold', 'falldown_hold6')
plot_feature()
# plot_compare_feature()
# same_catogary()

