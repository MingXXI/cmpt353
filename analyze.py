from Cleaning_Data import *
import matplotlib.pyplot as plt

'''
	difference between use Butterworth or not plot
'''
def Original_vs_butterworth():
	x=range(49)
	i='7'		#change data set.

	#compare downstair_hold
	orig=get_basic_feature(read_csv('downstairs_hold' , 'downstairs_hold'+i))
	butt=get_basic_feature_butterworth(read_csv('downstairs_hold' , 'downstairs_hold'+i))
	plt.figure(figsize = (30, 30))
	plt.subplot(2, 1, 1)
	plt.plot(x, orig, 'b-' , alpha = 0.5)
	plt.title('downstairs_hold_original'+i)
	plt.subplot(2 , 1 , 2)
	plt.plot(x, butt, 'b-' , alpha = 0.5)
	plt.title('downstairs_hold_butterworth'+i)

	#compare downstairs_inpocket
	orig=get_basic_feature(read_csv('downstairs_inpocket' , 'downstairs_inpocket'+i))
	butt=get_basic_feature_butterworth(read_csv('downstairs_inpocket' , 'downstairs_inpocket'+i))
	plt.figure(figsize = (30, 30))
	plt.subplot(2, 1, 1)
	plt.plot(x, orig, 'b-' , alpha = 0.5)
	plt.title('downstairs_inpocket_original'+i)
	plt.subplot(2, 1, 2)
	plt.plot(x, butt, 'b-' , alpha = 0.5)
	plt.title('downstairs_inpocket_butterworth'+i)

	#compare upstairs_hold
	orig=get_basic_feature(read_csv('upstairs_hold' , 'upstairs_hold'+i))
	butt=get_basic_feature_butterworth(read_csv('upstairs_hold' , 'upstairs_hold'+i))
	plt.figure(figsize = (30, 30))
	plt.subplot(2, 1, 1)
	plt.plot(x, orig, 'b-' , alpha = 0.5)
	plt.title('upstairs_hold_original'+i)
	plt.subplot(2, 1, 2)
	plt.plot(x, butt, 'b-' , alpha = 0.5)
	plt.title('upstairs_hold_butterworth'+i)

	#compare upstairs_inpocket
	orig=get_basic_feature(read_csv('upstairs_inpocket' , 'upstairs_inpocket'+i))
	butt=get_basic_feature_butterworth(read_csv('upstairs_inpocket' , 'upstairs_inpocket'+i))
	plt.figure(figsize = (30, 30))
	plt.subplot(2, 1, 1)
	plt.plot(x, orig, 'b-' , alpha = 0.5)
	plt.title('upstairs_inpocket_original'+i)
	plt.subplot(2, 1, 2)
	plt.plot(x, butt, 'b-' , alpha = 0.5)
	plt.title('upstairs_inpocket_butterworth'+i)

	#compare walk_hold
	orig=get_basic_feature(read_csv('walk_hold' , 'walk_hold'+i))
	butt=get_basic_feature_butterworth(read_csv('walk_hold' , 'walk_hold'+i))
	plt.figure(figsize = (30, 30))
	plt.subplot(2, 1, 1)
	plt.plot(x, orig, 'b-' , alpha = 0.5)
	plt.title('walk_hold_original'+i)
	plt.subplot(2, 1, 2)
	plt.plot(x, butt, 'b-' , alpha = 0.5)
	plt.title('walk_hold_butterworth'+i)

	#compare walk_inpocket
	orig=get_basic_feature(read_csv('walk_inpocket' , 'walk_inpocket'+i))
	butt=get_basic_feature_butterworth(read_csv('walk_inpocket' , 'walk_inpocket'+i))
	plt.figure(figsize = (30, 30))
	plt.subplot(2, 1, 1)
	plt.plot(x, orig, 'b-' , alpha = 0.5)
	plt.title('walk_inpocket_original'+i)
	plt.subplot(2, 1, 2)
	plt.plot(x, butt, 'b-' , alpha = 0.5)
	plt.title('walk_inpocket_butterworth'+i)

	#compare falldown_hold
	orig=get_basic_feature(read_csv('falldown_hold' , 'falldown_hold'+i))
	butt=get_basic_feature_butterworth(read_csv('falldown_hold' , 'falldown_hold'+i))
	plt.figure(figsize = (30, 30))
	plt.subplot(2, 1, 1)
	plt.plot(x, orig, 'b-' , alpha = 0.5)
	plt.title('falldown_hold_original'+i)
	plt.subplot(2, 1, 2)
	plt.plot(x, butt, 'b-' , alpha = 0.5)
	plt.title('falldown_hold_butterworth'+i)
	plt.show()

# Original_vs_butterworth()

#compair different catogary set: subplot
def plot_feature():
    x = range(49)
    y = get_X()
    i=0
    n=7*i
    # plt.plot(x[:-8], y[0][:-8], 'b-' , alpha = 0.5)
    plt.figure(figsize = (30, 30))
    plt.subplot(7 , 1 , 1)
    plt.plot(x, y[0+n], 'b-' , alpha = 0.5)
    plt.title('downstairs_hold')
    plt.subplot(7 , 1 , 2)
    plt.plot(x, y[1+n], 'b-' , alpha = 0.5)
    plt.title('downstairs_inpocket')
    plt.subplot(7 , 1 , 3)
    plt.plot(x, y[2+n], 'b-' , alpha = 0.5)
    plt.title('upstairs_hold')
    plt.subplot(7 , 1 , 4)
    plt.plot(x, y[3+n], 'b-' , alpha = 0.5)
    plt.title('upstairs_inpocket')
    plt.subplot(7 , 1 , 5)
    plt.plot(x, y[4+n], 'b-' , alpha = 0.5)
    plt.title('walk_hold')
    plt.subplot(7 , 1 , 6)
    plt.plot(x, y[5+n], 'b-' , alpha = 0.5)
    plt.title('walk_inpocket')
    plt.subplot(7 , 1 , 7)
    plt.plot(x, y[6+n], 'b-' , alpha = 0.5)
    plt.title('falldown_hold')

#compair different catogary set: plot together
def plot_compare_feature():
	x = range(49)
	y = get_X()
	i=0
	n=7*i
	plt.figure(figsize = (30, 30))
	plt.plot(x, y[0+n], 'r-' , alpha = 0.5, label='downstairs_hold')	 #red downstairs_hold
	plt.plot(x, y[1+n], 'g-' , alpha = 0.5, label='downstairs_inpocket') #green downstairs_inpocket
	plt.plot(x, y[2+n], 'b-' , alpha = 0.5, label='upstairs_hold')	  #blue upstairs_hold
	plt.plot(x, y[3+n], 'y-' , alpha = 0.5, label='upstairs_inpocket')   #yellow upstairs_inpocket
	plt.plot(x, y[4+n], 'k-' , alpha = 0.5, label='walk_hold')	       #black walk_hold
	plt.plot(x, y[5+n], 'c-' , alpha = 0.5, label='walk_inpocket')	#cyan walk_inpocket
	#plt.plot(x, y[6+n], 'm-' , alpha = 0.5, label='falldown_hold')       #magenta falldown_hold
	plt.legend()
	plt.show()
plot_compare_feature()

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


