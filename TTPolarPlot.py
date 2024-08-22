'''
Hz_per_tick =     How many Hz per radial tick in the polar projection. 

sec_per_rev =     How many second per one revolution of the platter:
                      33PRM = 1.8
                      45RPM = 1.33
                      78RPM = .766

revs =            The number of revolutions to plot.
                
revoffset =       Offset number of revolutions before plotting.               

pltpolar =        0 - Plot x,y projection. 
                  1 - Plot polar projection.

pltlegend =       0 - No legend. 
                  1 - Legend with mean frequency per rev.   

stereo_channel =  0 - Left Channel.
                  1 - Right Channel.


filter_freq =     Lowpass filter frequency in Hz.  Default is 60. 
                



**** Version 6 Beta ****


'''


from scipy import signal
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os



#edit here to adjust input parameters:


_FILE = 'DK2412D015 Motor FG PostMech-2(oil).wav'

info_line = 'MK3 FG'

Hz_per_tick = 1
sec_per_rev = 1.8
revs = 1
revoffset = 2

pltpolar = 1
pltlegend = 0

stereo_channel = 0
filter_freq = 5

#end edit




def instfreq(sig,Fs,filter_freq):
    z = signal.hilbert(sig)
    rawfreq = Fs/(2*np.pi)*np.diff(np.unwrap(np.angle(z)))
    rawfreq = np.append(rawfreq,rawfreq[len(rawfreq)-1])    #np.diff drops one end point

    b, a = signal.iirfilter(1,filter_freq/(Fs/2), btype='lowpass')
    zi = signal.lfilter_zi(b, a) #Initialize the filter to the mean of the leading edge of the data
    rawfreq,_ = signal.lfilter(b,a,rawfreq,zi=zi*np.mean(rawfreq[0:2000])) #reduces glitch, first pass

    b, a = signal.iirfilter(3,filter_freq/(Fs/2), btype='lowpass')
    instfreq = signal.filtfilt(b,a,rawfreq) #back and forth linear phase IIR filter (6 pole)

    return (instfreq)


y = read(_FILE)
Fs = float(y[0])
if np.size(y[1][0]) == 2:
    sig = y[1][:,stereo_channel][0:int(Fs*(sec_per_rev*(.5+revs+revoffset)))] #Grab revs + revoffest + .5 of audio from the specified channel
else:
    sig = y[1][0:int(Fs*(sec_per_rev*3))] #mono file



freq1 = instfreq(sig,Fs,filter_freq)

freq1 = np.roll(freq1,-int(Fs*.2))#Throw away the first .2sec to guarantee the IIR transient settles


maxf = (max(freq1[int((Fs*sec_per_rev)*(revoffset)):int((Fs*sec_per_rev)*(revoffset+revs))])+1)

plotdata = {}
for x in range(revs):
    plotdata[x] = freq1[int((Fs*sec_per_rev)*(x+revoffset)):int((Fs*sec_per_rev)*(x+1+revoffset))]
    if pltpolar == 1:
        plotdata[x] = 20.-(maxf-plotdata[x])/Hz_per_tick



if pltpolar == 1:

    plt.figure(figsize=(11,11))
    ax = plt.subplot(111, projection='polar')

    t = np.arange(sec_per_rev,0,-1/Fs)  #Reverse time (theta axis)
    theta = t*2*np.pi/sec_per_rev   #Time becomes degrees (1 rev = 2pi radians)
    theta = np.roll(theta,int((sec_per_rev*Fs/4)))  #Rotate 90 deg to put 0 on top (sec_per_rev*Fs/4)

    for key in plotdata:
        ax.plot(theta, plotdata[key], label = 'Rev ' + str(key+1) + ': {:4.3f}Hz'.format(np.mean(plotdata[key])))
                
    dgr = (2*np.pi)/360.

    mod_date = datetime.datetime.fromtimestamp(os.path.getmtime(_FILE))
    ax.text(226.*dgr, 28.5, _FILE + "\n" + \
        mod_date.strftime("%b %d, %Y %H:%M:%S"), fontsize=9)

    ax.set_rmax(20)
                    #Set up the ticks y is radial x is theta, it turns out x and y
                    #methods still work in polar projection but sometimes do funny things

    tick_loc = np.arange(1,21,1)

    myticks = []
    for x in range(0,20,1):
        myticks.append('{:4.2f}Hz'.format(maxf-(19*Hz_per_tick)+x*Hz_per_tick))

    ax.set_rgrids(tick_loc, labels = myticks, angle = 90, fontsize = 8)

    ax.set_xticklabels(['90'+u'\N{DEGREE SIGN}','45'+u'\N{DEGREE SIGN}','0'+u'\N{DEGREE SIGN}',\
                        '315'+u'\N{DEGREE SIGN}','270'+u'\N{DEGREE SIGN}','225'+u'\N{DEGREE SIGN}',\
                        '180'+u'\N{DEGREE SIGN}','135'+u'\N{DEGREE SIGN}'])


else:

    plt.figure(figsize=(20,8))
    ax = plt.subplot(111)

    t = np.arange(0,sec_per_rev,1/Fs)  #normal time (x axis)

    for key in plotdata:
        ax.plot(t, plotdata[key], label = 'Rev ' + str(key+1) + ': {:4.3f}Hz'.format(np.mean(plotdata[key])))

    if pltlegend == 1:
        ax.legend(loc='lower center', ncol=10)


    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")


if pltlegend == 1:
    ax.legend(loc='lower center', ncol=10)

ax.grid(True)

ax.set_title(info_line, va='bottom', fontsize=16)


plt.savefig(info_line.replace(' / ', '_') +'.png', bbox_inches='tight', pad_inches=.5)


plt.show()

