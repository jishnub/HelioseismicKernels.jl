import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os,sys
from astropy.io import fits
import warnings

class Point():
    def __init__(self,Line2D):
        self.x  = Line2D.get_xdata()[0]
        self.y  = Line2D.get_ydata()[0]
        self.artist = Line2D

    def remove(self):
        self.artist.remove()
        self.artist=None

'''
    Shift + click to create points, 
    press c to finalize the curve, 
    press d to delete the curve
'''
class Poly():
    def __init__(self):
        self.points=[]
        self.fitted_curve=None

    def fit(self,order=4):

        #~ If the number of points are not sufficient, remove fitted polynomial if it exists
        if len(self.points)<order+1:
            if self.fitted_curve is not None: self.fitted_curve.remove()
            self.fitted_curve=None
            return (None,None)

        #~ If there are sifficient points, fit the points with a polynomial function
        xcoords = [pt.x for pt in self.points]
        ycoords = [pt.y for pt in self.points]
        pfit = np.polyfit(xcoords,ycoords,order)

        #~ Generate points on a fine grid along the fitted polynomial
        #~ This is to plot a continuous line
        fit_t = np.polyval(pfit,phi)

        #~ Update fitted curve
        if self.fitted_curve is not None: self.fitted_curve.remove()
        self.fitted_curve,=plt.plot(phi,fit_t,'g')
        plt.ylim(0,t_max)

        #~ String to format fitted polynomial like p_2*k**2 + p_1*k +  p_0
        fmtstr=" + ".join(["{}*phi^"+str(i) if i>1 else "{}*phi" if i==1 else "{}"  for i in range(len(pfit))[::-1]])
        polystr=fmtstr.format(*pfit).replace("+ -","- ")

        fitstr=[]
        for index,p_i in enumerate(pfit[::-1]):
            fitstr.append("index("+str(index)+") = "+str(p_i))
        fitstr="\n".join(fitstr)

        return polystr,fitstr

    def remove_match(self,artist):
        ''' Find and remove the point that was clicked on '''
        for point in self.points:
            if point.artist == artist:
                point.remove()
                self.points.remove(point)
                break

    def clear(self):
        ''' Refresh the working slate by deleting plots and lines.
        Points and lines already finalized are untouched. '''
        for point in self.points:
            point.remove()
        if self.fitted_curve is not None:
            self.fitted_curve.remove()
            self.fitted_curve=None
        self.points=[]

class Track_interactions():
    def __init__(self,figure):
        self.button_press_event=figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.key_press_event=figure.canvas.mpl_connect('key_press_event', self.onpress)
        self.key_release_event=figure.canvas.mpl_connect('key_release_event', self.onrelease)
        self.pick_event=figure.canvas.mpl_connect('pick_event', self.onpick)
        self.shift_pressed=False
        self.poly=Poly()

    def onpick(self,event):
        ''' Remove a point when it is clicked on '''
        self.poly.remove_match(event.artist)
        self.poly.fit()
        plt.draw()

    def onclick(self,event):
        ''' Add a point at the (x,y) coordinates of click '''
        if event.button == 1 and self.shift_pressed:
            pt_artist,=plt.plot([event.xdata],[event.ydata],marker='o',color='b',linestyle='none',picker=5)
            self.poly.points.append(Point(pt_artist))
            self.poly.fit()
            plt.draw()

    def onpress(self,event):

        if event.key == 'c':
            polystr,fitstr = self.poly.fit()
            if polystr is None: return
            print(polystr,"\n",fitstr,"\n")
            self.poly.fitted_curve.set_color("black")
            plt.draw()
            self.poly=Poly()

        elif event.key=="d":
            self.poly.clear()

        elif event.key=="shift":
            self.shift_pressed = True

        plt.draw()

    def onrelease(self,event):
        if event.key=='shift':
            self.shift_pressed = False

filename="/home/jishnu/kernels_scratch/kernels/C_phi_t.fits"
data=np.squeeze(fits.open(filename)[0].data)

nt,nphi=data.shape

dnu = 2e-6; T=1/dnu; dt=T/nt;

phi=np.linspace(0,2*np.pi,nphi)
t=np.arange(nt)*dt

t_max = 2*3600
t = t[t<=t_max]

phi_max = np.pi
phi = phi[phi<=phi_max]

data = data[np.ix_(t<=t_max,phi<=phi_max)]


#########################################################################################

figure=plt.figure()

plt.pcolormesh(phi,t,data/abs(data).max(),cmap='Greys',vmax=0.1, vmin=-0.1)

_=Track_interactions(figure)

plt.show()
