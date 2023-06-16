import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import astropy
from numpy.polynomial import polynomial as P
from astropy import wcs

# check for a command line argument
nargs = len(sys.argv) - 1
if (nargs==1):
    filename = sys.argv[1]
else:
    print("Needs a fits file")
    print("e.g. cube-viewer.py foot.fits")
    sys.exit()

fits_file = fits.open(filename)    
header = fits_file[0].header
print(header)
data = fits_file[0].data

w = wcs.WCS(header)
print(w)

print(data.shape)
nt = data.shape[0]
nx = data.shape[1]
ny = data.shape[2]

#print("clipping first few frames")
#data = data[20:nt,:,:]
print("data file shape : "+str(data.shape))
nt = data.shape[0]

std_image = np.std(data,axis=0)
median_image = np.median(data,axis=0)

nsx = 2
nsy = 2

plt.subplot(nsx,nsy,3)#,projection=w, slices=('x', 'y', (nx//2,ny//2)))
plt.imshow(median_image,aspect='auto',origin='lower')
plt.title("Median of images")

plt.subplot(nsx,nsy,4)
plt.imshow(np.log10(std_image),aspect='auto',origin='lower')#,projection=w, slices=('x', 'y', (nx//2,ny//2)))
plt.title("log10(sigma of images)")

plt.subplot(4,1,1)
sigma = np.std(data[:,:,:],axis=(1,2))

plt.plot(sigma)
plt.xlabel("image sample number")
plt.ylabel("sigma for each image")
plt.title(filename)

plt.subplot(4,1,2)
ts = data[:,352,171]

plt.plot(ts)
plt.xlabel("image sample number")
plt.ylabel("flux in pixel for 0407")
plt.title(filename)



plt.show()

