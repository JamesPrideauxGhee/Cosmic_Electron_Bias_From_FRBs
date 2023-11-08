## Script to calculate the Milky Way contribute to an FRB's dispersion measure
## as given by the NE2001 model.
## NOTE: this script saves a copy of the data where the saved dispersion measure
## is the observed minus the Milky Way part.


import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord as SC
import pyne2001 as ne

# Load data set

file = 'localised_frbs_citations.txt'
loc_data = np.loadtxt(file,skiprows=1,usecols=(1,2,3,4),max_rows=20)
nfrb = np.shape(loc_data)[0]
for i in range(nfrb):
    # Convert angular coordinates
    c = SC(ra= loc_data[i,0]*u.degree, dec = loc_data[i,1]*u.degree)
    c = c.galactic
    glon = c.l.value
    glat = c.b.value

    # Use NE2001 model to get the galactic contribution to the DM
    dm_mw = ne.get_galactic_dm(glon,glat)

    # Subtract from the observed DM
    loc_data[i,2] -= dm_mw

ra = loc_data[:,0]
dec = loc_data[:,1]
dm = loc_data[:,2]
z = loc_data[:,3]

data = np.zeros((nfrb,4))
data[:,0] = ra
data[:,1] = dec
data[:,2] = dm
data[:,3] = z
# Save, NOTE edit file name 
np.savez('localised_frbs_20_ne2001',data=data)













