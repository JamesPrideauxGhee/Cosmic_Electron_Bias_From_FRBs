import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord as SC
import pyne2001 as ne

# Load data set

file = 'localised_frbs_citations.txt'
loc_data = np.loadtxt(file,skiprows=1,usecols=(1,2,3,4),max_rows=20)
nfrb = np.shape(loc_data)[0]
for i in range(nfrb):
    c = SC(ra= loc_data[i,0]*u.degree, dec = loc_data[i,1]*u.degree)
    c = c.galactic
    glon = c.l.value
    glat = c.b.value
    #print(f'glon = {glon}')
    #print(f'ra/dec = {loc_data[i,0]}/{loc_data[i,1]}')
    dm_mw = ne.get_galactic_dm(glon,glat)
    #print(f'dm_mw = {dm_mw}')
    loc_data[i,2] -= dm_mw
    print(f'z/dm = {loc_data[i,3]}/{loc_data[i,2]}')
#np.savez('localised_frbs_ne2001',data=loc_data)
#inds = loc_data[:,3].argsort()
#sort_z = loc_data[:,3][inds[::-1]]
#sort_dm = loc_data[:,2][inds[::-1]]
#
#print('reordered arrays:')
#for i in range(len(sort_z)):
#    print(f'z/dm = {sort_z[i]}/{sort_dm[i]}')
ra = loc_data[:,0]
dec = loc_data[:,1]
dm = loc_data[:,2]
z = loc_data[:,3]

data = np.zeros((nfrb,4))
data[:,0] = ra
data[:,1] = dec
data[:,2] = dm
data[:,3] = z
np.savez('localised_frbs_20_ne2001',data=data)


#file = 'sdss_frbs.npz'
#hf = np.load(file)
#ra = hf['ra']
#dec = hf['dec']
#dm = hf['dm']
#nfrb = len(ra)
#for i in range(nfrb):
#    c = SC(ra=ra[i]*u.degree, dec = dec[i]*u.degree)
#    c = c.galactic
#    glon = c.l.value
#    glat = c.b.value
   #print(f'glon = {glon}')
   #print(f'ra/dec = {loc_data[i,0]}/{loc_data[i,1]}')
#    dm_mw = ne.get_galactic_dm(glon,glat)
   #print(f'dm_mw = {dm_mw}')
#    dm[i] -= dm_mw

#data = np.zeros((nfrb,4))
#data[:,0] = ra
#data[:,1] = dec
#data[:,2] = dm
#data[:,3] = z
#np.savez('localised_frbs_20_ne2001',data=data)










