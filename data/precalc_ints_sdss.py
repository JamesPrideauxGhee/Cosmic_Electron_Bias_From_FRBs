## Script to calculate the line-of-sight integrals of an FRB
## There are a lot of options in this file, which should be 
## commented at each stage.

import numpy as np
from lss_new import los_int_ccl, tabulate_z, ccl_correction
import h5py as h5
import matplotlib.pyplot as plt


# Choose whether we make mock data, or load real data
make_mocks = True
# Choose whether we use a real field or not in the los integral
mock_field = False
# Choose whether we load existing mock data (to do precalculation for many mocks in steps)
load_mocks = False
# Choose whether we precalculate over a subset of the data (assumed true if we load mocks)
use_subset = False

if make_mocks == True:
    print('Making mock data...')
else:
    print('Using real data...')
if mock_field == True:
    print('Using fake field (warning, need the same field in analysis!)...')
else:
    print('Using real field(s)...')

# Start with making/loading FRB data and fields.
if make_mocks == True:
    if load_mocks == False:     
        # Choose number of FRBs to make
        nfrbs = 1000
        print(f'Making {nfrbs} mocks')
     
        if mock_field == True:
        # Describe a fake mock field
            N = 4                                             
            #L = 2000/(H0/100)       
            L = 100
            xmin = np.ones((3)) * -L/2
            nfields = 1                                                        
            fields = np.ones((nfields,N,N,N))    
                                                                     
            # Make fields:
            # Some examples given here...
            #fields[0,:,:,:] = 1
            #fields[0] = np.random.uniform(-0.5,0.5,size=(N,N,N))
            #fields[2] += np.random.uniform(-0.05,0.05,size=(N,N,N))
            #fields[3] += np.random.uniform(-0.15,0.15,size=(N,N,N))
            #fields[1,:,:,:] += np.random.uniform(-0.1,0,1)
        else:
            # Use a real field
            # Default here uses the SDSS BOSS BORG samples labelled by values in next line
            labs = [6000,6200,6400,6600,6800,7000,7200,7400,7600,7800,8000,8200,8400,8600,8800,9000]
            nfields = len(labs)
            N = 256 
            # Cycle through fields and save density fields to an array
            fields = np.zeros((nfields,N,N,N))
            for i in range(nfields):
                file = f'sdss_boss_final_density_{labs[i]}.h5'
                hf = h5.File(file,'r') 
                field = np.array(hf['scalars/BORG_final_density'])
                fields[i,:,:,:] = field
            # Describe SDSS BOSS field 
            L = 4000
            xmin = np.ones((3))
            xmin[0] = -2200
            xmin[1] = -2000
            xmin[2] = -300
    
        # Make mock FRB z, ra, dec
    
        ra = np.random.uniform(150,200,size=(nfrbs))#45 * np.ones((nfrbs))
        dec = np.random.uniform(20,50,size=(nfrbs))#45 * np.ones((nfrbs))
        z = np.linspace(0,1.0,nfrbs)
    
        # Save mock data
        np.savez('mock_data',z=z,ra=ra,dec=dec)
    else:
        # Load a subset of the mocks
        # Choose start and end indicies
        start = 0
        end = 1
        data = np.load('mock_data.npz')
        ra = data['ra'][start:end]
        dec = data['dec'][start:end]
        z = data['z'][start:end]
        nfrbs = len(z)
        use_subset = True
        print(f'Loading mocks...')
        print(f'Using frbs {start} - {end}')
        labs = [6000,6200,6400,6600,6800,7000,7200,7400,7600,7800,8000,8200,8400,8600,8800,9000]
        nfields = len(labs)
        N = 256 #np.shape(field)[0]
        fields = np.zeros((nfields,N,N,N))
        for i in range(nfields):
            file = f'final_density_{labs[i]}.h5'
            hf = h5.File(file,'r') 
            field = np.array(hf['scalars/field'])
            fields[i,:,:,:] = field
                                                                                                   
        L = 4000
        xmin = np.ones((3))
        xmin[0] = -2200
        xmin[1] = -2000
        xmin[2] = -300

else:
    # Use real FRB data
    
    # Load data (ra and dec, and known zs)
    
    file = 'sdss_frbs_ne2001.npz' # 'localised_frbs_20_ne2001.npz'
    hf = np.load(file)
    list(hf.keys())
    data = hf['data'] 
    ra = data[:,0]
    dec = data[:,1]
    dm_obs = data[:,2]


    nfrbs = len(ra)

    print('Data used:')
    print(file)

    # Load fields
    # NOTE: Default is SDSS BOSS fields with the numerical labels below
        
    labs = [6000,6200,6400,6600,6800,7000,7200,7400,7600,7800,8000,8200,8400,8600,8800,9000]
    nfields = len(labs)
    N = 256 
    fields = np.zeros((nfields,N,N,N))
    for i in range(nfields):
        file = f'sdss_boss_final_density_{labs[i]}.h5'
        hf = h5.File(file,'r')
        field = np.array(hf['scalars/BORG_final_density'])
        fields[i,:,:,:] = field
                                  
    L = 4000
    xmin = np.ones((3))
    xmin[0] = -2200
    xmin[1] = -2000
    xmin[2] = -300



## With data & fields uploaded, now begin to calculate the los integrals

# Choose upper z limit for increments which we use to discretise the integrals
z_max = 1.0

# Choose number of z increments to calculate at
M = 50
z_ints = np.linspace(0,z_max,M)
print(f'z_ints = {z_ints}')


if use_subset == False:
    # Use the whole of the presaved dataset.
    # This is the option we use with real data.

    # Define arrays to save things
    tabulated_z = np.zeros((M))
    tabulated_los_int = np.zeros((nfields,nfrbs,M))
    tabulated_los_int_grad = np.zeros((nfields,nfrbs,M))
    tabulated_exceeds = np.zeros((nfrbs,M))
    tabulated_ccl = np.zeros((nfrbs,M))
    tabulated_ccl_grad = np.zeros((nfrbs,M))
    
    tabulated_z[:] = z_ints
    
    print(f'nfrbs = {nfrbs}')
    print(f'nfields = {nfields}')
    print(f'z range precalculating over = {0} - {z_max}')
    print(f'number of intervals precalculating at = {M}')
    
    for j in range(nfrbs):
        # Loop over fields
        for i in range(nfields):
            field = fields[i,:,:,:]
            # Produce the unscaled los integral at each z interval, the density contrast at each interval,  and whether that interval has exceeded the box              
            integrals, exceeds, density  = los_int_ccl(field,L,xmin,ra[j],dec[j],z_ints)
            tabulated_los_int[i,j,:] = integrals 
            tabulated_exceeds[j,:] = exceeds
            tabulated_den[i,j,:] = density
            tabulated_los_int_grad[i,j,:] = np.gradient(integrals,z_ints)
    
        # Calculate the unconstrained contribution to the fluctuation part of the DM (only loop over FRBs here as it is the same for all fields)
        for k in range(len(z_ints)):
            if tabulated_exceeds[j,k] == 0:
                continue
            else:
                # The function name is a bit misleading because I didn't understand it for a while.
                uncon = ccl_correction(z_ints[k],z_ints,tabulated_exceeds[j,:])
                tabulated_ccl[j,k] = uncon        
        # Get the gradient for the ccl correction            
        tabulated_ccl_grad[j,:] = np.gradient(tabulated_ccl[j,:],z_ints)
else:
    # Here, use a subset of the FRB data.
    # As we can always calculate for the full set and split up the full thing
    # This option can probably be ignored
    if load_mocks == False:
        start = 0 
        end = 50
        nfrbs = end - start
        print(f'Using frbs {start} - {end}')

    tabulated_z = np.zeros((M))                                                                                                                                          
    tabulated_los_int = np.zeros((nfields,nfrbs,M))
    tabulated_los_int_grad = np.zeros((nfields,nfrbs,M))
    tabulated_exceeds = np.zeros((nfrbs,M))
    tabulated_ccl = np.zeros((nfrbs,M))
    tabulated_ccl_grad = np.zeros((nfrbs,M))
    
    tabulated_z[:] = z_ints
    
    print(f'nfrbs = {nfrbs}')
    print(f'nfields = {nfields}')
    print(f'z range precalculating over = {0} - {z_max}')
    print(f'number of intervals precalculating at = {M}')
    
    for j in range(nfrbs):                                                                                                                                               
        # Loop over fields
        for i in range(nfields):
            field = fields[i,:,:,:]
            # Produce the unscaled los integral at each z interval, the density contrast at each interval,  and whether that interval has exceed the box              
            integrals, exceeds, density  = los_int_ccl(field,L,xmin,ra[j],dec[j],z_ints)
            tabulated_los_int[i,j,:] = integrals 
            tabulated_exceeds[j,:] = exceeds
            #tabulated_den[i,j,:] = density
            tabulated_los_int_grad[i,j,:] = np.gradient(integrals,z_ints)
    
        # Do the ccl correction (only loop over FRBs here as it is the same for all fields)
        for k in range(len(z_ints)):
            if tabulated_exceeds[j,k] == 0:
                continue
            else:
                uncon = ccl_correction(z_ints[k],z_ints,tabulated_exceeds[j,:])
                tabulated_ccl[j,k] = uncon        
        # Get the gradient for the ccl correction            
        tabulated_ccl_grad[j,:] = np.gradient(tabulated_ccl[j,:],z_ints)


# Save the results of the integration
# Naming convention: 
# 'mocks' = mock data
# sdss/localised = that type of FRB
# {labs[0]}_{labs[-1]} = range of SDSS BOSS indices used to calculate the integrals

if make_mocks == True:
    if mock_field == True:
        np.savez('tabulated_integrals_mocks',tabulated_integrals=tabulated_los_int,tabulated_z=tabulated_z,tabulated_exceeds=tabulated_exceeds, tabulated_ccl = tabulated_ccl, tabulated_los_int_grad = tabulated_los_int_grad, tabulated_ccl_grad = tabulated_ccl_grad)
    else:
        if load_mocks == False:
            if use_subset == False:
                np.savez(f'tabulated_integrals_mocks_{labs[0]}_{labs[-1]}',tabulated_integrals=tabulated_los_int,tabulated_z=tabulated_z,tabulated_exceeds=tabulated_exceeds, tabulated_ccl = tabulated_ccl, tabulated_los_int_grad = tabulated_los_int_grad, tabulated_ccl_grad = tabulated_ccl_grad)
            else:
                # Label file with both the fields calculated over, and the FRB index
                np.savez(f'tabulated_integrals_mocks_{labs[0]}_{labs[-1]}_{start}_{end}',tabulated_integrals=tabulated_los_int,tabulated_z=tabulated_z,tabulated_exceeds=tabulated_exceeds, tabulated_ccl = tabulated_ccl, tabulated_los_int_grad = tabulated_los_int_grad, tabulated_ccl_grad = tabulated_ccl_grad)
        else:
            # Label file with both the fields calculated over, and the FRB index
            np.savez(f'tabulated_integrals_mocks_{labs[0]}_{labs[-1]}_{start}_{end}',tabulated_integrals=tabulated_los_int,tabulated_z=tabulated_z,tabulated_exceeds=tabulated_exceeds, tabulated_ccl = tabulated_ccl, tabulated_los_int_grad = tabulated_los_int_grad, tabulated_ccl_grad = tabulated_ccl_grad)


else:
    np.savez(f'tabulated_integrals_sdss_{labs[0]}_{labs[-1]}',tabulated_integrals=tabulated_los_int,tabulated_z=tabulated_z,tabulated_exceeds=tabulated_exceeds, tabulated_ccl = tabulated_ccl)

#######
## Diagnostics

# Print some outputs
print('integrals precalculated')
print('tabulated integral for field 0, frb 0 =')
print(tabulated_los_int[0,0,:])
print(f'unconstrained variance for frb 0 =')
print(tabulated_ccl[0,:])

# Generate diagnostic figures for the integrals you've made
# NOTE: this aren't scaled by parameters, so the values are unphysical. But, that shouldn't matter in general. 

# Plot 1: for singles field, plot the integrals as a function of redshift to see what they look like
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,ncols=2)
for i in range(nfrbs):
    ax1.plot(tabulated_z,tabulated_los_int[0,i,:])
    ax2.plot(tabulated_z,tabulated_los_int[1,i,:])
    ax3.plot(tabulated_z,tabulated_los_int[2,i,:])
    ax4.plot(tabulated_z,tabulated_los_int[3,i,:])
ax1.set_xlabel('z')
ax2.set_xlabel('z')
ax3.set_xlabel('z')
ax4.set_xlabel('z')
ax1.set_ylabel('Los Int')
ax2.set_ylabel('Los Int')
ax3.set_ylabel('Los Int')
ax4.set_ylabel('Los Int')
ax1.set_title('Field 0')
ax2.set_title('Field 0')
ax3.set_title('Field 0')
ax4.set_title('Field 0')
plt.savefig('diag_1.png')

# Plot 2: for a given frb, plot the integral for different fields
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,ncols=2)
for i in range(4):
    ax1.plot(tabulated_z,tabulated_los_int[i,0,:])
    ax2.plot(tabulated_z,tabulated_los_int[i,1,:])
    ax3.plot(tabulated_z,tabulated_los_int[i,2,:])
    ax4.plot(tabulated_z,tabulated_los_int[i,3,:])
ax1.set_xlabel('z')
ax2.set_xlabel('z')
ax3.set_xlabel('z')
ax4.set_xlabel('z')
ax1.set_ylabel('Los Int')
ax2.set_ylabel('Los Int')
ax3.set_ylabel('Los Int')
ax4.set_ylabel('Los Int')
ax1.set_title('FRB 0')
ax2.set_title('FRB 1')
ax3.set_title('FRB 2')
ax4.set_title('FRB 4')
plt.savefig('diag_2.png')


# Plot 3: for a given field, at a given redshift, we histogram the los integrals, to see if they centre on zero. If so, we expect it to struggle to get the bias here (the term is negligible)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,ncols=2)
#for i in range(nfrbs):
ax1.hist(tabulated_los_int[0,:,-1])
ax2.hist(tabulated_los_int[1,:,-1])
ax3.hist(tabulated_los_int[2,:,-1])
ax4.hist(tabulated_los_int[3,:,-1])
ax1.plot(np.mean(tabulated_los_int[0,:,-1])*np.ones((len(np.arange(0,30)))),np.arange(0,30))
ax2.plot(np.mean(tabulated_los_int[1,:,-1])*np.ones((len(np.arange(0,30)))),np.arange(0,30))
ax3.plot(np.mean(tabulated_los_int[2,:,-1])*np.ones((len(np.arange(0,30)))),np.arange(0,30))
ax4.plot(np.mean(tabulated_los_int[3,:,-1])*np.ones((len(np.arange(0,30)))),np.arange(0,30))
ax1.set_xlabel('Hist')
ax2.set_xlabel('Hist')
ax3.set_xlabel('Hist')
ax4.set_xlabel('Hist')
ax1.set_xlabel('Los Int')
ax2.set_xlabel('Los Int')
ax3.set_xlabel('Los Int')
ax4.set_xlabel('Los Int')
ax1.set_title('FRB 0')
ax2.set_title('FRB 1')
ax3.set_title('FRB 2')
ax4.set_title('FRB 4')
plt.savefig('diag_3.png')
