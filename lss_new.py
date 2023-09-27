import numpy as np
from scipy.integrate import quad
from numpy import pi
#import jax.numpy as jnp
import pyccl as ccl
from astropy.cosmology import FlatLambdaCDM

from numba import jit
import time


#### Python script to calculate the large-scale structure contribution to an FRB's DM

####
def z_to_r(z):
    """
    Calculates the comoving radial distance at some redshift (assumes Om is fixed here)
    Arguments:
    - z: float, redshift values for which we want the comoving distance
    Returns:
    - r: float, comoving distance given by r = c * int_(0)^(z) dz/E(z). Note this needs to be divided by H0 to get the true comoving distance
    """
    Om = 0.238
    def invE(z,Om):
        return 1/np.sqrt(Om * (1+ z)**3 + (1 - Om))
   
    c = 3*10**8
    return c * quad(invE,0,z,args=(Om))[0]
    
    # CODE TEST: try simple analytic relationships
    # r = 100 x z
    #return 100 * z

####

#### 
def tabulate_z():
    """ 
    Numerical inverse of the function 'z_to_r' below.
    A cheap tabulation method that works because we know the lower and upper bounds for the problem, ish
    """
    lower = 0.0
    upper = 1.5
    step = 0.000001
    z_vals = np.arange(lower,upper,step)
    r_at_z = np.zeros((len(z_vals)))
    for q in range(len(z_vals)):
        r_at_z[q] = z_to_r(z_vals[q])
        #r_at_z[q] = z_vals[q]
    return z_vals, r_at_z
####

# Set up global cosmo param values
Om = 0.3089
Ob = 0.048597
Ol = 1 - Om
H0 = 67.74
h = H0/100
sig8 = 0.8159
ns = 0.9667
# Set up the global cosmology object in astropy to use
cosmo = FlatLambdaCDM(H0=H0,Om0=Om)

####
def los_int_ccl(field,L,xmin,ra,dec,z_ints):
    """
    inputs:
    field: BORG density field, numpy array of size (N,N,N)
    z_ints: redshift integrals to calculate integrand at, numpy array of size (M)

    """

    # This version of the integral uses the ccl to estimate the contribution when the FRB lies outside the BORG box

    # Get number of voxels in field and physical voxel side length 
    N = field.shape[0]
    dx = L / N 


    # Get comoving distances of the intervals which we evaluate the integrand at
    # For good accuracy, we need z_ints to be finely spaced
    nr = len(z_ints)
    r_ints = np.zeros((nr))
    for i in range(nr):
        #r_ints[i] = z_to_r(z_ints[i])/(10*H0**2) #divide by 1000*H0 to get r in Mpc, then divide by h = |H0|/100 to get r in Mpc/h. As we don't include H0 dimensions here, we just need 10*H0**2
        # Astropy method:
        comoving_dist = cosmo.comoving_distance(z_ints[i]).value # in Mpc
        r_ints[i] = comoving_dist * h # in Mpc/h


    dr = np.diff(r_ints)

    # Get the Cartesian intervals corresponding to the z intervals we calculate at
    cartx_ints = r_ints * np.cos(ra*pi/180) * np.cos(dec*pi/180)                                                                   
    carty_ints = r_ints * np.sin(ra*pi/180) * np.cos(dec*pi/180)
    cartz_ints = r_ints * np.sin(dec*pi/180)                                                                                   
    #cartx_ints = l_ints * np.sin(ra*pi/180) * np.cos(dec*pi/180)                                                              
    #carty_ints = l_ints * np.sin(ra*pi/180) * np.sin(dec*pi/180)                                                              
    #cartz_ints = l_ints * np.cos(ra*pi/180)                                                                                   
    print(f'x values = {cartx_ints}')
    print(f'y values = {carty_ints}')
    print(f'z values = {cartz_ints}')
    # Find voxels in which each interval falls
    cartx = cartx_ints - xmin[0]                                                                                               
    carty = carty_ints - xmin[1]
    cartz = cartz_ints - xmin[2]                                                                                               
    hx = np.floor(cartx/dx)                                                                                                    
    hy = np.floor(carty/dx)                                                                                                    
    hz = np.floor(cartz/dx)                                                                                                    
    ix = (hx).astype(int)                                                                                                      
    iy = (hy).astype(int)                                                                                                      
    iz = (hz).astype(int)    
    print(f'Voxel x indices = {ix}')
    print(f'Voxel y indices = {iy}')
    print(f'Voxel z indices = {iz}')
    jx = ix + 1 
    jy = iy + 1                                                                                                                
    jz = iz + 1                                                                                                                
    
    # The snippet below may not be necessary if we are modelling beyond the box
    # Impose boundary condition                                                                                                
    #for i in range(nr):                                                                                                        
    #    if jx[i] >= N:                                                                                                         
    #        jx[i] -= N                                                                                                         
    #    if jy[i] >= N:                                                                                                         
    #        jy[i] -= N                                                                                                         
    #    if jz[i] >= N:                                                                                                         
    #        jz[i] -= N                                                                                                         
    #    if ix[i] >= N:                                                                                                         
    #        ix[i] -= N                                                                                                         
    #    if iy[i] >= N:                                                                                                         
    #        iy[i] -= N                                                                                                         
    #    if iz[i] >= N:                                                                                                         
    #        iz[i] -= N                                                                                                         
                                                                                                                           
                                                                                                                           
    # Get distances to the lowest index voxel vertex                                                                           
    rx = cartx/dx - hx                                                                                                         
    ry = carty/dx - hy                                                                                                         
    rz = cartz/dx - hz                                                                                                         
    # Get complementary distances.                                                                                             
    qx = 1 - rx                                                                                                                
    qy = 1 - ry                                                                                                                
    qz = 1 - rz                                                                                                                
                                                                                                                           
    qqq = qx*qy*qz                                                                                                             
    rqq = rx*qy*qz                                                                                                             
    qrq = qx*ry*qz                                                                                                             
    qqr = qx*qy*rz                                                                                                             
    rrq = rx*ry*qz                                                                                                             
    rqr = rx*qy*rz                                                                                                             
    qrr = qx*ry*rz
    rrr = rx*ry*rz

    # We now must loop over the intervals, and get the integrand here (density * (1 + z)).
    integrands, exceeds, densities = comp_los_loop_ccl(nr, ix, iy, iz, jx, jy, jz, N, field, qqq, rqq, qrq, qqr, rrq, rqr, qrr, rrr, z_ints)

    #print(f'integrands = {integrands}')

    ccl_cosmo = ccl.Cosmology(h=h, Omega_b=Ob, Omega_c=Om-Ob, Omega_k=0, n_s=ns, sigma8=sig8)
    integrands *= ccl.growth_factor(ccl_cosmo, (1./(1+z_ints)))

    #print(f'integrands = {integrands}')


    # Calculate integral via trapezium rule
    integrals = np.zeros((nr))
    for i in range(nr):
        # Use trapezium rule
        # All integrands beyond the box are 0, so we can safely use intervals beyond the box, as they just contribute nothing
        integrals[i] = np.trapz(integrands[:i+1],r_ints[:i+1])

    return integrals, exceeds, densities
####            


####
# Compiled loop over the increments in the los integral
@jit(nopython=True)
def comp_los_loop_ccl(nr, ix, iy, iz, jx, jy, jz, N, field, qqq, rqq, qrq, qqr, rrq, rqr, qrr, rrr, z_ints):
                                                                                                                                                                              
    # Use trilinear interpolation to weight density part for integral
    # Loop on intervals because we can't vectorise this part 
    # Initialise arrays to store the integrand, whether this interval is in the box, and the density contrast here
    integrands = np.zeros((nr)) 
    exceeds = np.zeros((nr))        
    densities = np.zeros((nr))                                                                                                                                               
    for i in range(nr):
                                                                                                                                                                              
        # Check whether the stepping has to exceed the box. If so, we finish this loop, and make a note
        # As the ccl probably won't run with jit, we must do the ccl calculation outside of this function 
        if (ix[i] < 0 or ix[i] > N-2 or iy[i] < 0 or iy[i] > N-2 or iz[i] < 0 or iz[i] > N-2):
            #print('exceeds box: periodic BCs applied.') 
            # Note that, when a case exceeds the box, we still need the box part, so we save that here
            # We note that it exceeds the box to find when we need to add the ccl contribution to this part
            #integrands[i] = integrands[i-1]
            
            # When the step exceeds the box, we set the integrand here to zero, because we no longer have any contribution. 
            # Doing so means that the trapezium rule value will stay the same beyond the box edge
            integrands[i] = 0
            exceeds[i] = 1
            
            continue

        # Begin by calculating delta(r) using trilinear interpolation
        density = field[ix[i],iy[i],iz[i]] * qqq[i] #qx[i] * qy[i] * qz[i]
        density += field[jx[i],iy[i],iz[i]] * rqq[i] #rx[i] * qy[i] * qz[i]
        density += field[ix[i],jy[i],iz[i]] * qrq[i] #qx[i] * ry[i] * qz[i]
        density += field[ix[i],iy[i],jz[i]] * qqr[i] #qx[i] * qy[i] * rz[i]
        density += field[jx[i],jy[i],iz[i]] * rrq[i] #rx[i] * ry[i] * qz[i]
        density += field[jx[i],iy[i],jz[i]] * rqr[i] #rx[i] * qy[i] * rz[i]
        density += field[ix[i],jy[i],jz[i]] * qrr[i] #qx[i] * ry[i] * rz[i]
        density += field[jx[i],jy[i],jz[i]] * rrr[i] #rx[i] * ry[i] * rz[i]
        
        # Get the redshift value that corresponds to the comoving distance we are currently calculating at 
        #z_of_r = z_values[np.argmin(np.abs(r_at_z - l_ints[i]))]
        z_of_r = z_ints[i]
        integrands[i] = density * (1 + z_of_r)      
        densities[i] = density    
            
    return integrands, exceeds, densities          
####

####
def ccl_correction(z, z_ints, exceeds):
    # Add in ccl calculation for parts where the interval exceeds the box
    # Find index of last interval in box
    # We use this to calculate the integral contribution to the box edge
    
    indices = np.nonzero(exceeds)
    #print(f' indices = {indices}')
    last_inside = indices[0][0] - 1
    #edge_integral = integrals[last_inside]
    
    index = np.argmin(np.abs(z - z_ints))
                                                                                      
    #nr = len(z_ints)
    # Then, we add the ccl contribution where needed
    # NEED TO FIND CORRECT COSMO PARAMS HERE
                                                                                          
    ccl_cosmo = ccl.Cosmology(h = h, Omega_b = Ob, Omega_c = Om-Ob, Omega_k = 0, n_s = ns, sigma8 = sig8)
    #dm_uncon_2 = np.zeros((nr))
    #for i in range(nr):
    #    if exceeds[i] == 0:
    #        continue
    #    else:
    custom = ccl.Tracer()                                                                                  
    # Radial kernel extends from box to here                                                              
    z_range = np.linspace(z_ints[last_inside],z_ints[index],1000)                                             
    chi_range = np.zeros((1000))
    for i in range(1000):
        comoving_dist = cosmo.comoving_distance(z_range[i]).value # This should be in Mpc
        chi_range[i] = comoving_dist
    #chi_range = z_to_r(z_range)/(1000*73.2)                                             
    kernel = np.zeros((2,1000))                                                                           
    kernel[0] = chi_range #z_range
    kernel[1] = ccl.growth_factor(ccl_cosmo,(1./(1+z_range))) * (1 + z_range)

    # Need to ask about the Bessel function here...                                                       
    custom.add_tracer(ccl_cosmo, kernel=kernel, der_angles=0, der_bessel=0)                                   
                                                                                                          
    # Then, we construct the power spectrum                                                               
    # Choose an l range                                                                                   
    # Need to think about the upper limit                                                                 
    ells = np.arange(0,3000)                                                                               
    # Impose k cut on power spectrum (just use modes in box)                                              
    # Copying the documentation example:                                                                  
    sf = (1./(1+z_range))[::-1]
    #print(f'sf = {sf}')                                                                           
    # Of course, we don't need the cut if we only use the set of modes we want in the first place         
    ks = np.logspace(2*pi/750,2*pi*256/750,256)                                                         
    #ks = np.logspace(0,2,256)
    lpk_arr = np.log(np.array([ccl.nonlin_matter_power(ccl_cosmo,ks,a) for a in sf]))                         
    pk_cut = ccl.Pk2D(a_arr=sf,lk_arr=np.log(ks), pk_arr = lpk_arr, is_logp=True)                         
    # Then, calculate the power spectrum using the k cut pspec.                                           
    cls = ccl.angular_cl(ccl_cosmo,custom,custom,ells, p_of_k_a = pk_cut) 
                                                                                                          
    # Sum contributions                                                                                   
    #dm_uncon_2 = np.sum( ( (2*ells + 1) * cls) / (4*pi) )                                              
    dm_uncon_2 = ccl.correlations.correlation(ccl_cosmo,ells,cls,theta=2,method='Bessel')  
    return dm_uncon_2
####

#### 
def cl_calc(z):
    # Calculate the unconstrained variance at some redshift.
    # Note, here, we calculate from z=0 to some value, rather than
    # from the edge of the box.

    ccl_cosmo = ccl.Cosmology(h = h, Omega_b = Ob, Omega_c = Om-Ob, Omega_k = 0, n_s = ns, sigma8 = sig8)

    custom = ccl.Tracer()                                                                                  

    # Radial kernel extends from 0 to z
    z_range = np.linspace(0,z,1000)                                             
    chi_range = np.zeros((1000))
    for i in range(1000):
        comoving_dist = cosmo.comoving_distance(z_range[i]).value # This should be in Mpc
        chi_range[i] = comoving_dist

    kernel = np.zeros((2,1000))                                                                           
    kernel[0] = chi_range 
    kernel[1] = ccl.growth_factor(ccl_cosmo,(1./(1+z_range))) * (1 + z_range)

    custom.add_tracer(ccl_cosmo, kernel=kernel, der_angles=0, der_bessel=0)                                   
                                                                                                          
    # Then, we construct the power spectrum                                                               
    # Choose an l range                                                                                   
    # Need to think about the upper limit                                                                 
    ells = np.arange(0,3000)                                                                               
    # Impose k cut on power spectrum (just use modes in box)                                              
    # Copying the documentation example:                                                                  
    sf = (1./(1+z_range))[::-1]
    #print(f'sf = {sf}')                                                                           
    # Of course, we don't need the cut if we only use the set of modes we want in the first place         
    ks = np.logspace(2*pi/4000,2*pi*256/4000,256)                                                         
    #ks = np.logspace(0,2,256)
    lpk_arr = np.log(np.array([ccl.nonlin_matter_power(ccl_cosmo,ks,a) for a in sf]))                         
    #kcut = 2*pi/750
    #lpk_arr -= (ks/kcut)**2
    #kcut = 2*256*pi/750
    #lpk_arr -= (ks/kcut)**2
    pk_cut = ccl.Pk2D(a_arr=sf,lk_arr=np.log(ks), pk_arr = lpk_arr, is_logp=True)                         
    # Then, calculate the power spectrum using the k cut pspec.                                           
    cls = ccl.angular_cl(ccl_cosmo,custom,custom,ells, p_of_k_a = pk_cut) 
                                                                                                          
    # Sum contributions                                                                                   
    #dm_uncon_2 = np.sum( ( (2*ells + 1) * cls) / (4*pi) )                                              
    dm_uncon_2 = ccl.correlations.correlation(ccl_cosmo,ells,cls,theta=2,method='Bessel')  
                                                                                                           
    return dm_uncon_2
####

####
def cl_calc2(z):
    # Calculate the unconstrained variance at some redshift.
    # Note, here, we calculate from z=0 to some value, rather than
    # from the edge of the box.

    ccl_cosmo = ccl.Cosmology(h = h, Omega_b = Ob, Omega_c = Om-Ob, Omega_k = 0, n_s = ns, sigma8 = sig8)

    custom = ccl.Tracer()                                                                                  

    # Radial kernel extends from 0 to z
    z_range = np.linspace(0,z,1000)                                             
    chi_range = np.zeros((1000))
    for i in range(1000):
        comoving_dist = cosmo.comoving_distance(z_range[i]).value # This should be in Mpc
        chi_range[i] = comoving_dist

    kernel = np.zeros((2,1000))                                                                           
    kernel[0] = chi_range 
    kernel[1] = ccl.growth_factor(ccl_cosmo,(1./(1+z_range))) * (1 + z_range)

    custom.add_tracer(ccl_cosmo, kernel=kernel, der_angles=0, der_bessel=0)                                   
                                                                                                          
    # Then, we construct the power spectrum                                                               
    # Choose an l range                                                                                   
    # Need to think about the upper limit                                                                 
    ells = np.arange(0,3000)                                                                               
    # Impose k cut on power spectrum (just use modes in box)                                              
    # Copying the documentation example:                                                                  
    sf = (1./(1+z_range))[::-1]
    #print(f'sf = {sf}')                                                                           
    # Of course, we don't need the cut if we only use the set of modes we want in the first place         
    ks = np.logspace(2*pi/750,2*pi*256/750,256)                                                         
    lpk_arr = np.log(np.array([ccl.nonlin_matter_power(ccl_cosmo,ks,a) for a in sf]))                         
    pk_cut = ccl.Pk2D(a_arr=sf,lk_arr=np.log(ks), pk_arr = lpk_arr, is_logp=True)                         
    # Then, calculate the power spectrum using the k cut pspec.                                           
    #cls = ccl.angular_cl(ccl_cosmo,custom,custom,ells, p_of_k_a = pk_cut) 
    cls = ccl.angular_cl(ccl_cosmo,custom,custom,ells)                                                                                                      
    # Sum contributions                                                                                   
    dm_uncon_2 = np.sum( ( (2*ells + 1) * cls) / (4*pi) )                                              

    # Or, use a built in method
    uncon1 = ccl.correlations.correlation(ccl_cosmo,ells,cls,theta=1,method='Legendre') 
    uncon2 = ccl.correlations.correlation(ccl_cosmo,ells,cls,theta=1,method='Bessel') 
    uncon3 = ccl.correlations.correlation(ccl_cosmo,ells,cls,theta=5,method='Legendre') 
    #print(f'dm_uncon_2 = {dm_uncon_2}')
    print(f'uncon1 = {uncon1}, theta = 1, Legendr')
    print(f'uncon2 = {uncon2}, theta = 1, Bessel')
    #print(f'uncon3 = {uncon3}, theta = 5')                                                                                                          
    return uncon1 #dm_uncon_2
####
