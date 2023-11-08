import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt


cosmo =  ccl.Cosmology(h = 0.6774, Omega_b = 0.04865, Omega_c = 0.3089, n_s = 0.9667, sigma8 = 0.8159)

z_range = np.linspace(0,3,1000)
chi_range = np.zeros((1000))
for i in range(1000):
    dist = ccl.comoving_radial_distance(cosmo,1/(1+z_range[i]))
    chi_range[i] = dist

kernel = np.zeros((2,1000))
kernel[0] = chi_range #z_range
kernel[1] = ccl.growth_factor(cosmo,(1./(1+z_range))) * (1 + z_range)

custom = ccl.Tracer()
custom.add_tracer(cosmo, kernel=kernel, der_angles=0, der_bessel=0)

max_ells = [500000,1000000,1500000,2000000,2500000,3000000,3500000,4000000,4500000,5000000,6000000,7000000,8000000]
dm_uncon_2 = np.zeros((len(max_ells)))
for i in range(len(max_ells)):
    # Create l array
    ells = np.arange(0,max_ells[i])
    # Create scale factor array from z values
    sf = (1./(1+z_range))[::-1]
    # Create array of k modes from BORG box modes
    ks = np.logspace(2*np.pi/4000,2*np.pi*256/4000,256)
    
    # Construct a power spectrum
    lpk_arr = np.log(np.array([ccl.nonlin_matter_power(cosmo,ks,a) for a in sf]))
    # Create a k space cut
    pk_cut = ccl.Pk2D(a_arr=sf, lk_arr= np.log(ks), pk_arr = lpk_arr, is_logp=True)
    # Get the cls from the power spectrum
    cls = ccl.angular_cl(cosmo,custom,custom,ells, p_of_k_a = pk_cut)
    # Calculate the cls for this case
    dm_uncon_2[i] = np.sum( ( (2*ells + 1) * cls) / (4*np.pi) )
    other = ccl.correlations.correlation(cosmo,ells,cls,theta=0,method='Bessel')
    print(f'sum cls = {dm_uncon_2[i]}, other = {other}')

# Plot the results
plt.figure()
plt.plot(max_ells,dm_uncon_2,'+')
plt.xlabel('max l')
plt.ylabel('dm_uncon_2')
plt.savefig('l_cl_convergence_2.png')

# Save results
np.savez('dm_uncon_convergence',max_ells=max_ells,dm_uncon_2=dm_uncon_2)

