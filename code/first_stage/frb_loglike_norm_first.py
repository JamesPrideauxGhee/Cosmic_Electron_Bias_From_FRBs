# -*- coding: utf-8 -*-

import jax.numpy as jnp
from jax import lax, vmap, random
import numpyro
import numpyro.distributions as dist
import astropy as ap
import astropy.units as apu
import matplotlib.pyplot as plt
import corner
import numpy as np
from jax.config import config
#config.update('jax_debug_nans',True)

# Script to run the first inference stage
# Assuming all relevant files and modules are present, 
# This script can be run and should perform the full sampling
# and generate diagnostics


"""# **Forward and statistical model**"""

def bkgrnd_prefactor():
    # get the prefactor to the background contribution
    prefactor = 3 * H0 * ap.constants.c  / (8 * jnp.pi * ap.constants.G * ap.constants.m_p)
    prefactor = prefactor.to(apu.pc / apu.cm**3)
    prefactor = prefactor.value
    return prefactor

def bkgrnd_integrand(x):
    # Integrand for background contribution integral
    return (1 + x)/ jnp.sqrt(Om*(1+x)**3 + 1 - Om)

def bkgrnd_integral(y):
    # Calculate background contribution integral
    x = jnp.linspace(0,y,1000)
    integrand = bkgrnd_integrand(x)
    return jnp.trapz(integrand, x, axis=0)
    
    
def fluct_prefactor():
    # Get prefactor for fluctuation part
    prefactor = 3 * H0**2 / (8 * jnp.pi * ap.constants.G * ap.constants.m_p)
    prefactor = prefactor.to(1 / apu.cm**3)
    prefactor = prefactor.value
    return prefactor
    
    
def get_interp(z, tabulated_vals):
    # General interpolation function
    return jnp.interp(z,tabulated_z,tabulated_vals)
v_get_interp = vmap(get_interp, in_axes=(0,0), out_axes=(0))
 

def fluct_integral(z, i, tabulated_integral, tabulated_z):
    # Compare the drawn z values to the tabulated ones, and extract the precalculated integral values here
    return v_get_interp(z, tabulated_integral[i,:,:])
    

def uncon_integral(z, tabulated_ccl, tabulated_z):
    return v_get_interp(z, tabulated_ccl)

def make_mock(ztrue, obchif_true, muh_true, sigmah_true, obchif_bias_true, dm_err):

    nfrb = len(ztrue)    
    dm_los = bkgrnd_prefactor() * obchif_true * bkgrnd_integral(ztrue)
    f_prefactor = fluct_prefactor() * obchif_bias_true
    dm_fluct = f_prefactor * fluct_integral(ztrue, borg_index, tabulated_integral, tabulated_z) * 1.e6
    # TEST
    print(f'ztrue = ')
    print(ztrue)
    print(f'dm_los = ')
    print(dm_los)
    print(f'dm_fluct = ')
    print(dm_fluct)
    dm_los = random.normal(random.PRNGKey(1), (nfrb,)) * dm_err + dm_los + dm_fluct
    
    dm_host = random.normal(random.PRNGKey(2), (nfrb,)) * sigmah_true + muh_true
    # 1/(1+z) factor
    dm_host = dm_host.at[:].divide(1+ztrue[:])
    
    sig_uncon = jnp.sqrt(uncon_integral(ztrue, tabulated_ccl, tabulated_z)) * f_prefactor * 1.e6
    dm_uncon = random.normal(random.PRNGKey(3), (nfrb,)) * sig_uncon
    
    dm_tot = dm_los + dm_host + dm_uncon
       
    return dm_tot
   
# Get loglike from sample means
def get_loglike(obchif,muh,sigmah,obchif_bias):
    dm_b = bkgrnd_prefactor() * bkgrnd_integral(ztrue) * obchif
    # Host: array of length nfrb
    dm_h = muh/(1+ztrue)
                                                                                                     
    dm_pred = dm_b + dm_h #+ dm_fluct
                                                                                                     
    # Variance 
    # host: array of length nfrb
    sig_host2 = (sigmah/(1+ztrue))**2 
   
    f_prefactor = fluct_prefactor() * obchif_bias
    sig_uncon = jnp.sqrt(uncon_integral(z, tabulated_ccl, tabulated_z)) * f_prefactor * 1.e6 * littleh * 2
 
    dm_sig2 = dm_err**2 + dm_mw_err**2 + sig_host2 + sig_uncon**2

    ll = mylike(dm_pred,dm_sig2)
    loglike = ll.log_prob(dm_meas)
    print(f'Total loglike = {jnp.sum(loglike)}')
    return jnp.sum(loglike)   

    
def closest_arg(ymeas, yall):
    return jnp.argmin(jnp.abs(yall - ymeas))
v_closest_arg = vmap(closest_arg, in_axes=(0, None), out_axes=(0))

def divide_1_z(arr,z_arr):
    div_arr = jnp.zeros((len(arr)))
    div_arr = arr.at[:].divide(1+z_arr)
    return div_arr

# Cycle over FRBs
def do_like(mu, sig2): 
    like = -0.5 * mu**2/(sig2) - 0.5*jnp.log(2 * jnp.pi * sig2)
    return like
v_do_like = vmap(do_like, in_axes=(0,0), out_axes=(0))

   
class mylike(dist.Distribution):
    support = dist.constraints.real

    def __init__(self, mu, sig2): 
        self.mu, self.sig2, = dist.util.promote_shapes(mu, sig2)
        batch_shape = lax.broadcast_shapes(jnp.shape(mu), jnp.shape(sig2))
        super(mylike, self).__init__(batch_shape = batch_shape)
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
        
    def log_prob(self, value):
        mu = self.mu
        sig2 = self.sig2
         
        like = v_do_like(mu-value, sig2)
        print(f'like = {like}')
        return like

def model():
    # First stage: sample obchif, muh, sigmah

    # Sample bias from prior
    obchif = numpyro.sample("obchif", dist.Uniform(obchif_min, obchif_max))
    muh = numpyro.sample("muh", dist.Uniform(muh_min, muh_max))
    sigmah = numpyro.sample("sigmah", dist.Uniform(sigmah_min, sigmah_max))
    obchif_bias = numpyro.sample("obchif_bias", dist.Uniform(obchif_bias_min, obchif_bias_max))

    # Sample redshift
    #if nfrb > nlocal:
    #    z_unknown = numpyro.sample("z", dist.Uniform(z_min, z_max), sample_shape=(nfrb - nlocal,))
    #    z = jnp.concatenate([z_known,z_unknown])
    #else:
    #    z = z_known
    z = z_known    

    # Construct quantities for likelihood.
    # Mean: Note: we do not include a fluctuation component in this stage
    # Background: array of length nfrb
    dm_b = bkgrnd_prefactor() * bkgrnd_integral(z) * obchif
    # Host: array of length nfrb
    dm_h = muh/(1+z)
   
    # TEST: add in fluct part - does the inference work?
    f_prefactor = fluct_prefactor() * obchif_bias
    #dm_fluct = f_prefactor * fluct_integral(z, borg_index, tabulated_integral, tabulated_z) * 1.e6 
    sig_uncon = jnp.sqrt(uncon_integral(z, tabulated_ccl, tabulated_z)) * f_prefactor * 1.e6 * littleh * 2
 
    dm_pred = dm_b + dm_h #+ dm_fluct

    # Variance 
    # host: array of length nfrb
    sig_host2 = (sigmah/(1+z))**2 
    
    dm_sig2 = dm_err**2 + dm_mw_err**2 + sig_host2 + sig_uncon**2
    print(f'dm_pred = {dm_pred}')
    print(f'dm_sig2 = {dm_sig2}')

    numpyro.sample("obs", mylike(dm_pred, dm_sig2), obs=dm_meas)


##############
"""# **Set up inference parameters**"""

# PRIORS
obchif_min, obchif_max = 0, 0.1
obchif_bias_min, obchif_bias_max = -0.5, 0.5
muh_min, muh_max = 0, 500
sigmah_min, sigmah_max = 0, 400
z_min, z_max = 0, 1.5

# TRUTHS
H0 = 67.74 * apu.km / apu.s / apu.Mpc
littleh = H0.value/100
Om = 0.3089
obchif_true = 0.04 #0.3
obchif_bias_true = 0.04 * 1
host_mean = 500
host_std = 300
muh_true = 150 #jnp.log(host_mean ** 2 / jnp.sqrt(host_mean ** 2 + host_std ** 2))
sigmah_true = 20 #jnp.log(1 + host_std ** 2 / host_mean ** 2)
dm_err = 5


# HYPERPARAMETERS
delta_t = 10            # Spacing of t grid
delta_z = 0.01          #Â Spacing for initial z guess
tmin = 0                # Min value for convolution
tmax = 3.e4             # Max value for convolution
num_samples = 50000
num_warmup = 1500
borg_index = 0
nlocal = 100
nunknown = 0
imin = 0              # where to start mock (not important is shuffle_mock=True)
shuffle_mock = False

# DIRECTORIES
mock_path = './'
fig_dir = './'
data_path = './'

print(f'true parameters: obchif = {obchif_true}, muh = {muh_true}, sigmah = {sigmah_true}, obchif_bias = {obchif_bias_true}')

"""# **Make mocks**"""

# Load mock data
"""
data = jnp.load(mock_path + '/mock_data_z005.npz')
use_idx = jnp.ones(len(data['z']), dtype=bool)
use_idx = jnp.zeros(len(data['z']), dtype=bool)
use_idx = use_idx.at[imin:imin+nlocal+nunknown].set(True)
if shuffle_mock:
    use_idx = random.permutation(random.PRNGKey(4), use_idx, independent=True)
ztrue = jnp.array(data['z'])[:100]
nfrb = len(ztrue)
z_known = ztrue[:nlocal]

print('NFRB', nfrb)
print('NLOCAL', nlocal)

# Load BORG data
# Single file case

data = jnp.load(mock_path + 'tabulated_integrals_mocks_6000_9000.npz')
tabulated_z = jnp.array(data['tabulated_z'])
tabulated_integral = jnp.array(data['tabulated_integrals'])[:,use_idx,:]
tabulated_exceeds = jnp.array(data['tabulated_exceeds'])[use_idx,:]
tabulated_ccl = jnp.array(data['tabulated_ccl'])[use_idx,:]
"""

# Multiple file case
# Note: we need as many FRBs here as we specify above
"""
common = 'tabulated_integrals_mocks_6000_9000_'

index = '0_50'
data = jnp.load(mock_path + common + index + '.npz')
tabulated_z_1 = jnp.array(data['tabulated_z'])
tabulated_integral_1 = jnp.array(data['tabulated_integrals'])
tabulated_exceeds_1 = jnp.array(data['tabulated_exceeds'])
tabulated_ccl_1 = jnp.array(data['tabulated_ccl'])
index = '50_100'
data = jnp.load(mock_path + common + index + '.npz')
tabulated_z_2 = jnp.array(data['tabulated_z'])
tabulated_integral_2 = jnp.array(data['tabulated_integrals'])
tabulated_exceeds_2 = jnp.array(data['tabulated_exceeds'])    
tabulated_ccl_2 = jnp.array(data['tabulated_ccl'])
index = '100_150'
data = jnp.load(mock_path + common + index + '.npz')
tabulated_z_3 = jnp.array(data['tabulated_z'])
tabulated_integral_3 = jnp.array(data['tabulated_integrals'])
tabulated_exceeds_3 = jnp.array(data['tabulated_exceeds'])    
tabulated_ccl_3 = jnp.array(data['tabulated_ccl'])

# Combine into single files
tabulated_z = tabulated_z_1 # All FRBs have the same z intervals
tabulated_integral = np.concatenate((tabulated_integral_1,tabulated_integral_2,tabulated_integral_3),axis=1)
tabulated_exceeds = np.concatenate((tabulated_exceeds_1,tabulated_exceeds_2,tabulated_exceeds_3),axis=0)
tabulated_ccl = np.concatenate((tabulated_ccl_1,tabulated_ccl_2,tabulated_ccl_3),axis=0)
print(f'concatenated shapes: integral = {np.shape(tabulated_integral)}, exceeds = {np.shape(tabulated_exceeds)}, ccl = {np.shape(tabulated_ccl)}')
"""

# Combine to make mock
# Make mock data from all loaded mocks
"""
dm_meas = make_mock(ztrue, obchif_true, muh_true, sigmah_true, obchif_bias_true, dm_err)
# Save mock data for future use
file = 'mock_dm_z005'
np.savez(file,dm_obs=dm_meas)
# Extract subset to run with
nfrb = nlocal
ztrue = ztrue[:nlocal]
tabulated_integral = tabulated_integral[:,:nlocal,:]
tabulated_exdeed = tabulated_exceeds[:nlocal,:]
tabulated_ccl = tabulated_ccl[:nlocal,:] 
dm_meas = dm_meas[:nlocal]
z_known = ztrue[:nlocal]

# Load data: either real data or previously made mocks
#file = 'mock_dm.npz'
#hf = np.load(file)
#dm_meas = hf['dm_obs'][use_idx]
dm_mw_err = 0


print('DM range:', dm_meas.min(), dm_meas.max())
"""


"""# **First Stage inference - infer obchif, muh, sigmah from localised frbs**"""

# Load localised frbs
hf = jnp.load(data_path+'/localised_frbs_20_ne2001.npz')
data = hf['data']

dm_meas = data[:,2]
ztrue = data[:,3]
z_known = ztrue
nfrb = len(ztrue)
nlocal = nfrb

# Load integrals (but we don't use them here!)
data = jnp.load(mock_path + 'tabulated_integrals_localised_6000_6000.npz')
tabulated_z = jnp.array(data['tabulated_z'])
tabulated_integral = jnp.array(data['tabulated_integrals'])
tabulated_exceeds = jnp.array(data['tabulated_exceeds'])
tabulated_ccl = jnp.array(data['tabulated_ccl'])

# Set true obchif_bias to 0 so we ignore integral information
obchif_bias_true = 0

# Set up error for Milky Way contribution
dm_mw_err = 30

# Set up truths for initialisation
#obchif_true = 0.03
#muh_true = 5
#sigmah_true = 0.6
#obchif_bias_true = 0
#ztrue = ztrue

# CHECK THAT THE OBCHIF_BIAS AND Z SAMPLING STEPS ARE OFF!
print('NFRB', nfrb)
print('NLOCAL', nlocal)
print('dm_meas', dm_meas)
print('ztrue', ztrue)


"""# **Initialise and run**

Initialise to the true values: useful mock test
"""
"""
# Run sampler
rng_key = random.PRNGKey(6)
rng_key, rng_key_ = random.split(rng_key)
values={'muh':muh_true,'sigmah':sigmah_true,'obchif':obchif_true,'obchif_bias':obchif_bias_true,'z':ztrue[nlocal:]}
kernel = numpyro.infer.NUTS(model,
            init_strategy=numpyro.infer.initialization.init_to_value(values=values))
mcmc = numpyro.infer.MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(rng_key_)

mcmc.print_summary()"""
"""Initialise to random position within the prior: useful for data"""

# Run sampler
val = 6
rng_key = random.PRNGKey(val)
rng_key, rng_key_ = random.split(rng_key)
kernel = numpyro.infer.NUTS(model)
print(f'Init to prior, prngkey = {val}')

#values={'muh':muh_true,'sigmah':sigmah_true,'obchif':obchif_true, 'obchif_bias':obchif_bias_true,'z':ztrue[nlocal:]}
#kernel = numpyro.infer.NUTS(model,init_strategy=numpyro.infer.initialization.init_to_value(values=values))
#print(f'Init to values)

#kernel = numpyro.infer.NUTS(model,init_strategy=numpyro.infer.initialization.init_to_median())
#print(f'Init to median')


mcmc = numpyro.infer.MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(rng_key_)

mcmc.print_summary()
#print(f'True values: obchif = {obchif_true}, muh = {muh_true}, sigmah = {sigmah_true}')

"""# **Extract Samples**"""

# Convert samples into a single array
samples = mcmc.get_samples()
nparam = len(samples.keys())
if nfrb > nlocal:
    nparam += nfrb - nlocal - 1
samps = jnp.empty((num_samples, nparam))
param_names = ['obchif', 'muh', 'sigmah','obchif_bias'] #['global_params','obchif_bias']
param_labels = [r'$\Omega_{\rm b} \chi f$', r'$\mu_{\rm h}$', r'$\sigma_{\rm h}$',r'$\Omega_{\rm b} \chi f b$']

# Save samples
np.savez('first_stage_constraints',obchif_samples=samples['obchif'],muh_samples=samples['muh'],sigmah_samples=samples['sigmah'],obchif_bias_samples=samples['obchif_bias'])


for i, p in enumerate(param_names):
    samps = samps.at[:,i].set(samples[p])
if nfrb > nlocal:
    samps = samps.at[:,len(param_names):].set(samples['z'])
    param_names = param_names + ['z%i'%i for i in range(nfrb-nlocal)]
    param_labels = param_labels + [r'$z_{%i}$'%i for i in range(nfrb-nlocal)]
samps = np.asarray(samps)

truths = list([obchif_true, muh_true, sigmah_true]) + list(ztrue[nlocal:])
truths = [float(f) for f in truths]

"""## **Plotting Scripts**"""

#Â Trace plot of non-redshift quantities
fig1, axs1 = plt.subplots(4, 1, figsize=(6,7), sharex=True)
for i in range(4):
    axs1[i].plot(samps[:,i])
    axs1[i].set_ylabel(param_labels[i])
    #axs1[i].axhline(truths[i], color='k')
axs1[-1].set_xlabel('Step Number')
fig1.tight_layout()
fig1.savefig(fig_dir + '/trace.png')

# Corner plot
fig2, axs2 = plt.subplots(4, 4, figsize=(8,8))
if nfrb > nlocal:
    corner.corner(
        samps[:,:-(nfrb-nlocal)],
        labels=param_labels[:-(nfrb-nlocal)],
        fig=fig2,
        truths=truths[:-(nfrb-nlocal)]
    )
else:
    corner.corner(
        samps,
        labels=param_labels,
        fig=fig2
    )
fig2.savefig(fig_dir + '/first_stage_corner.png')

# Redshift residuals plot
if nfrb > nlocal:
    fig3, axs3 = plt.subplots(1, 1, figsize=(8,4))
    median = np.median(samps, axis=0)
    sigma_plus = np.percentile(samps, 84, axis=0) - median
    sigma_minus = median - np.percentile(samps, 16, axis=0)
    median = median[-(nfrb-nlocal):]
    sigma_plus = sigma_plus[-(nfrb-nlocal):]
    sigma_minus = sigma_minus[-(nfrb-nlocal):]
    x = ztrue[nlocal:]
    y = median - x
    plot_kwargs = {'fmt':'.', 'markersize':3, 'zorder':-1,
                 'capsize':1, 'elinewidth':1, 'color':'k', 'alpha':1}
    axs3.errorbar(x, y, yerr=[sigma_minus, sigma_plus], **plot_kwargs)
    axs3.axhline(0, color='k')
    axs3.set_xlabel(r'$z_{\rm true}$')
    axs3.set_xlabel(r'$z_{\rm infer} - z_{\rm true}$')
    fig3.tight_layout()
    fig3.savefig(fig_dir + '/inferred.png')

plt.show()

ll = get_loglike(jnp.mean(samples['obchif']),jnp.mean(samples['muh']),jnp.mean(samples['sigmah']),jnp.mean(samples['obchif_bias']))
