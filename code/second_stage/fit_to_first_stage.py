import numpy as np
from sklearn.mixture import GaussianMixture as GM
import corner
import matplotlib.pyplot as plt

# Load samples
file = 'first_stage_constraints.npz'
d = np.load(file)
#obs = d['obchif_samples']
#mus = d['muh_samples']
sigs = d['sigmah_samples']
obbs = d['obchif_bias_samples']

total_samples = np.zeros((np.shape(obbs)[0],2))
#total_samples[:,0] = obs
#total_samples[:,1] = mus
total_samples[:,0] = sigs
total_samples[:,1] = obbs

# Fit with 2 component Gaussian mixture
# Choose initial weights: even
weights_guess = np.asarray([0.5,0.5])
means_guess = np.asarray([[np.mean(sigs),np.mean(obbs[obbs>0])],[np.mean(sigs),np.mean(obbs[obbs<0])]])
print(f'means_guess = {means_guess}')

gm = GM(n_components=2,random_state=0, weights_init=weights_guess, means_init=means_guess).fit(total_samples)


means = gm.means_
print(f' fitted means = {means}')
covs = gm.covariances_
print(f' fitted covariances = {covs}')
weights = gm.weights_
print(f' fitted weights = {weights}')

# Save
np.savez('gm_fit_test',weights=weights,means=means,covs=covs)

# Generate samples from the mixture model
new_samples, comp_labs = gm.sample(1000)
#new_samples_mpos = new_samples[new_samples[:,1]>=0]
#new_samples_mspos = new_samples_mpos[new_samples_mpos[:,2]>=0]
print(f'Original length = {np.shape(new_samples)}, pruned length = {np.shape(new_samples_mspos)}')


# Save prior samples to use to compare to posterior
np.savez('prior_samples',samples=new_samples)

# Compare the two via a corner plot
#fig = plt.figure()
#plt.hist(obbs,bins=50,label='first stage')
#plt.hist(new_samples,bins=50,label='fit')
#plt.xlabel('obchif_bias')
#plt.savefig('gm_prior_fit.png')


labs = ['sigmah','obchifb']
fig = corner.corner(total_samples,labels=labs,color='red')
corner.corner(new_samples,labels=labs,color='blue',fig=fig)
plt.savefig('gm_mix_corner_test.png')










