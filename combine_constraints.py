import numpy as np
import corner
import matplotlib.pyplot as plt

common = 'second_stage_constraints_'
inds = np.arange(0,16)

for ind in inds:
    con = np.load(common+f'{ind}'+'_med.npz')
    #obs = con['obchif_samples']
    #muhs = con['muh_samples']
    sigs = con['sigmah_samples']
    obcs = con['obchif_bias_samples']
    zs = con['z_samples']
    if ind == 0:
        #full_obs = obs
        #full_muhs = muhs
        full_sigs = sigs
        full_obcs = obcs
        full_zs = zs
        print(f'Shapes per field: obchif_bias = {np.shape(full_obcs)}, z = {np.shape(full_zs)}')
    else:
        #full_obs = np.concatenate((full_obs,obs))
        #full_muhs = np.concatenate((full_muhs,muhs))
        full_sigs = np.concatenate((full_sigs,sigs))
        full_obcs = np.concatenate((full_obcs,obcs))
        full_zs = np.concatenate((full_zs,zs))

print(f'Shapes: obchif_bias = {np.shape(full_obcs)}, z = {np.shape(full_zs)}')
#print(f'Means: obchif = {np.mean(full_obs)}, muh = {np.mean(full_muhs)}, sigmah = {np.mean(full_sigs)}, obchif_bias = {np.mean(full_obcs)}')
#print(f'Std dev: obchif = {np.std(full_obs)}, muh = {np.std(full_muhs)}, sigmah = {np.std(full_sigs)}, obchif_bias = {np.std(full_obcs)}')


file = 'full_second_stage_constraints.npz'
#np.savez(file,obchif_samples=full_obs,muh_samples=full_muhs,sigmah_samples=full_sigs,obchif_bias_samples=full_obcs,z_samples=full_zs)
np.savez(file,sigmah_samples=full_sigs,obchif_bias_samples=full_obcs,z_samples=full_zs)


total_samps = np.zeros((len(full_obcs),5))
#total_samps[:,0] = full_obs
#total_samps[:,1] = full_muhs
total_samps[:,0] = full_sigs
total_samps[:,1] = full_obcs
total_samps[:,2] = full_zs[:,0]
total_samps[:,3] = full_zs[:,1]
total_samps[:,4] = full_zs[:,2]
labels = ['sigmah','obchifb','z0','z1','z2']
total_samps_b = np.zeros((len(full_obcs),5))
#total_samps_b[:,0] = full_obs
#total_samps_b[:,1] = full_muhs
total_samps_b[:,0] = full_sigs
#total_samps_b[:,1] = full_obcs/full_obs
total_samps_b[:,2] = full_zs[:,0]
total_samps_b[:,3] = full_zs[:,1]
total_samps_b[:,4] = full_zs[:,2]
labels_b = ['sigmah','b','z0','z1','z2']

total_samps_so = np.zeros((len(full_obcs),2))
total_samps_so[:,0] = full_sigs
total_samps_so[:,1] = full_obcs


file = 'first_stage_constraints.npz'
con = np.load(file)
obs = con['obchif_samples']
muhs = con['muh_samples']
sigs = con['sigmah_samples']
obcs = con['obchif_bias_samples']
first_total_samps = np.empty((len(obcs),5))
#first_total_samps[:,0] = obs
#first_total_samps[:,1] = muhs
first_total_samps[:,0] = sigs
first_total_samps[:,1] = obcs
first_total_samps[:,2] = -1
first_total_samps[:,3] = -1
first_total_samps[:,4] = -1

first_total_samps_b = np.empty((len(obcs),7))
first_total_samps_b[:,0] = obs
first_total_samps_b[:,1] = muhs
first_total_samps_b[:,2] = sigs
#first_total_samps_b[:,3] = obcs/obs
first_total_samps_b[:,4] = -1
first_total_samps_b[:,5] = -1
first_total_samps_b[:,6] = -1

first_samps_so = np.zeros((len(obcs),2))
first_samps_so[:,0] = sigs
first_samps_so[:,1] = obcs

file = 'prior_samples_for_plot.npz'
con = np.load(file)
psamps = con['samples']

# 
np.savez('prior_second_samps',second_samples=total_samps_so,prior_samples=psamps)
print(f'prior and second stage samples saved!')


print(f'First stage:')
print(f'Min/max obchif = {np.amin(obs)}/{np.amax(obs)}')
print(f'Min/max muh = {np.amin(muhs)}/{np.amax(muhs)}')
print(f'Min/max sigmah = {np.amin(sigs)}/{np.amax(sigs)}')
print(f'Min/max obchif_bias = {np.amin(obcs)}/{np.amax(obcs)}')
print(f'Min/max bias = {np.amin(obcs/obs)}/{np.amax(obcs/obs)}')

print(f'Second stage:')
#print(f'Min/max obchif = {np.amin(full_obs)}/{np.amax(full_obs)}')
#print(f'Min/max muh = {np.amin(full_muhs)}/{np.amax(full_muhs)}')
#print(f'Min/max sigmah = {np.amin(full_sigs)}/{np.amax(full_sigs)}')
print(f'Min/max obchif_bias = {np.amin(full_obcs)}/{np.amax(full_obcs)}')
#print(f'Min/max obchif_bias = {np.amin(full_obcs/full_obs)}/{np.amax(full_obcs/full_obs)}')
print(f'Min/max z0 = {np.amin(full_zs[:,0])}/{np.amax(full_zs[:,0])}')
print(f'Min/max z1 = {np.amin(full_zs[:,1])}/{np.amax(full_zs[:,1])}')
print(f'Min/max z2 = {np.amin(full_zs[:,2])}/{np.amax(full_zs[:,2])}')


ranges = [(np.amin(obs),np.amax(obs)),(np.amin(muhs),np.amax(muhs)),(np.amin(sigs),np.amax(sigs)),(np.amin(obcs),np.amax(obcs)),(np.amin(full_zs[:,0]),np.amax(full_zs[:,0])),(np.amin(full_zs[:,1]),np.amax(full_zs[:,1])),(np.amin(full_zs[:,2]),np.amax(full_zs[:,2]))]

# Plot second stage samples
fig = corner.corner(total_samps,bin=50,labels=labels)
plt.savefig('full_second_stage_corner.png')


# Plot first and second stage samples
ranges = [(np.amin(sigs),np.amax(sigs)),(np.amin(obcs),np.amax(obcs)),(np.amin(full_zs[:,0]),np.amax(full_zs[:,0])),(np.amin(full_zs[:,1]),np.amax(full_zs[:,1])),(np.amin(full_zs[:,2]),np.amax(full_zs[:,2]))]

fig = corner.corner(first_samps_so,labels=labels,color='red')#,range=ranges,bins=20)
corner.corner(total_samps_so,labels=labels,fig=fig,color='blue')#,range=ranges,bins=20)
plt.savefig('full_first_second_stage_corner.png')


# Make z hist
fig = plt.figure()
plt.hist(full_zs.ravel(),bins=50,density=True)
plt.xlabel('z')
plt.ylabel('n(z)')
plt.savefig('full_second_stage_z_hist.png')


# Plot just electron bias
#fig = corner.corner(obcs/obs,labels=['b'],color='red',range=[np.amin(obcs/obs),np.amax(obcs/obs)],bins=50)
#corner.corner(full_obcs/full_obs,labels=['b'],fig=fig,color='blue',range=[np.amin(obcs/obs),np.amax(obcs/obs)],bins=50)
fig = plt.figure()
H1 = np.histogram(obcs/obs,density=True,bins=50,range=[np.amin(obcs/obs),np.amax(obcs/obs)])
H2 = np.histogram(full_obcs,density=True,range=[np.amin(full_obcs),np.amax(full_obcs/obs)],bins=50)

#plt.hist(obcs/obs,color='red',density=True,range=[np.amin(obcs/obs),np.amax(obcs/obs)],bins=50)
#plt.hist(full_obcs/full_obs,color='blue',density=True,range=[np.amin(obcs/obs),np.amax(obcs/obs)],bins=50)
plt.savefig('first_second_stage_b.png')


# Plot second stage samples (electron bias)
fig = corner.corner(total_samps_b,bin=50,labels=labels_b)
plt.savefig('full_second_stage_b_corner.png')



# Plot first and second stage samples (electron bias)
fig = corner.corner(first_total_samps_b,labels=labels_b,color='red',range=ranges,bins=50)
corner.corner(total_samps_b,labels=labels_b,fig=fig,color='blue',range=ranges,bins=50)
plt.savefig('full_first_second_stage_b_corner.png')

















