import numpy as np
from texttable import Texttable
import latextable

# Load bits and bobs
file = 'full_second_stage_constraints_covar.npz'
hf = np.load(file)
z_samples = hf['z_samples']

mean_z = np.mean(z_samples, axis=0)
std_z = np.std(z_samples, axis=0)

file = 'sdss_labels_no_repeats.npz'
hf = np.load(file, allow_pickle=True)
labels = np.array(hf['labels'],dtype=object)

N = len(mean_z)

# Construct the array that we use to make the table


tab_arr = np.zeros((N//4+1,3*4),dtype=object)
print(tab_arr.shape)
tab_arr[0,:] = ['FRB Label', 'Mean redshift', 'Std. redshift','FRB Label', 'Mean redshift', 'Std. redshift','FRB Label', 'Mean redshift', 'Std. redshift','FRB Label', 'Mean redshift', 'Std. redshift']
for i in range(N//4):
    tab_arr[i+1,0] = labels[i]
    tab_arr[i+1,1] = str(mean_z[i])
    tab_arr[i+1,2] = str(std_z[i])
for i in range(N//4,N//2):
    tab_arr[i+1-N//4,3] = labels[i]
    tab_arr[i+1-N//4,4] = str(mean_z[i])
    tab_arr[i+1-N//4,5] = str(std_z[i])
for i in range(N//2,3*N//4):
    tab_arr[i+1-N//2,6] = labels[i]
    tab_arr[i+1-N//2,7] = str(mean_z[i])
    tab_arr[i+1-N//2,8] = str(std_z[i])
for i in range(3*N//4,N):
    tab_arr[i+1-3*N//4,9] = labels[i]
    tab_arr[i+1-3*N//4,10] = str(mean_z[i])
    tab_arr[i+1-3*N//4,11] = str(std_z[i])
table = Texttable()
table.set_cols_align(['c'] * 12)
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(tab_arr)


"""
tab_arr = np.zeros((N//2+1,3*2),dtype=object)
print(tab_arr.shape)
tab_arr[0,:] = ['FRB Label', 'Mean redshift', 'Std. redshift','FRB Label', 'Mean redshift', 'Std. redshift']
for i in range(N//2):
    tab_arr[i+1,0] = labels[i]
    tab_arr[i+1,1] = str(mean_z[i])
    tab_arr[i+1,2] = str(std_z[i])
for i in range(N//2,N):
    tab_arr[i+1-N//2,3] = labels[i]
    tab_arr[i+1-N//2,4] = str(mean_z[i])
    tab_arr[i+1-N//2,5] = str(std_z[i])


table = Texttable()
table.set_cols_align(['c'] * 6)
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(tab_arr)
"""
print(table.draw())

print(latextable.draw_latex(table))


f = open('z_cons.txt','w')
f.write(latextable.draw_latex(table))
f.close()
