# Cosmic_Electron_Bias_From_FRBs

Repository for the code for the Cosmic Electron Bias from Fast Radio Bursts project.

File descriptions:

frb_loglike_norm_first.py - numpyro sampling script to run first stage
fit_to_first_stage.py - scikit-learn script to fit a Gaussian mixture model to the results of the first stage
frb_loglike_norm_second.py - numpyro sampling script to run second stage (must be run once per BORG field)
combine_constraints.py - combines the outputs of the second stage for each BORG field into a single file
write_latex_table.py - makes a table in latex format containing the redshift constraints

first_stage_constraints.npz - first stage constraints used in draft
gm_fit_test.npz - GMM fit to first stage used in draft
sdss_frbs_ne2001.npz - unlocalised FRBs (with Milky Way contribution subtracted)
sdss_labels_no_repeats.npz - CHIME labels for unlocalised FRBs
tabulated_integrals_sdss_6000_9000.npz - line-of-sight integrals for unlocalised FRBs for all BORG fields
