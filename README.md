
This repository is the working directory for my M.Sc. in High Performance Computing thesis "Chained Equations for incomplete climate networks". 

This thesis focused on accelerating the elastic net chained equations (ENCE) and multiple imputation by chained equations with direct use of regularized regression (MICE DURR) imputation methods developed in the paper: https://doi.org/10.1002/joc.8513. All experiments were run on daily rainfall records from the Republic of Ireland, collected from Met Éireann's network. The code in this repository does not run since access to the data requires permission from Met Éireann. 

For a presentation of the code running, see: [Parallel-Chained-Equations](https://github.com/mclane2/Parallel-Chained-Equations). This repository runs a comparison between the original R version, the synchronous R version and C implementation of ENCE and MICE DURR on simulated data.

**`cluster_code/`**
This directory contains the code used to obtain all the results of the thesis. All this code was run on the Seagull cluster, maintained by Research IT at Trinity College Dublin.

- `section_4.1/` - For the synchronous vs serial results, obtained entirely in R.
- `section_4.2/` - Contains the C implementation results.
- `section_4.3/` - Contains the method comparison results.
- `section_cov_updates/` - Repeated experiments for the C implementation after the covariance updates optimisation was added.

**`development_worklog/`**
This repository was my original working. It shows the development of the code over the course of the project.

**`original_implementation/`**
An unmodified clone of Brian O'Sullivan's [ENCE_MICE_DURR](https://github.com/BrianOSullivan-2000/ENCE_MICE_DURR),
