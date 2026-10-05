# Changelog
## v0.6.1
 - Tiny addition: Detect whether single panel data or single patern data for multipanel images are given and suppress the extra output dimension.
   e.g. (single panel data) Input of shape (Nx,Ny) -> (N_samples,) instead of previousely (1,N_samples)
        (multipanel data)  Input of shape (Npanels,Nx,Ny) -> (N_samples,) instead of (1,N_samples)
## v0.6.2
 - Add policy.Masking.Renormalization option. Under this masking condition output samples are evaluated from its partially masked input pixels by weight renormalization as long as the relative unmasked area fraction is above a given threshold
