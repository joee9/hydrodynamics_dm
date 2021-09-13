## Hydrodynamical neutron star simulation

This project is the simulation of a neutron star, containing bosonic dark matter, using a model of the neutron star as a fluid. The bosonic dark matter is modelled as a complex scalar field.

### Code
Broken into "auxillary" files (like `C` headers) and the main simulation file. Global parameters and variables are declared in `hd_params.py`.

### Flags/Options
All flags and parameters are consolidated to the `hd_params.py` file in root.

#### Recording
Data recording can be toggled with the `record_data` flag. Currently, this record data flag is configured to write data to a file for every parameter. The `output_number` flag will add any integer to the start of the file names for an individual run to allow for multiple runs to be stored and distinguished from one another. The `record_ringdown` flag creates an output file that stores all of the field values at one spatial point at more temporal points. Currently, values are stored at r = 0 (or, for staggered fields, r = 1/2 \* dr.


### I/O
Make sure that the folders `input` and `data` are created such that the program can write correctly. Otherwise, and error will occur.
*To be included in the future*: the static solutions will automatically be generated to the correct directory to be read by the main simulation code. Depending on the chosen initial conditions, the correct static solutions will be chosen.
