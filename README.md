# Final Project (Group 6) - Notes

## Data format

The data files are organized as a series of events (rows) where each event consist of the coordinates of the hits recorded in that event (N.B.: as the number of hits for even is not fixed, the dataset cannot be seen as a unique big table with a defined number of "columns"). More specifically every raw consists of:

- event number
- number of recorded hits (N)
- hit 1 coordinates
- ...
- hit N coordinates

where N could be zero too and the hits coordinates are:

- *chamber*, which of the four detector registered the hit (numbering is in the range $0-4$)
- *layer*, which layer of that detector
- *xleft*, the x position of the hit **in the local detector coordinates** assuming the track passed at the left of the wire
- *xright*, the x position of the hit **in the local detector coordinates** assuming the track passed at the right of the wire
- *time*, the drift time (redundant)

The local detector coordinates are defined with respect to one side (the left one) of the detector. All the detectors however were placed rotated by 180 degrees, i.e. the x axes of the local and global coordinates have different orientation.

### Notes on data format

- for each recordered hit we have 5 coordinates; this means that the length of a row is **2 + 5N**
- assuming each row is a python list (i.e. `list[0] = event number`, `list[1] = N`, ...) and $i \in [0, 1, ..., N-1]$:
  - *chamber*: `list[2+5i]`
  - *layer*: `list[3+5i]`
  - *xleft*: `list[4+5i]`
  - *xright*: `list[5+5i]`
  - *time*: `list[6+5i]`
  - the possible layout for a pandas dataframe could be:

|Ev_Number | $n \in [1, N]$ | chamber | layer | XL_local | XR_local | Z_local | time | XL_global | XR_global | Z_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Coordinates transformation

The relevant dimensions and the overall coordinate transformations in $z$ and $x$ are given below. Note that all the quantities are expressed in $mm$.

```python
# Cell dimensions
XCELL = 42.
ZCELL = 13.

# X coordinates translation
global_x_shifts = [994.2, 947.4,-267.4,-261.5,]

# Z coordinates translations
local_z_shifts = [z*ZCELL for z in range(0,4)]
global_z_shifts = [823.5, 0, 823.5, 0]
```

### Notes on the transformation

```python
# nedd to understand the 180 degrees rotation of detectors
x_global = x_local + global_x_shifts[chamber]

# need to understand the z coordinate (-> ground of the layer?)
z_local = local_z_shifts[layer]
z_global = z_local + global_z_shifts[chamber]
```

## Functions to write

### Read data

- **Input**: 1 row of data_file.txt, e.g. 1 event
- **Output**: pandas dataframe as in the *Data Format* paragraph
- **TO DO**:
  - transform local coordinates to glob (add column to dataframe)

### Plot events

- **Input**: pandas dataframe (1 event)
- **Output**: pyplot Figure (global image + 4 detectors zooms)
- **TO DO**:

### Select Events (Calibration)

- **Input**: pandas dataframe (1 event)
- **Output**: True/False
- **TO DO**:

### Local linear fit