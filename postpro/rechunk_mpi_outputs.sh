#!/bin/bash

Filename='Ross_cavity_du195_5d_2201-2211_diffu_0_dt1days.zarr'
InDir='/home/jingjin/work/ocean_parcels/diffusion_exp/'

export Filename
export InDir

python rechunk_mpi_outputs.py 

OutFile=${InDir}Combined_${Filename}
if [ -d ${OutFile} ]; then
   rm -rf ${InDir}${Filename}
   mv ${OutFile} ${InDir}${Filename}
   echo "Combined output is created."
else
   echo "${OutFile} does not exit."
fi
