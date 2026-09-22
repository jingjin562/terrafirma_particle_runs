#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 12:49:07 2025

@author: jingjin
"""

from glob import glob
from os import path
from os import environ
import xarray as xr

import dask
import xarray as xr
from dask.diagnostics import ProgressBar
from numpy import *
from pylab import *
import zarr

in_dir = environ['InDir']
fname = environ['Filename']

# specify chunksize and where the output zarr file should go; also set chunksize of output file
chunksize = {"trajectory": 5 * int(1e3), "obs": 10}
outputDir = f'{in_dir}/Combined_{fname}'

if path.exists(outputDir):
    print("the ouput path", outputDir, "exists")
    print("please delete if you want to replace it")
    assert False, "stopping execution"

varType = {
    "lat": dtype("float32"),
    "lon": dtype("float32"),
    "time": dtype("float64"),  # to avoid bug in xarray
    "z": dtype("float32"),
}

files = glob(path.join(f'{in_dir}/{fname}', "proc*"))
dataIn = xr.concat(
    [xr.open_zarr(f, decode_times=False) for f in files],
    dim="trajectory",
    compat="no_conflicts",
    coords="minimal",
).sortby(["trajectory"])

for v in varType.keys():
    dataIn[v] = dataIn[v].astype(varType[v])

#compressor = numcodecs.blosc.Blosc(cname='zstd', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
compressor = zarr.codecs.BloscCodec(cname="zstd")

for v in dataIn.variables:
    dataIn[v].encoding.update({"compressors": compressor})
    if "chunks" in dataIn[v].encoding:
        del dataIn[v].encoding["chunks"]
     
dataIn = dataIn.chunk(chunksize)
   
delayedObj = dataIn.to_zarr(outputDir, mode='w')  

# dataProcessed = xr.open_zarr(outputDir)
