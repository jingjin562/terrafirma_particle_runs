from parcels import AdvectionRK4_3D
from parcels import FieldSet
from parcels import Field
from parcels import JITParticle
from parcels import ScipyParticle
from parcels import ParticleSet
from parcels import Variable
from parcels import plotTrajectoriesFile
from parcels import ErrorCode
from parcels import ParcelsRandom

import numpy as np
import math
import xarray as xr
from datetime import timedelta as delta
from operator import attrgetter
import datetime

class TSParticle(JITParticle):
     prev_lon = Variable('prev_lon', dtype=np.float32, to_write=False, initial=attrgetter('lon'))
     prev_lat = Variable('prev_lat', dtype=np.float32, to_write=False, initial=attrgetter('lat'))
     prev_depth = Variable('prev_depth', dtype=np.float32, to_write=False, initial=attrgetter('depth'))
     prev_time = Variable('prev_time', dtype=np.float32, to_write=False, initial=attrgetter('time'))
     rec = Variable('rec', dtype=int, to_write=False, initial=0)
     temp = Variable('temp', initial=0, dtype=np.float32)
     sal = Variable('sal', initial=0, dtype=np.float32)
     distance = Variable('distance', initial=0., dtype=np.float32)
     dep = Variable('dep', initial=0, dtype=np.float32)
     ice = Variable('ice', initial=0, dtype=np.float32)
      


def RecoveryParticle(particle, fieldset, time):
    if particle.state == StatusCode.ErrorOutOfBounds:
       particle_dlat += particle.prev_lat - particle.lat + 0.05*ParcelsRandom.uniform(-1., 1.)
       particle_dlon += particle.prev_lon -particle.lon + 0.1*ParcelsRandom.uniform(-1., 1.)
       particle_ddepth += particle.prev_depth - particle.depth + 1.*ParcelsRandom.uniform(-1., 1.)
       if particle.dep == 0 :
         particle.rec += 1
         particle.state = StatusCode.Success
       else :
         particle.rec = 0
         particle.state = StatusCode.Success
       if particle.rec == 20 :
         particle.delete()
