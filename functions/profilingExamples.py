# %load_ext line_profiler

# !lprun /Users/glennjocher/Documents/python/VideoExample.py

# pip install pprofile
# !pprofile /Users/glennjocher/Documents/python/VideoExample.py

# !pip install vprof
# !vprof -c h /Users/glennjocher/Documents/PyCharmProjects/Velocity/vidExample.py

import time
from functions.fcns import *

tic = time.time()
for i in range(0, 1000000):
    a = worldPointsLicensePlate()
print(time.time() - tic)
