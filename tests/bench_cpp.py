import numpy as np
from ckmeans_1d_dp import ckmeans

samples = np.random.default_rng().uniform(1.0, 3.0, 110000)
for x in range(100):
    result = ckmeans(samples, 5)
