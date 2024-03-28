import ckmeans
import numpy as np

samples = np.random.default_rng().uniform(1.0, 3.0, 110000)
for x in range(100):
    result = ckmeans.ckmeans(samples, 5)
