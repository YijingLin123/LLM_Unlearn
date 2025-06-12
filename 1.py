import numpy as np


Original = list(map(float, ['0.2540', '0.3604', '0.4440', '0.5147', '0.5749', '0.6260']))

Reproduce = list(map(float, ['0.5363', '0.6145', '0.6712', '0.7287', '0.7892', '0.8541']))

Malicious = list(map(float, ['0.5408', '0.6951', '1.0127', '0.9469', '1.2487', '1.3455']))

r = np.corrcoef(Original, Reproduce)[0, 1]
r1 = np.corrcoef(Original, Malicious)[0, 1]

print(r)
print(r1)


