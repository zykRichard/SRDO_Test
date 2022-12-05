import numpy as np
from sklearn.decomposition import PCA

data = np.random.random((1000, 120))
pca = PCA(n_components=0.9)

pca.fit(data)
newdata = pca.transform(data)

print(data)