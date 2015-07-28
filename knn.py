import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

pca = PCA(n_components=0.8)
neigh = KNeighborsClassifier(n_neighbors=5)
train_data = pd.read_csv("data/train.csv")
data = train_data.values[:,1:]
label = train_data.values[:,0]
data = pca.fit_transform(data)
neigh.fit(data,label)
print("step1")
test = pd.read_csv("data/test.csv")
test_data = pca.transform(test.values)
print(test_data.shape)
res = neigh.predict(test_data)
df = pd.DataFrame({'ImageId':range(1,28001),'Label':res})
df.to_csv("knn_res.csv",index=False)