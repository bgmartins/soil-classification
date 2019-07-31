
import sys
import getopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer, InterclusterDistance, KElbowVisualizer


# DEFAULTS
inputfile = '../data/test/mexico_k_1_layers_5.csv'
h = '04_kmeans.py -h <help> -i <input file> '

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:")
except getopt.GetoptError:
    print(h)
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print(h)
        sys.exit()
    elif opt in ('-i'):
        inputfile = arg


data = pd.read_csv(inputfile)
data.dropna(inplace=True)


X = data.drop(['profile_id'], axis=1)

score = []
max_clusters = 50

for cluster in range(1, max_clusters):
    print(f'Testing k = {cluster}')
    kmeans = KMeans(n_clusters=cluster, init="k-means++")
    kmeans.fit(X)
    score.append(kmeans.inertia_)


plt.plot(range(1, max_clusters), score)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()


# ELBOW
visualizer = KElbowVisualizer(KMeans(), k=(1, 50))
visualizer.fit(X)    # Fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


model = KMeans(n_clusters=10, init="k-means++")

# SILHOUETTE
visualizer = SilhouetteVisualizer(model)
visualizer.fit(X)  # Fit the training data to the visualizer
visualizer.poof()  # Draw/show/poof the data

# DISTANCE
visualizer = InterclusterDistance(model)
visualizer.fit(X)  # Fit the training data to the visualizer
visualizer.poof()
