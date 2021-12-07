import numpy as np
import plotly.graph_objects as go

from glob import glob

sample_dir = "/home/hjyoon/transferability/core/opportunity/results/"

feature_files = glob(sample_dir+"*-sensor.npy")
label_files = glob(sample_dir+"*-sensor_poslabel.npy")

orig_RShoe = []
orig_LLA = []
orig_LShoe = []
RShoe_from_LLA = []
RShoe_from_LShoe = []

for i in range(len(feature_files)):
    features_file = feature_files[i]
    labels_file = features_file.split('.')[0]+"_poslabel.npy"

    features = np.load(features_file)
    labels = np.load(labels_file)

    LLA_indices = []
    LShoe_indices = []
    RShoe_indices = []

    for k, l in enumerate(labels):
        if (l == [1, 0, 0]).all(): LLA_indices.append(k)
        if (l == [0, 1, 0]).all(): LShoe_indices.append(k)
        if (l == [0, 0, 1]).all(): RShoe_indices.append(k)
    
    for RShoe_index in RShoe_indices:
        orig_RShoe.append(features[RShoe_index][0].T)
    for LLA_index in LLA_indices:
        orig_LLA.append(features[LLA_index][3].T)
        RShoe_from_LLA.append(features[LLA_index][0].T)
    for LShoe_index in LShoe_indices:
        orig_LShoe.append(features[LShoe_index][0].T)
        RShoe_from_LShoe.append(features[LShoe_index][3].T)

orig_RShoe_features = []
orig_LLA_features = []
orig_LShoe_features = []
RShoe_from_LLA_features = []
RShoe_from_LShoe_features = []

for i in range(len(orig_RShoe)):
    f = np.mean(orig_RShoe[i], axis=1)
    # f = np.concatenate((f, np.std(orig_RShoe[i], axis=1)), axis=0)
    # f = np.concatenate((f, np.max(orig_RShoe[i], axis=1)), axis=0)
    # f = np.concatenate((f, np.min(orig_RShoe[i], axis=1)), axis=0)
    orig_RShoe_features.append(f)

    f = np.mean(orig_LLA[i], axis=1)
    # f = np.concatenate((f, np.std(orig_LLA[i], axis=1)), axis=0)
    # f = np.concatenate((f, np.max(orig_LLA[i], axis=1)), axis=0)
    # f = np.concatenate((f, np.min(orig_LLA[i], axis=1)), axis=0)
    orig_LLA_features.append(f)

    f = np.mean(orig_LShoe[i], axis=1)
    # f = np.concatenate((f, np.std(orig_LShoe[i], axis=1)), axis=0)
    # f = np.concatenate((f, np.max(orig_LShoe[i], axis=1)), axis=0)
    # f = np.concatenate((f, np.min(orig_LShoe[i], axis=1)), axis=0)
    orig_LShoe_features.append(f)

    f = np.mean(RShoe_from_LLA[i], axis=1)
    # f = np.concatenate((f, np.std(RShoe_from_LLA[i], axis=1)), axis=0)
    # f = np.concatenate((f, np.max(RShoe_from_LLA[i], axis=1)), axis=0)
    # f = np.concatenate((f, np.min(RShoe_from_LLA[i], axis=1)), axis=0)
    RShoe_from_LLA_features.append(f)

    f = np.mean(RShoe_from_LShoe[i], axis=1)
    # f = np.concatenate((f, np.std(RShoe_from_LShoe[i], axis=1)), axis=0)
    # f = np.concatenate((f, np.max(RShoe_from_LShoe[i], axis=1)), axis=0)
    # f = np.concatenate((f, np.min(RShoe_from_LShoe[i], axis=1)), axis=0)
    RShoe_from_LShoe_features.append(f)

print("feature means(mean/std/max/min of accXYZ/gyroXYZ):")
print("orig_Rshoe features:")
print(np.mean(np.array(orig_RShoe_features), axis=0))
print("\norig_LLA features:")
print(np.mean(np.array(orig_LLA_features), axis=0))
print("\norig_Lshoe features:")
print(np.mean(np.array(orig_LShoe_features), axis=0))
print("\nRShoe_from_LLA features:")
print(np.mean(np.array(RShoe_from_LLA_features), axis=0))
print("\nRShoe_from_LShoe features:")
print(np.mean(np.array(RShoe_from_LShoe_features), axis=0))

from sklearn.manifold import TSNE
import matplotlib.pyplot as pp

num_points=100
all_features = orig_RShoe_features[:num_points]
all_features += orig_LLA_features[:num_points]
all_features += orig_LShoe_features[:num_points]
all_features += RShoe_from_LLA_features[:num_points]
all_features += RShoe_from_LShoe_features[:num_points]
all_features = np.array(all_features)

all_colors = [0 for i in range(len(orig_RShoe_features))][:num_points]
all_colors += [1 for i in range(len(orig_LLA_features))][:num_points]
all_colors += [2 for i in range(len(orig_LShoe_features))][:num_points]
all_colors += [3 for i in range(len(RShoe_from_LLA_features))][:num_points]
all_colors += [4 for i in range(len(RShoe_from_LShoe_features))][:num_points]

all_labels = [['orig_RShoe_features'] for i in range(len(orig_RShoe_features))][:num_points]
all_labels += [['orig_LLA_features'] for i in range(len(orig_LLA_features))][:num_points]
all_labels += [['orig_LShoe_features'] for i in range(len(orig_LShoe_features))][:num_points]
all_labels += [['RShoe_from_LLA_features'] for i in range(len(RShoe_from_LLA_features))][:num_points]
all_labels += [['RShoe_from_LShoe_features'] for i in range(len(RShoe_from_LShoe_features))][:num_points]

tsne = TSNE(n_components = 2, init = "pca", random_state = 0).fit_transform(all_features)

figure, axesSubplot = pp.subplots()
axesSubplot.scatter(tsne[:, 0], tsne[:, 1], c = all_colors, label = all_colors)
axesSubplot.legend()
axesSubplot.set_xticks(())
axesSubplot.set_yticks(())
pp.show()

# assert(0)

# LLA_index = LLA_indices[seed]
# LShoe_index = LShoe_indices[seed]
# RShoe_index = RShoe_indices[seed]

# print(np.mean(features[RShoe_index][0].T[4]))
# print(np.mean(features[LLA_index][2].T[4]))
# print(np.mean(features[LShoe_index][2].T[4]))

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=np.arange(60), y=features[RShoe_index][0].T[0], name="accX_org",
#                     line_shape='linear'))
# fig.add_trace(go.Scatter(x=np.arange(60), y=features[RShoe_index][0].T[1], name="accY_org",
#                     line_shape='linear'))
# fig.add_trace(go.Scatter(x=np.arange(60), y=features[RShoe_index][0].T[2], name="accZ_org",
#                     line_shape='linear'))
            
# fig.add_trace(go.Scatter(x=np.arange(60), y=features[LLA_index][3].T[0], name="accX_genfromLLA",
#                     line_shape='linear'))
# fig.add_trace(go.Scatter(x=np.arange(60), y=features[LLA_index][3].T[1], name="accY_genfromLLA",
#                     line_shape='linear'))
# fig.add_trace(go.Scatter(x=np.arange(60), y=features[LLA_index][3].T[2], name="accZ_genfromLLA",
#                     line_shape='linear'))

# fig.add_trace(go.Scatter(x=np.arange(60), y=features[LShoe_index][3].T[0], name="accX_genfromLShoe",
#                     line_shape='linear'))
# fig.add_trace(go.Scatter(x=np.arange(60), y=features[LShoe_index][3].T[1], name="accY_genfromLShoe",
#                     line_shape='linear'))
# fig.add_trace(go.Scatter(x=np.arange(60), y=features[LShoe_index][3].T[2], name="accZ_genfromLShoe",
#                     line_shape='linear'))

# fig.update_traces(hoverinfo='text+name', mode='lines+markers')
# fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))

# fig.show()