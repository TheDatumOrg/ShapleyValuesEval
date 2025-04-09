import numpy as np
import shap.utils as U

def similarity_ces(explicand, data, categorical, ratio=0.01, scaled=False):
    num_features = data.shape[1]
    matches = np.ones(data.shape)
    for i in range(num_features):
        if not scaled:
            if categorical[i] == 0:
                xmax = np.max(data[:, i])
                xmin = np.min(data[:, i])
                xdist = (xmax-xmin) * ratio
                cmax = explicand[i] + xdist
                cmin = explicand[i] - xdist
                for j in range(data.shape[0]):
                    if data[j, i] >= cmax and data[j, i] <= cmin:
                        matches[j, i] = 0    
            else:
                val = explicand[i]
                for j in range(data.shape[0]):
                    if data[j, i] != val:
                        matches[j, i] = 0
        else:
            xmax = np.max(data[:, i])
            xmin = np.min(data[:, i])
            xdist = (xmax-xmin) * ratio
            cmax = explicand[i] + xdist
            cmin = explicand[i] - xdist
            for j in range(data.shape[0]):
                if data[j, i] >= cmax and data[j, i] <= cmin:
                    matches[j, i] = 0    
            
    
    cond = np.count_nonzero(matches, axis=1)

    count = 0
    baseline = []
    for idx in range(len(cond)):
        if cond[idx] == num_features:
            count = count + 1 
            baseline.append(data[idx, :])
    
    if count == 0:
        return U.sample(data, 100)
    else:
        return U.sample(np.array(baseline), 100)


def similarity_cohort(explicand, data, vertex, categorical, ratio=0.01, scaled=False):
    num_features = data.shape[1]
    matches = np.ones(data.shape)
    for i in range(num_features):
        if vertex[i] == 0:
            continue
        else:
            if not scaled:
                if categorical[i] == 0:
                    xmax = np.max(data[:, i])
                    xmin = np.min(data[:, i])
                    xdist = (xmax-xmin) * ratio
                    cmax = explicand[i] + xdist
                    cmin = explicand[i] - xdist
                    for j in range(data.shape[0]):
                        if data[j, i] >= cmax and data[j, i] <= cmin:
                            matches[j, i] = 0    
                else:
                    val = explicand[i]
                    for j in range(data.shape[0]):
                        if data[j, i] != val:
                            matches[j, i] = 0
            else:
                xmax = np.max(data[:, i])
                xmin = np.min(data[:, i])
                xdist = (xmax-xmin) * ratio
                cmax = explicand[i] + xdist
                cmin = explicand[i] - xdist
                for j in range(data.shape[0]):
                    if data[j, i] >= cmax and data[j, i] <= cmin:
                        matches[j, i] = 0    
            
    
    cond = np.count_nonzero(matches, axis=1)
    ccond = np.zeros(cond.shape)
    for idx in range(len(cond)):
        if cond[idx] == num_features:
            ccond[idx] = 1

    return ccond
