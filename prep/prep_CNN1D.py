from general_transform import Transform 
import numpy as np
import pandas as pd

def convert(filePath, features, labels, MaxParticles):
  '''
  features must have "j_index" and 'j_index" must located at the end of the list.
  Data shape: (number of total particles, MaxParticle, number of features/labels)
  '''
  df = Transform(filePath,)
  label_arr = df[labels].to_numpy()
  feature_arr = df[features].to_numpy()

  Data = {'features':[],'labels':[]}
  j_feat = np.zeros((MaxParticles, len(features)-1))
  j_lb = np.zeros((MaxParticles,len(labels)))
  countPar = 0
  for i in range(df.shape[0]):
    countPar += 1
    if countPar <= MaxParticles:
      j_feat[countPar-1] += feature_arr[i,:-1]
      j_lb[countPar-1] += label_arr[i]
    if i == df.shape[0]-1:
      Data['features'].append(j_feat)
      Data['labels'].append(j_lb)
      break
    if feature_arr[i,-1] != feature_arr[i+1,-1]:
      countPar =0
      Data['features'].append(j_feat)
      Data['labels'].append(j_lb)
      j_feat = np.zeros((MaxParticles, len(features)-1))
      j_lb = np.zeros((MaxParticles,len(labels)))
  X = np.reshape(Data['features'], (len(Data['features']),MaxParticles,len(features)-1))
  y = np.reshape(Data['labels'], (len(Data['labels']),MaxParticles,len(labels)))
  
  return X, y
  