import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyjet import cluster

def _load (filePath, nJets=200000, nConstituents=40):
    '''
    Returns:
        momenta: (nJets, 4, nConstituents)
    '''
    cols = ['E_'+str(i) for i in range(nConstituents)]+ ['PX_'+str(i) for i in range(nConstituents)] + ['PX_'+str(i) for i in range(nConstituents)] + ['PY_'+str(i) for i in range(nConstituents)] + ['PZ_'+str(i) for i in range(nConstituents)] + ['is_signal_new']
    df = pd.read_hdf(filePath,key='table',stop=nJets, columns = cols)
    # Take all the 4 momentum from 200 particles in all jets and reshape them into one particle per row
    momenta = df.iloc[:,:-1].to_numpy()
    momenta = momenta.reshape(-1,nConstituents,4)
    nJets = slice(nJets)
    momenta = momenta[nJets, :nConstituents, :]
    momenta = np.transpose(momenta, (0, 2, 1))
    label = df['is_signal_new']
    return momenta, label

def _deltaPhi(phi1,phi2):
    x = phi1-phi2
    while x>= np.pi: x -= np.pi*2.
    while x< -np.pi: x += np.pi*2.
    return x

def _rotate2D(x, y, a):
    xp = x * np.cos(a) - y * np.sin(a)
    yp = x * np.sin(a) + y * np.cos(a)
    return xp, yp

def _rotate(event, R0 = 0.2,  p = 1):
    '''
    input:
        event: (nConstituents,4)
        R0 = Clustering radius for the main jets
        p = -1, 0, 1 => anti-kt, C/A, kt Algorithm
    '''
    event = np.transpose(event,(1,0))
    eventCopy = np.core.records.fromarrays( [event[:,0],event[:,1],event[:,2],event[:,3]], names= 'E, PX, PY, PZ' , formats = 'f8, f8, f8,f8')
    sequence = cluster(eventCopy, R=R0, p= p, ep=True)
    # List of jets
    jets = sequence.inclusive_jets()
    if len(jets)<2:
        return []
    else:
        subjet_data = event
        subjet_array = jets
        
        p = np.linalg.norm(event[:, 1:], axis=1)
        eta = 0.5 * np.log((p + event[:, 3]) / (p - event[:, 3]))
        phi = np.arctan2(event[:, 2], event[:, 1])
                
        # Shift all data such that the leading subjet
        #jet new center is located at (eta,phi) = (0,0)
        eta -= subjet_array[0].eta
        phi = np.array( [_deltaPhi(i,subjet_array[0].phi) for i in phi])
        
        # Rotate the jet image such that the second leading
        # jet is located at -pi/2
        s1x, s1y = subjet_array[1].eta - subjet_array[0].eta, _deltaPhi(subjet_array[1].phi,subjet_array[0].phi)
        
        theta = np.arctan2(s1y, s1x)
        if theta < 0.0:
            theta += 2 * np.pi
        etaRot, phiRot = _rotate2D(eta, phi, np.pi - theta)
        
        # Collect the trimmed subjet constituents
        return etaRot, phiRot
  
def getRot(momenta,nJets, nConstituents):
    '''
    momenta: (nJets, 4, nConstituents)
    '''
    etaRot = []
    phiRot = []
    for i in tqdm(range(nJets)):
        event = momenta[i,:,:]
        etaR,phiR = _rotate(event)
        etaRot.append(etaR)
        phiRot.append(phiR)
    return etaRot, phiRot

def _transform(momenta,nJets,nConstituents,model='DGCNN'):
    '''
    Input:
     momenta: (nJets, 4, nConstituents)
    Returns:
     features: (nJets, nConstituents, number of features)
    '''
    # Jet features
    jetMomenta = np.sum(momenta, axis=2)
    jetPt = np.linalg.norm(jetMomenta[:, 1:3], axis=1)[..., np.newaxis]
    jetE = jetMomenta[:, 0][..., np.newaxis]
    jetP = np.linalg.norm(jetMomenta[:, 1:], axis=1)
    jetEta = 0.5 * np.log((jetP + jetMomenta[:, 3]) / (jetP - jetMomenta[:, 3]))[..., np.newaxis]
    jetPhi = np.arctan2(jetMomenta[:, 2], jetMomenta[:, 1])[..., np.newaxis]
    jetTheta = 2*np.arctan(np.exp(-jetEta))
    
    # Constituent features
    # delta eta, delta phi, log pT, log E,log pT / pTjet, log E / Ejet, delta R
    pT = np.linalg.norm(momenta[:, 1:3, :], axis=1)
    e = momenta[:, 0, :]
    p = np.linalg.norm(momenta[:, 1:, :], axis=1)
    eta = 0.5 * np.log((p + momenta[:, 3, :]) / (p - momenta[:, 3, :]))
    etaRel = eta - jetEta
    phi = np.arctan2(momenta[:, 2, :], momenta[:, 1, :])
    phiRel = np.unwrap(phi - jetPhi)
    dR = np.sqrt(phi ** 2 + eta ** 2)
    theta = 2*np.arctan(np.exp(-eta))
    cosThetaRel = np.cos(theta-jetTheta)

    etaRot,phiRot = getRot(momenta,nJets,nConstituents)

    # Set calculated features of non-particle entries to zero
    eta[pT==0] = 0
    phi[pT==0] = 0
    dR[pT==0] = 0
    
    if model == "DGCNN":
        # Stack a new feature vector
        newVec = np.stack([eta, phi, np.log(pT), np.log(e),np.log(pT / jetPt), np.log(e / jetE), dR], axis=-1)
        newVec[newVec==-np.inf] = 0  # Deal with infinities
    else:
        newVec =np.stack([etaRot,phiRot,etaRel,phiRel,cosThetaRel,np.log(pT / jetPt), e, np.log(e / jetE), dR ],axis=-1)
        newVec[newVec==-np.inf] = 0  # Deal with infinities
    return newVec

def _save(features,labels,nConstituents, filePath):
    with h5py.File(filePath, 'w') as f:
        f.create_dataset('features',data=features)
        f.create_dataset('labels',data = np.repeat(labels,nConstituents))

def Transform(filePath,nJets=20000,nConstituents=40,model='DGCNN', saveFile=False):
    momenta,labels = _load(filePath,nJets , nConstituents)
    features = _transform(momenta,nJets,nConstituents,model)
    if saveFile:
        _save(features,labels,nConstituents, filePath)
    else:
        return features, labels