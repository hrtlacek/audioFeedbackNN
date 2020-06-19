import numpy as np
from numba import jit


from tqdm import tqdm
import librosa
import matplotlib.pyplot as plt


# ===CONFIG VARS==
conf = {
    "nFFT":1024,
    "hopLength":512,
    "seqLen":20,
    "usePolar":False,
    "sr":44100,
    "useLog":True,
    "normLow":-200,
    "normHigh":70,
    "ampOnly":False
}
#================


def printConfig():
    print(conf)

@jit(forceobj=True)
def cart2pol(x, y, setZeroPhase=True, thresh=1e-5):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)/(2*np.pi)
    if conf['useLog']:
        rhoc = np.clip(20*np.log10(rho), conf['normLow'], conf["normHigh"])
        normRange = conf['normHigh']-conf["normLow"]
        rhoc-=conf['normLow']
        rhoc/=normRange 
        rhof = rhoc
        mask = rhof<=0.6
    else:
        rhof = rho
        mask = rhof<=thresh
    if setZeroPhase:
        phi[mask] = -0.5

    phiNorm = phi+0.5
    return(rhof, phiNorm)

@jit(forceobj=True)
def pol2cart(rho, phiNorm):
    phi = phiNorm-0.5
    if conf['useLog']:
        normRange = conf['normHigh']-conf["normLow"]
        rhoc=rho*normRange
        rhoc+=conf["normLow"]
        rhoc = 10**(rhoc/20)
    else:
        rhoc = rho

    x = rhoc * np.cos(phi*2*np.pi)
    y = rhoc * np.sin(phi*2*np.pi)
    return(x, y)

@jit(forceobj=True)
def fromSplit(r,i):
    if conf['usePolar']:
        r,i = pol2cart(r,i)
    comp = np.zeros([r.shape[0], r.shape[1]], dtype=np.complex64)
    comp.real = r
    comp.imag = i
    return comp

def getFFT(x, doPlot=False): 
    ft = librosa.stft(x, n_fft=conf["nFFT"],hop_length=conf["hopLength"])
    if doPlot:
        stftPlot(ft)
    return ft

def getIFFT(stft, doPlot=True):
    


    x = librosa.istft(stft,hop_length=conf["hopLength"])
    if doPlot:
        N =  len(x)
        n = np.arange(N)
        t = n/conf['sr']
        plt.plot(t, x)
        plt.grid()
        plt.xlabel("Time [sec]")
    return x

@jit(forceobj=True)
def stftToXy(stft):
    seqLen = conf["seqLen"]
    frameSize, numFrames = stft.shape
    if numFrames-seqLen<=0:
        raise ValueError("Sequence Length Bigger than the Number of Frames Received!")
    ftCorr = stft.T



    xData = np.zeros([numFrames-seqLen,seqLen, frameSize], dtype=stft.dtype)
    yData = np.zeros([numFrames-seqLen,frameSize], dtype=stft.dtype)

    for i in range(numFrames-seqLen):
        xData[i,:,:] = ftCorr[i:i+seqLen,:]
        yData[i,:] = ftCorr[i+seqLen,:]
    
    return xData,yData


def audioToDataSet(x, doPlot=True):
    stft = getFFT(x, doPlot=doPlot)
    xData,yData = stftToXy(stft)
    xData,yData = complexSplit(xData,yData)
    return xData,yData

@jit(forceobj=True)
def complexSplit(cX, cY):
    """
    #-----samples|time|real+imag-----------------
    """
    usePolar=conf["usePolar"]

    frameSize = cX.shape[2]
    xr = cX.real
    xi = cX.imag

    yr = cY.real
    yi = cY.imag

    finX = np.zeros([xr.shape[0],xr.shape[1],xr.shape[2]*2])
    finY = np.zeros([yr.shape[0],yr.shape[1]*2])

    if usePolar:
        poxRho,poxPhi = cart2pol(xr, xi)
        poyRho,poyPhi = cart2pol(yr, yi)

        finX[:,:,0:frameSize] = poxRho
        finX[:,:,frameSize:] = poxPhi

        finY[:,0:frameSize] = poyRho
        finY[:,frameSize:] = poyPhi
    else:
        finX[:,:,0:frameSize] = xr
        finX[:,:,frameSize:] = xi

        finY[:,0:frameSize] = yr
        finY[:,frameSize:] = yi

    return finX, finY

# @jit(forceobj=True)
def feedbackPredict(model, initialCond, addSig, modSig, numPreds=200,fbAtten=1, doPlot = False): 
    usePolar=conf["usePolar"]
    ampOnly = conf["ampOnly"]

    # n = randint(0,finX.shape[1])
    nFeatures = model.output_shape[1]
    frameSize = nFeatures//2

    rec = np.zeros([numPreds, nFeatures])

    ringBuf = initialCond#finX[newaxis, n,:]
    fig = None
    axs = None
    if doPlot:
        fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(20, 12),
                            subplot_kw={})


    amp = np.zeros(numPreds)

    j = 0
    for i in range(numPreds):

        pred = model.predict(ringBuf)
        pred = np.nan_to_num(pred)
        pred[pred == -np.inf] = -1
        pred[pred == np.inf] = 1
        ringBuf = np.roll(ringBuf,-1, axis=1)
        ringBuf[0,-1,:] = pred*fbAtten*modSig[i,:] + addSig[i,:]#(1+(random.random(nFeatures)-0.5)*rMod)+(random.random(nFeatures)-0.5)*rAdd#+(random.random(nFeatures))*amp[i]*20
        rec[i,:] = pred
        if doPlot:
            if i%50==0:
                try:
                    cRing = fromSplit(ringBuf[0,:,0:frameSize], ringBuf[0,:,frameSize:])
                    
                    axs.flat[j].imshow(20*np.log10(abs(cRing.T)), aspect='auto', origin='lower', cmap='jet', interpolation='bilinear')
                    j+=1
                except:
                    pass
    if doPlot:
        plt.show()
    return rec

def stftPlot(ft):
    # fAxis = np.linspace(0, conf["sr"]/2, ft.shape[0])
    plt.figure (figsize=[20,10])
    p = plt.imshow(20*np.log10(abs(ft)), aspect='auto', origin='lower', cmap = 'jet', interpolation='bilinear')
    plt.colorbar()
    plt.ylabel('bin')
    plt.xlabel('frame')
    plt.show()
    return p


def plotOverTime(m, finX):
    nFeatures = finX.shape[2]
    fig, axs = plt.subplots(2, 2, figsize=[15,15])
    for k in range(4):
        fc = round(nFeatures*k/4)
        numPreds = 50
        plotLength = 4
        # timeLine = np.arange(plotLength+numPreds)
        # for i in range(1):
        ax = axs[k//2, k%2]
    #     n = randint(0,finX.shape[1]-1) #choose random start point in time
        n = 0
        for j in range(numPreds):
            xin = finX[np.newaxis,n+j,:,:] #get x data with offset for each pred
            pred = m.predict(xin)
            xPlotData = xin[0,-plotLength:,fc]+j*0.00
            ax.plot(np.arange(plotLength)+j, xPlotData)
            ax.scatter([plotLength+j], pred[:,fc], c='r', marker='x', alpha=0.5)

        ax.grid()
    plt.show()
    return fig
