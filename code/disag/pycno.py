import numpy as np
from scipy import ndimage
from disag import massp
from utils import osgeoutils as osgu, nputils as npu


def runPycno(idsdataset, polygonvaluesdataset, rastergeo, niter=100, converge=0.01, tempfileid=None):
    print('| PYCNOPHYLACTIC INTERPOLATION')
    pycnodataset = massp.runMassPreserving(idsdataset, polygonvaluesdataset, rastergeo, tempfileid)[0]
    oldpycnodataset = pycnodataset

    idpolvalues = npu.polygonValuesByID(polygonvaluesdataset, idsdataset)
    pycnomask = np.copy(idsdataset)
    pycnomask[~np.isnan(pycnomask)] = 1

    for it in range(1, niter+1):
        print('| - Iteration', it)

        # Calculate the mean of the cells in the 3 by 3 neighborhood
        mask = np.array([[0,1,0],[1,0,1],[0,1,0]])
        mask = np.expand_dims(mask, axis=2)
        pycnodataset = ndimage.generic_filter(pycnodataset, np.nanmean, footprint=mask, mode='constant', cval=np.NaN)

        # Summarizes the values within each polygon
        stats = npu.statsByID(pycnodataset, idsdataset, 'sum')

        # Divide the true polygon values by the estimated polygon values (= ratio)
        polygonratios = {k: idpolvalues[k] / stats[k] for k in stats.keys() & idpolvalues}

        # Multiply ratio by the different cells within each polygon
        for polid in polygonratios:
            pycnodataset[idsdataset == polid] = (pycnodataset[idsdataset == polid] * polygonratios[polid])

        pycnodataset = pycnodataset * pycnomask

        # Check if the algorithm has converged
        error = np.nanmean(abs(pycnodataset - oldpycnodataset))
        rangeds = np.nanmax(oldpycnodataset) - np.nanmin(oldpycnodataset)
        stopcrit = converge # * rangeds
        print('Error:', error)

        if ((it > 1) and (error < stopcrit)):
            break
        else:
            oldpycnodataset = pycnodataset


    if tempfileid:
        tempfile = 'tempfilepycno_' + tempfileid + '.tif'
        osgu.writeRaster(pycnodataset[:, :, 0], rastergeo, tempfile)

    return pycnodataset, rastergeo
