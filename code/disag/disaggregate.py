from utils import osgeoutils as osgu, nputils as npu, gputils as gput, metricsev as mev, neigPairs
from models import kerasutils as ku, caret
from disag import pycno
import numpy as np
from sklearn import metrics
import os


def runDisaggregation(fshape, ancdatasets, yraster=None, rastergeo=None, perc2evaluate = 0.1, poly2agg = 'NUTSIII',
                method='lm', cnnmod='unet', patchsize=16, epochspi=3, batchsize=64, lrate=0.001, filters=[8, 16, 32, 64, 128],
                lweights=[0.5, 0.5], extdataset=None, p=[1], min_iter=30, max_iter=30, converge=2,
                hubervalue=1, stdivalue=40, dropout=0.5, casestudy='pcounts',
                resetweights=True, previters=False, lossparams=[2,0,0,False],
                tempfileid=None, verbose=False):

    print('\n| DISAGGREGATION\n')
    indicator = casestudy.split('_')[0]
    filenamemetrics2e = 'pcounts_' + casestudy + '_2e.csv'

    if patchsize >= 16 and (cnnmod == 'lenet' or cnnmod == 'uenc' or cnnmod == 'vgg'):
        cstudyad = 'tempdata_' + indicator + '_' + str(patchsize) + '_wpadd_extended'
    elif patchsize >= 16:
        cstudyad = 'tempdata_' + indicator + '_' + str(patchsize) + '_nopadd_extended'
    else:
        cstudyad = None

    nrowsds = ancdatasets[:,:,0].shape[1]
    ncolsds = ancdatasets[:,:,0].shape[0]
    idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0] # ID's polígonos originais

    print('| Computing polygons areas')
    polygonareas = gput.computeAreas(fshape)

    if yraster:
        disaggdataset, rastergeo = osgu.readRaster(yraster)
        idpolvalues = npu.statsByID(disaggdataset, idsdataset, 'sum')
    else:
        polygonvaluesdataset, rastergeo = osgu.ogr2raster(fshape, attr='VALUE', template=[rastergeo, nrowsds, ncolsds])
        idpolvalues = npu.polygonValuesByID(polygonvaluesdataset, idsdataset)
        disaggdataset, rastergeo = pycno.runPycno(idsdataset, polygonvaluesdataset, rastergeo, tempfileid)

    iterdissolv = False
    if any(meth.startswith('ap') for meth in method):
        if iterdissolv:
            adjpolygons = gput.computeNeighbors(fshape, polyg = poly2agg, verbose=True)
        else:
            adjpairs, newpairs = neigPairs.createNeigSF(fshape, polyg=poly2agg)

    dissmask = np.copy(idsdataset)
    dissmask[~np.isnan(dissmask)] = 1
    ancvarsmask = np.dstack([dissmask] * ancdatasets.shape[2])

    olddisaggdataset = disaggdataset


    if any(meth.endswith('cnn') for meth in method):
        padd = True if cnnmod == 'lenet' or cnnmod == 'uenc' or cnnmod == 'vgg' else False

        # Create anc variables patches (includes replacing nans by 0, and 0 by nans)
        print('| Creating ancillary variables patches')
        # ancdatasets[np.isnan(ancdatasets)] = 0
        ancpatches = ku.createpatches(ancdatasets, patchsize, padding=padd, stride=1, cstudy=cstudyad)
        ancdatasets = ancdatasets * ancvarsmask

        if previters:  # Include previous iterations
            previterspath = os.path.join('Results', indicator, 'Baselines', 'massp_200m.tif')
            previtersmap = osgu.readRaster(previterspath)[0]
            previtersarr = ku.createpatches(previtersmap, patchsize, padding=padd, stride=1)

        # Compile model and save initial weights
        cnnobj = ku.compilecnnmodel(cnnmod, [patchsize, patchsize, ancdatasets.shape[2]], lrate, dropout,
                                    filters=filters, lweights=lweights, hubervalue=hubervalue, stdivalue=stdivalue,
                                    previters=previters, lossparams=lossparams)

        ku.savemodel(cnnobj, 'Temp/models_' + casestudy, includeoptimizer=True)

        # If two UNets ('1cnn' and '2cnn'), compile and save 2nd UNet
        if any(meth.endswith('2cnn') for meth in method):
            cnnobj2 = ku.compilecnnmodel(cnnmod, [patchsize, patchsize, ancdatasets.shape[2]], lrate, dropout,
                                         filters=filters, lweights=lweights, hubervalue=hubervalue, stdivalue=stdivalue,
                                         previters=previters, lossparams=lossparams)
            ku.savemodel(cnnobj2, 'Temp/models_' + casestudy + '_2cnn', includeoptimizer=True)


    strat = True
    if any(meth.startswith('ap') for meth in method):
        if iterdissolv:
            if strat:
                numpols = round((perc2evaluate/2) * (len(adjpolygons)))
                pquartilles = gput.polygonsQuartilles(idpolvalues, numpols)
                adjpairs = gput.createAdjPairs(adjpolygons, perc2evaluate/2, strat=pquartilles, verbose=True)
            else:
                adjpairs = gput.createAdjPairs(adjpolygons, perc2evaluate/2, verbose=True)

            initadjpairs = adjpairs
            if(verbose): print('Fixed adjacent pairs (' + str(len(initadjpairs)) + ') -', initadjpairs)

    lasterror = -np.inf
    lowesterror = np.inf
    for k in range(1, max_iter+1):
        print('\n| - Iteration', k)

        methodit = method[0] if (k % 2) == 1 else method[1]

        previdpolvalues = idpolvalues # Original polygon values
        if methodit.startswith('ap'):
            if iterdissolv:
                if strat:
                    pquartilles = gput.polygonsQuartilles(idpolvalues, numpols)
                    adjpairs = gput.createAdjPairs(adjpolygons, perc2evaluate / 2, strat=pquartilles, initadjpairs=initadjpairs, verbose=True)
                else:
                    adjpairs = gput.createAdjPairs(adjpolygons, perc2evaluate/2, initadjpairs=initadjpairs)

                if (verbose): print('Adjacent pairs (' + str(len(adjpairs)) + ') -', adjpairs)
                newshape, newpairs = gput.dissolvePairs(fshape, adjpairs)


            idsdataset2e = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0]

            # Edit idpolvalues
            pairslist = [item for t in adjpairs for item in t]

            ids2keep = list(set(previdpolvalues.keys()))
            idpolvalues = dict((k, previdpolvalues[k]) for k in ids2keep)

            idpolvalues2e = dict((k, previdpolvalues[k]) for k in pairslist)
            polygonarea2e = dict((k, polygonareas[k]) for k in pairslist)


        if methodit.endswith('cnn'):
            print('| -- Updating disaggregation patches')
            disaggdataset = disaggdataset * dissmask
            disspatches = ku.createpatches(disaggdataset, patchsize, padding=padd, stride=1)

            print('| -- Fitting the model')
            if not methodit.endswith('2cnn'):
                cnnobj = ku.loadmodel('Temp/models_' + casestudy, hubervalue, stdivalue)
            else:
                cnnobj2 = ku.loadmodel('Temp/models_' + casestudy + '_2cnn', hubervalue, stdivalue)


            if previters:  # Include previous iterations
                disspatches = np.concatenate((disspatches, previtersarr), axis=3)

            if not methodit.endswith('2cnn'):
                fithistory = caret.fitcnn(ancpatches, disspatches, p, cnnmod=cnnmod, cnnobj=cnnobj,
                                          casestudy=casestudy, epochs=epochspi, batchsize=batchsize,
                                          extdataset=extdataset)
                if not resetweights: ku.savemodel(cnnobj, 'Temp/models_' + casestudy, includeoptimizer=True)
            else:
                fithistory = caret.fitcnn(ancpatches, disspatches, p, cnnmod=cnnmod, cnnobj=cnnobj2,
                                          casestudy=casestudy + '_2cnn', epochs=epochspi, batchsize=batchsize,
                                          extdataset=extdataset)
                if not resetweights: ku.savemodel(cnnobj2, 'Temp/models_' + casestudy + '_2cnn', includeoptimizer=True)


            print('| -- Predicting new values')
            if not methodit.endswith('2cnn'):
                predictedmaps = caret.predictcnn(cnnobj, cnnmod, fithistory, casestudy, ancpatches,
                                                 disaggdataset.shape, batchsize=batchsize)
            else:
                predictedmaps = caret.predictcnn(cnnobj2, cnnmod, fithistory, casestudy + '_2cnn', ancpatches,
                                                 disaggdataset.shape, batchsize=batchsize)


            ## Ensemble of two regression algorithms
            # methodoa = 'aprf'
            # ancdatasets[np.isnan(ancdatasets)] = 0
            # modoa = caret.fit(ancdatasets, disaggdataset, p, methodoa, batchsize, lrate, epochspi)
            # predictedmapsoa = caret.predict(modoa, ancdatasets)
            # for i in range(len(predictedmapsoa)): predictedmapsoa[i] = np.expand_dims(predictedmapsoa[i], axis=2)
            # for i in range(len(predictedmaps)): predictedmaps[i] = 0.5 * predictedmaps[i] + 0.5 * predictedmapsoa[i]


        else:
            print('| -- Fitting the model')
            # Replace NaN's by 0
            ancdatasets[np.isnan(ancdatasets)] = 0
            disaggdataset = disaggdataset * dissmask

            mod = caret.fit(ancdatasets, disaggdataset, p, methodit, batchsize, lrate, epochspi)

            print('| -- Predicting new values')
            predictedmaps = mod if methodit.endswith('rnoise') else caret.predict(mod, ancdatasets)
            for i in range(len(predictedmaps)): predictedmaps[i] = np.expand_dims(predictedmaps[i], axis=2)
            if k == 1: fithistory = [0]


        bestmaepredictedmaps = float("inf")
        for i, predmap in enumerate(predictedmaps):
            # Replace NaN zones by Nan
            predmap = predmap * dissmask
            predmap[predmap < 0] = 0
            predmap2e = np.copy(predmap)
            ancdatasets = ancdatasets * ancvarsmask
            metricsmap = mev.report_sdev_map(predmap)

            if verbose: print('| -- Computing adjustement factor')
            stats = npu.statsByID(predmap, idsdataset, 'sum')

            if methodit.startswith('ap'):
                stats2e = npu.statsByID(predmap2e, idsdataset2e, 'sum')
                stats2e = dict((k, stats2e[k]) for k in pairslist)

            # Horrible hack, avoid division by 0
            for s in stats: stats[s] = stats[s] + 0.00001
            for s in stats2e: stats2e[s] = stats2e[s] + 0.00001

            polygonratios = {k: idpolvalues[k] / stats[k] for k in stats.keys() & idpolvalues}
            polygonratios2e = {k: idpolvalues2e[k] / stats2e[k] for k in stats2e.keys() & idpolvalues2e}
            idpolvalues = previdpolvalues

            # Mass-preserving adjustment
            for polid in polygonratios:
                predmap[idsdataset == polid] = (predmap[idsdataset == polid] * polygonratios[polid])
            for polid in polygonratios2e:
                predmap2e[idsdataset2e == polid] = (predmap2e[idsdataset2e == polid] * polygonratios2e[polid])

            if methodit.startswith('ap'):
                # Compute metrics for the evaluation municipalities
                actual2e = list(idpolvalues2e.values())
                predicted2e = list(stats2e.values())
                areas2e = list(polygonarea2e.values())
                range2e = max(actual2e) - min(actual2e)

                mae2e, wae2e = mev.mean_absolute_error(actual2e, predicted2e, areas2e)
                rmse2e = np.sqrt(metrics.mean_squared_error(actual2e, predicted2e))
                metricsmae2e = mev.report_mae_y(actual2e, predicted2e)
                metricsrmse2e = mev.report_rmse_y(actual2e, predicted2e)
                metricsr22e = mev.report_r2_y(actual2e, predicted2e)

                if os.path.exists(filenamemetrics2e):
                    with open(filenamemetrics2e, 'a') as myfile:
                        myfile.write(str(k) + ';' + str(mae2e) + ';' + str(rmse2e))
                        for metric in metricsmap: myfile.write(';' + str(metric))
                        for metric in metricsmae2e: myfile.write(';' + str(metric))
                        for metric in metricsrmse2e: myfile.write(';' + str(metric))
                        for metric in metricsr22e: myfile.write(';' + str(metric))
                        myfile.write(';' + str(fithistory[i]))
                else:
                    with open(filenamemetrics2e, 'w+') as myfile:
                        myfile.write('IT;MAE;RMSE;STDMEAN;MAEMEAN;RMSEMEAN;R2;LOSS;ERROR2IT\n')
                        myfile.write(str(k) + ';' + str(mae2e) + ';' + str(rmse2e))
                        for metric in metricsmap: myfile.write(';' + str(metric))
                        for metric in metricsmae2e: myfile.write(';' + str(metric))
                        for metric in metricsrmse2e: myfile.write(';' + str(metric))
                        for metric in metricsr22e: myfile.write(';' + str(metric))
                        myfile.write(';' + str(fithistory[i]))

                if metricsmae2e[0] < bestmaepredictedmaps:
                    bestmaepredictedmaps = metricsmae2e[0]


            if methodit.endswith('cnn') and previters:  # Include previous iterations
                # Update dataset of patches from previous iterations
                prevpatchesaux = np.copy(disspatches[:,:,:,0:1]) # Only 1st, with 4 dimensions
                updatep = 0.25
                # updatep = 1/(pow(k, 0.25)) * updatep  # Apply decay
                idspreviters = np.random.choice(len(prevpatchesaux), round(len(prevpatchesaux) * updatep), replace=False)
                previtersarr[idspreviters] = prevpatchesaux[idspreviters]

            # Assuming 1 predicted map
            disaggdataset = predmap


        osgu.writeRaster(disaggdataset[:, :, 0], rastergeo, 'pcounts_' + casestudy + '_' + str(k).zfill(2) + 'it.tif')

        # Check if the algorithm converged
        error = np.nanmean(abs(disaggdataset-olddisaggdataset))
        with open(filenamemetrics2e, 'a') as myfile: myfile.write(';' + str(error) + '\n')
        errorrat = (error/lasterror) if lasterror>0 else np.inf
        lasterror = error
        print('Error:', error)

        if k >= min_iter:
            if errorrat < converge:
                if error < lowesterror:
                    lowesterror = error
                    lowesterriter = k
                    lowesterrdisaggdataset = np.copy(disaggdataset)
            else:
                if k == min_iter:
                    lowesterriter = k
                else:
                    disaggdataset = lowesterrdisaggdataset
                print('Retaining model fitted at iteration', lowesterriter)
                break
        olddisaggdataset = disaggdataset


    if tempfileid:
        tempfile = 'tempfiledisagg_' + tempfileid + '.tif'
        osgu.writeRaster(disaggdataset, rastergeo, tempfile)


    ## Remove dataset containing the patches of ancillary data from disk
    # os.remove(cstudyad + '.dat')

    return disaggdataset[:,:,0], rastergeo
