import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import osgeoutils as osgu
from disag import disaggregate

import numpy as np
import itertools, random
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = '1'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Turn off GPU


# Parameters to be defined
indicator = 'Withdrawals'
admboundary2 = 'NUTSIII'
methodopts = [['apcnn', 'aprf']]
cnnmodelopts = ['unet']
epochspiopts = [3]
ymethodopts = ['smoothtd']
resetwopts = [False]

# Extra
psamplesopts = [[0.01]]
batchsizeopts = [512]
learningrateopts = [0.001]
extendeddatasetopts = [None]
lossweightsopts = [[0.1, 0.9]]
perc2evaluateopts = [1]
hubervalueopts = [1]
stdivalueopts = [40]
dropoutopts = [0.5]
lossparamsopts = [[2,0,0,False]]

# Read rasters and shapefiles
fshapea = os.path.join('shapefiles', indicator, (admboundary2 + '.shp'))
fcsv = os.path.join('statistics', indicator, (admboundary2 + '.csv'))

ancdataset1, rastergeo = osgu.readRaster(os.path.join('rasters', indicator, 'Normalized', 'ghspg_2015_200m.tif'))
ancdataset2 = osgu.readRaster(os.path.join('rasters', indicator, 'Normalized', 'hs_2012_200m.tif'))[0]
ancdataset3 = osgu.readRaster(os.path.join('rasters', indicator, 'Normalized', 'bua_2018_200m.tif'))[0]
ancdataset4 = osgu.readRaster(os.path.join('rasters', indicator, 'Normalized', 'lc_2018_200m.tif'))[0]
ancdataset5 = osgu.readRaster(os.path.join('rasters', indicator, 'Normalized', 'nl_2016_200m.tif'))[0]
ancdatasets = np.dstack((ancdataset1, ancdataset2, ancdataset3, ancdataset4, ancdataset5))
ancdatasetsopts = [ancdatasets]

# Merge shapefile and .csv file
fshape = osgu.copyShape(fshapea, 'disaggregate')
if not ymethodopts: osgu.addAttr2Shapefile(fshape, fcsv, admboundary2.upper())


def setseedandgrowth(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu_instance in gpus:
        tf.config.experimental.set_memory_growth(gpu_instance, True)


i = 1 # Id of current test
for (ancdts, psamples, method, perc2evaluate, ymethod) in itertools.product(ancdatasetsopts,
                                                                            psamplesopts,
                                                                            methodopts,
                                                                            perc2evaluateopts,
                                                                            ymethodopts):
    if all(not meth.endswith('cnn') for meth in method):
        print('\n--- Running disaggregation leveraging a', method, 'model')

        yraster = os.path.join('results', indicator, 'Baselines', (ymethod + '_200m.tif'))
        casestudy = indicator + '_' + str(ancdts.shape[2]) + 'va'
        dissdataset, rastergeo = disaggregate.runDisaggregation(fshape, ancdts, min_iter=30, max_iter=30,
                                                      perc2evaluate=perc2evaluate, poly2agg=admboundary2.upper(),
                                                      rastergeo=rastergeo, method=method, p=psamples,
                                                      yraster=yraster, casestudy=casestudy,
                                                      verbose=True)

        print('- Writing raster to disk...')
        osgu.writeRaster(dissdataset, rastergeo, 'disaggestimates_' + casestudy + '.tif')

    else:
        for cnnmodel in cnnmodelopts:
            if cnnmodel.endswith('unet'):
                filtersopts = [[8, 16, 32, 64, 128]]
                patchsizeopts = [16]
            elif cnnmodel.endswith('unetplusplus'):
                filtersopts = [[8, 16, 32, 64, 128]]
                patchsizeopts = [16]
            else:
                filtersopts = [[14, 28, 56, 112, 224]]
                patchsizeopts = [7]

            for (lossweights, batchsize, epochpi, learningrate,
                 filters, patchsize, extendeddataset,
                 hubervalue, stdivalue, dropout,
                 resetweights, lossparams) in itertools.product(lossweightsopts,
                                                                      batchsizeopts,
                                                                      epochspiopts,
                                                                      learningrateopts,
                                                                      filtersopts,
                                                                      patchsizeopts,
                                                                      extendeddatasetopts,
                                                                      hubervalueopts,
                                                                      stdivalueopts,
                                                                      dropoutopts,
                                                                      resetwopts,
                                                                      lossparamsopts):
                print('\n--- Running disaggregation with the following CNN configuration:')
                print('- Method:', cnnmodel,
                      '| Epochs per iteration:', epochpi,
                      '| Batch size:', batchsize)

                setseedandgrowth(SEED)

                yraster = os.path.join('results', indicator, 'Baselines', (ymethod + '_200m.tif'))
                casestudy = indicator + '_' + cnnmodel + '_' + str(epochpi) + '_batchsize' + str(batchsize) + \
                            '_' + str(ancdts.shape[2]) + 'va-t' + str(i)
                dissdataset, rastergeo = disaggregate.runDisaggregation(fshape, ancdts, min_iter=30, max_iter=30,
                                                              perc2evaluate=perc2evaluate,
                                                              poly2agg=admboundary2.upper(),
                                                              rastergeo=rastergeo, method=method, p=psamples,
                                                              cnnmod=cnnmodel, patchsize=patchsize, batchsize=batchsize,
                                                              epochspi=epochpi, lrate=learningrate, filters=filters,
                                                              lweights=lossweights, extdataset=extendeddataset,
                                                              yraster=yraster, converge=1.5,
                                                              hubervalue=hubervalue, stdivalue=stdivalue,
                                                              dropout=dropout, casestudy=casestudy,
                                                              resetweights=resetweights, previters=False,
                                                              lossparams=lossparams,
                                                              verbose=True)

                print('- Writing raster to disk...')
                osgu.writeRaster(dissdataset, rastergeo, 'disaggestimates' + casestudy + '.tif')

    i = i + 1


osgu.removeShape(fshape)
