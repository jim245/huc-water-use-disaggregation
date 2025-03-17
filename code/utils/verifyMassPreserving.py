from utils import osgeoutils as osgu, nputils as npu
import numpy as np
import os, collections


indicators = [['Withdrawals', 'NUTSIII', 'NUTSIII']]

for indicator in indicators:
    fshapea = os.path.join('Shapefiles', indicator[0], (indicator[2] + '.shp'))
    fshape = osgu.copyShape(fshapea, 'verifymass')
    fcsv = os.path.join('Statistics', indicator[0], (indicator[2] + '.csv'))

    osgu.addAttr2Shapefile(fshape, fcsv, [indicator[2].upper()], encoding='utf-8')

    for file in os.listdir('Temp'):
        if file.startswith(indicator[0] + '_'):
            raster, rastergeo = osgu.readRaster('Temp/' + file)
            nrowsds = raster.shape[1]
            ncolsds = raster.shape[0]

            idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0]
            polygonvaluesdataset = osgu.ogr2raster(fshape, attr='VALUE', template=[rastergeo, nrowsds, ncolsds])[0]

            # Summarizes the values within each polygon
            stats_predicted = npu.statsByID(raster, idsdataset, 'sum')
            stats_true = npu.polygonValuesByID(polygonvaluesdataset, idsdataset)

            predicted = np.fromiter(collections.OrderedDict(sorted(stats_predicted.items())).values(), dtype=float)
            true = np.fromiter(collections.OrderedDict(sorted(stats_true.items())).values(), dtype=float)
            diff = predicted - true
            if np.abs(np.mean(diff)) > (0.00001 * np.mean(true)):
                print('Problem with file', file)
                print('- Difference:', np.abs(np.mean(diff)))
                print('- Max value:', 0.00001 * np.mean(true))

    osgu.removeShape(fshape)
