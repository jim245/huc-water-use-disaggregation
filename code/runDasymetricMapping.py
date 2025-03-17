import osgeoutils as osgu, dasymmapping as dm
import os


indicators = [['Withdrawals', 'NUTSIII', 'NUTSIII']]
popraster = 'ghspg_2015_200m.tif'
templateraster = 'template_200m.tif'


for indicator in indicators:
    print('--- Running dasymetric mapping for the indicator', indicator[0])

    # Read rasters and shapefiles
    ds, rastergeo = osgu.readRaster(os.path.join('Rasters', indicator[0], popraster))
    dstemplate = osgu.readRaster(os.path.join('Results', indicator[0], 'Baselines', templateraster))[0]
    nrowsds = ds.shape[1]
    ncolsds = ds.shape[0]

    fshapea = os.path.join('Shapefiles', indicator[0], (indicator[2] + '.shp'))
    fshape = osgu.copyShape(fshapea, 'dasymapping')
    fcsv = os.path.join('Statistics', indicator[0], (indicator[2] + '.csv'))

    fancdataset = os.path.join('Rasters', indicator[0], popraster)

    # Merge shapefile and .csv file
    osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
    osgu.addAttr2Shapefile(fshape, fcsv, [indicator[2].upper()], encoding='UTF-8')

    # Create datasets and run dasymetric mapping
    tempfileid = None #None
    idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0]
    polygonvaluesdataset, rastergeo = osgu.ogr2raster(fshape, attr='VALUE', template=[rastergeo, nrowsds, ncolsds])
    ancdataset = osgu.readRaster(fancdataset)[0]
    tddataset, rastergeo = dm.rundasymmapping(idsdataset, polygonvaluesdataset, ancdataset, rastergeo, tempfileid=tempfileid)
    tddataset = tddataset * dstemplate

    # Write raster and remove temp files
    osgu.writeRaster(tddataset[:,:,0], rastergeo, indicator[0] + '_td.tif')
    osgu.removeShape(fshape)
