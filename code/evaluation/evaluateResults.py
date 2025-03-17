from sklearn import metrics
import numpy as np, pandas as pd
import os, shutil, sys

sys.path.append('..')
from utils import metricsev as mev


indicators = [['Withdrawals', 'NUTSIII', 'MUNICIP']]

def rmse_error(actual, predicted):
    return np.sqrt(metrics.mean_squared_error(actual, predicted))

def mae_error(actual, predicted):
    return metrics.mean_absolute_error(actual, predicted)

def nrmse_error(actual, predicted):
    range = max(actual) - min(actual)
    return (np.sqrt(metrics.mean_squared_error(actual, predicted)))/range

def nmae_error(actual, predicted):
    range = max(actual) - min(actual)
    return (metrics.mean_absolute_error(actual, predicted))/range


for indicator in indicators:
    print('\n\n--- EVALUATING', indicator[0])

    # Read census .csv
    data_census = pd.read_csv(os.path.join('../../statistics', indicator[0], indicator[2] + '.csv'), sep=';', index_col=False)
    data_census[indicator[1].upper()] = data_census[indicator[1].upper()].astype(str)
    data_census[indicator[2].upper()] = data_census[indicator[2].upper()].astype(str)
    data_census[indicator[1].upper()] = data_census[indicator[1].upper()].str.upper()
    data_census[indicator[2].upper()] = data_census[indicator[2].upper()].str.upper()

    path = os.path.join('../../estimates', indicator[0], '2Evaluate')
    newpath = os.path.join('../../estimates', indicator[0])

    file_names = os.listdir(path)
    for file in file_names:
        if file.endswith('.csv'):
            filee = os.path.join(path, file)
            newpathe = os.path.join(newpath, file)

            data_estimated = pd.read_csv(filee, sep=";")
            data_estimated[indicator[2].upper()] = data_estimated[indicator[2].upper()].astype(str)
            data_estimated[indicator[2].upper()] = data_estimated[indicator[2].upper()].str.upper()

            actual = []
            predicted = []
            for index, row in data_estimated.iterrows():
                name_ab1 = row[indicator[1].upper()]
                name_ab2 = row[indicator[2].upper()]
                predicted_value = row['VALUE']

                actual_value = data_census.loc[(data_census[indicator[1].upper()] == name_ab1) &
                                               (data_census[indicator[2].upper()] == name_ab2), 'VALUE']

                if (len(actual_value.index) == 1):
                    actual.append(actual_value.values[0])
                    predicted.append(predicted_value)
                else:
                    print(actual_value)
                    print(name_ab1, name_ab2)


            print('\n- Error metrics (' + file + '):')

            r1 = round(rmse_error(actual, predicted), 4)
            r2 = round(mae_error(actual, predicted), 4)
            r3 = round(nrmse_error(actual, predicted), 4)
            r4 = round(nmae_error(actual, predicted), 4)
            maenorm = mev.report_mae_y(actual, predicted)
            rmsenorm = mev.report_rmse_y(actual, predicted)
            r2norm = mev.report_r2_y(actual, predicted)

            print('-', r1, '&', r2, '&', r3, '&', r4, end='')
            for maeit in maenorm: print(' &', maeit, end='')
            for rmseit in rmsenorm: print(' &', rmseit, end='')
            for r2it in r2norm: print(' &', r2it)

            shutil.copyfile(filee, newpathe)
            os.remove(filee)

