import pandas as pd
import re


csdict = {}

chosopt = False
with open("ev.txt") as f:
    for line in f:
        if line.startswith('- Error metrics'):
            try:
                casestudy = re.findall(r'\((.*)_\d{2}it', line)[0]
            except:
                casestudy = re.findall(r'\((.*)_\d{3}it', line)[0]
            it = line.split('it.')[0].split('_')[-1]

            line2 = f.readline().lstrip('- ').rstrip()
            errorvalues = line2.split(' & ')
            if not chosopt:
                print(errorvalues)
                opt = input('METRIC (0, 1, 2, 3..)?: ')
                chosopt = True

            metricvalue = errorvalues[int(opt)]
            rmsevalue = errorvalues[0]
            r2value = errorvalues[6]
            if casestudy not in csdict:
                csdict[casestudy] = pd.DataFrame([[it, metricvalue, rmsevalue, r2value]], columns = ['IT', 'MAE', 'RMSE', 'R2'])
            else:
                csdict[casestudy] = csdict[casestudy].append(pd.DataFrame([[it, metricvalue, rmsevalue, r2value]], columns = ['IT', 'MAE', 'RMSE', 'R2']))


for cs in csdict:
    csdict[cs].sort_values(by=['IT']).to_csv(cs + '.csv', sep=';', index=False)
