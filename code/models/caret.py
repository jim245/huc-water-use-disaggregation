import warnings, gc
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np, math
import skimage.measure
from utils import nputils as npu
from models import kerasutils as ku
from numpy import random
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras import utils
import tensorflow as tf

from scipy import ndimage


SEED = 42


def fitlm(X, y):
    mod = LinearRegression()
    mod = mod.fit(X, y)
    return mod

def fitsgdregressor(X, y, batchsize, lrate, epoch):
    mod = SGDRegressor(max_iter=epoch, alpha=0, learning_rate='constant', eta0=lrate, verbose=1)
    mod = mod.fit(X, y)
    return mod

def fitrf(X, y):
    mod = RandomForestRegressor(random_state=SEED, criterion='squared_error') # n_estimators = 10
    mod = mod.fit(X, y)
    return mod

def fitxgbtree(X, y):
    gbm = xgb.XGBRegressor(seed=SEED, eval_metric='rmse')
    mod = gbm.fit(X, y)
    return mod


def fit(X, y, p, method, batchsize, lrate, epoch):
    X = X.reshape((-1, X.shape[2]))
    originalshape = y.shape
    y = np.ravel(y)

    relevantids = np.where(~np.isnan(y))[0]
    relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p[0]), replace=False)
    idsamples = relevantids[relevsamples]
    print('| --- Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

    X = X[idsamples,:]
    y = y[idsamples]

    if(method.endswith('lm')):
        print('|| Fit: Linear Model')
        return fitlm(X, y)
    if (method.endswith('sgdregressor')):
        print('|| Fit: SGD Regressor')
        return fitsgdregressor(X, y, batchsize, lrate, epoch)
    elif(method.endswith('rf')):
        print('|| Fit: Random Forests')
        return fitrf(X, y)
    elif(method.endswith('xgbtree')):
        print('|| Fit: XGBTree')
        return fitxgbtree(X, y)
    elif (method.endswith('rnoise')):
        print('|| Applying random noise')
        originaly = np.resize(y, originalshape)
        row, col, ch = originaly.shape
        mean, sigma = 1, np.nanmean(originaly)*1
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        originalywnoise = originaly + gauss
        originalywnoise[originalywnoise < 0] = 0
        return [originalywnoise[:,:,0]]
    else:
        return None


def get_callbacks():
    return [
        EarlyStopping(monitor='loss', min_delta=0.01, patience=3, verbose=1, restore_best_weights=True)
    ]

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def loadgeary(path, patchsize):
    fp = np.memmap(path, mode='r')
    print('Found .dat file')
    ninstances = int(fp.shape[0] / patchsize / patchsize / 1 / 4)  # Divide by dimensions
    shapemmap = (ninstances, patchsize, patchsize, 1)
    fp = np.memmap(path, dtype='float32', mode='r', shape=shapemmap)
    return fp


def computeGeary(batch):
    def squareddiff(values):
        reshvalues = np.reshape(values, (3, 3))
        geary = np.square(reshvalues - reshvalues[1, 1]).sum()
        return geary

    gearybatch = np.empty(shape = (batch.shape[0], batch.shape[1], batch.shape[2]))
    mask = np.ones(shape = (3, 3))

    for i in range(len(batch)):
        instgeary = np.zeros(shape = (batch[i].shape[0], batch[i].shape[1]))
        for axisinst in np.rollaxis(batch[i], 2):
            instgeary = instgeary + ndimage.generic_filter(axisinst, squareddiff, footprint=mask, mode='nearest')
        gearybatch[i] = instgeary

    return gearybatch


class DataGenerator(utils.Sequence):
    'Generates data for Keras'

    def __init__(self, Xdataset, ydataset, idsamples, batch_size, p):
        'Initialization'
        self.X, self.y = Xdataset, ydataset
        self.batch_size = batch_size
        self.p = p
        self.idsamples = idsamples

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.idsamples) / self.batch_size)

    def __getitem__(self, idx):
        'Generate one batch of data'
        idsamplesbatch = self.idsamples[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X[idsamplesbatch]
        batch_X[np.isnan(batch_X)] = 0
        batch_y = self.y[idsamplesbatch]
        return np.array(batch_X), np.array(batch_y)

    def on_epoch_end(self):
        # Housekeeping
        gc.collect()
        tf.keras.backend.clear_session()


def fitcnn(X, y, p, cnnmod, cnnobj, casestudy, epochs, batchsize, extdataset):
    tf.random.set_seed(SEED)

    if cnnmod == 'lenet':
        print('| --- Fit - 1 resolution Le-Net')

        # Select midd pixel from patches
        middrow, middcol = int((y.shape[1]-1)/2), int(round((y.shape[2]-1)/2))
        y = y[:,middrow,middcol,0]

        relevantids = np.where(~np.isnan(y))[0]
        relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p[0]), replace=False)
        idsamples = relevantids[relevsamples]
        print('Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

        if extdataset:
            newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
            return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
        else:
            Xfit = X[idsamples, :, :, :]
            yfit = y[idsamples]
            hislist = [cnnobj.fit(Xfit, yfit, epochs=epochs, batch_size=batchsize)]
            return hislist

    elif cnnmod == 'vgg':
        print('| --- Fit - 1 resolution VGG-Net')

        # Select midd pixel from patches
        middrow, middcol = int((y.shape[1]-1)/2), int(round((y.shape[2]-1)/2))
        y = y[:,middrow,middcol,0]

        relevantids = np.where(~np.isnan(y))[0]
        relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p), replace=False)
        idsamples = relevantids[relevsamples]
        print('Number of instances (All, >1, X%>1):', y.shape[0], len(relevantids), len(idsamples))

        if extdataset:
            newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
            return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
        else:
            return cnnobj.fit(X[idsamples, :, :, :], y[idsamples], epochs=epochs, batch_size=batchsize)

    elif cnnmod == 'uenc':
        print('| --- Fit - 1 resolution U-Net encoder')

        # Select midd pixel from patches
        middrow, middcol = int((y.shape[1]-1)/2), int(round((y.shape[2]-1)/2))
        y = y[:,middrow,middcol,0]

        relevantids = np.where(~np.isnan(y))[0]
        relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p), replace=False)
        idsamples = relevantids[relevsamples]
        print('Number of instances (All, >1, X%>1):', y.shape[0], len(relevantids), len(idsamples))

        if extdataset:
            newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
            return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
        else:
            return cnnobj.fit(X[idsamples, :, :, :], y[idsamples], epochs=epochs, batch_size=batchsize)

    elif cnnmod.endswith('unet') or cnnmod.endswith('unetplusplus'):
        # Compute midd pixel from patches
        middrow, middcol = int((y.shape[1]-1)/2), int(round((y.shape[2]-1)/2))

        # Train only with patches having middpixel different from NaN
        relevantids = np.where(~np.isnan(y[:,middrow,middcol,0]))[0]

        # Train only with patches having finit values in all pixels
        # relevantids = np.where(~np.isnan(y).any(axis=(1,2,3)))[1]

        if len(p) > 1:
            relevsamples = np.random.choice(len(relevantids), round(len(relevantids)), replace=False)
        else:
            relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p[0]), replace=False)
        idsamples = relevantids[relevsamples]


        print('Number of instances (All, >1, X%>1):', y.shape[0], len(relevantids), len(idsamples))

        if(cnnmod == 'unet' or cnnmod == 'unetplusplus'):
            computegeary = False
            pathgeary = 'Withdrawals_ghspghsbualcnl_16_nopadd_extended_geary.dat'

            if extdataset:
                Xfit = X[idsamples, :, :, :]
                yfit = y[idsamples]
                yfit[np.isnan(yfit)] = 0
                newX, newy = npu.extenddataset(Xfit, yfit, transf=extdataset)
                hislist = [cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)]
                return hislist

            else:
                print('| --- Fit - 1 resolution U-Net Model')

                if epochs == 100: # With callback
                    ## Training generator
                    gearyarr = loadgeary(pathgeary, 16) if computegeary == True else np.empty(shape=0)
                    if computegeary: y = np.concatenate((y, gearyarr), axis=3)
                    training_generator = DataGenerator(X, y, idsamples, batch_size=batchsize, p=p[0])
                    hislist = [cnnobj.fit(training_generator, epochs=100, callbacks=get_callbacks())]

                else:
                    ## Training generator
                    lhistory = LossHistory()
                    gearyarr = loadgeary(pathgeary, 16) if computegeary == True else np.empty(shape=0)
                    if computegeary: y = np.concatenate((y, gearyarr), axis=3)
                    training_generator = DataGenerator(X, y, idsamples, batch_size=batchsize, p=p[0])
                    cnnobj.fit(training_generator, epochs=epochs, callbacks=lhistory)
                    hislist = [lhistory.losses[-1]]

                return hislist

        elif(cnnmod.startswith('2r')):
            print('| --- Fit - 2 resolution U-Net Model')
            if extdataset:
                newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
            else:
                newX, newy = X[idsamples, :, :, :], y[idsamples]
            newylr = skimage.measure.block_reduce(newy, (1,4,4,1), np.sum) # Compute sum in 4x4 blocks
            newy = [newy, newylr]

            return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)

    else:
        print('Fit CNN - Unknown model')


def predict(mod, X):
    newX = X.reshape((-1, X.shape[2]))
    pred = mod.predict(newX)
    pred = pred.reshape(X.shape[0], X.shape[1])
    return [pred]


def predictloop(cnnmod, patches, batchsize):
    # Custom batched prediction loop
    final_shape = [patches.shape[0], patches.shape[1], patches.shape[2], 1]
    y_pred_probs = np.empty(final_shape,
                            dtype=np.float32)  # pre-allocate required memory for array for efficiency

    # Array with first number for each batch
    batch_indices = np.arange(start=0, stop=patches.shape[0], step=batchsize)  # row indices of batches
    batch_indices = np.append(batch_indices, patches.shape[0])  # add final batch_end row

    i=1
    for index in np.arange(len(batch_indices) - 1):
        batch_start = batch_indices[index]  # first row of the batch
        batch_end = batch_indices[index + 1]  # last row of the batch

        # Replace NANs in patches by zero
        ctignore = np.isnan(patches[batch_start:batch_end])
        patchespred = patches[batch_start:batch_end].copy()
        patchespred[ctignore] = 0

        # Mean (pred original, pred rotated)
        y_pred_probs[batch_start:batch_end] = np.expand_dims(
            np.mean(cnnmod.predict_on_batch(patchespred)[:,:,:,:2], axis=3), axis=3)

        # Replace original NANs with NAN
        patchespred[ctignore] = np.nan

        i = i+1
        if(i%1000 == 0): print('»» Batch', i, '/', len(batch_indices), end='\r')

    return y_pred_probs


def predictcnn(obj, mod, fithistory, casestudy, ancpatches, dissshape, batchsize, stride=1):
    if mod == 'lenet':
        print('| --- Predicting new values, Le-Net')
        predhr = obj.predict(ancpatches, batch_size=batchsize, verbose=1)
        predhr = predhr.reshape(dissshape)
        return [predhr]

    elif mod == 'vgg':
        print('| --- Predicting new values, VGG-Net')
        predhr = obj.predict(ancpatches, batch_size=batchsize, verbose=1)
        predhr = predhr.reshape(dissshape)
        return [predhr]

    elif mod == 'uenc':
        print('| --- Predicting new values, U-Net encoder')
        predhr = obj.predict(ancpatches, batch_size=batchsize, verbose=1)
        predhr = predhr.reshape(dissshape)
        return [predhr]

    elif mod.endswith('unet') or mod.endswith('unetplusplus'):
        if(mod == 'unet' or mod == 'unetplusplus'):
            if len(fithistory) > 1:
                print('| --- Predicting new patches from several models, 1 resolution U-Net / U-Net++')
                predhr = []
                for i in range(len(fithistory)):
                    obj.load_weights('snapshot_' + casestudy + '_' + str(i) + '.h5')
                    predhr.append(obj.predict(ancpatches, batch_size=batchsize, verbose=1))
                print('| ---- Reconstructing HR images from patches..')
                for i in range(len(predhr)): predhr[i] = ku.reconstructpatches(predhr[i], dissshape, stride)
                return predhr
            else:
                print('| --- Predicting new patches, 1 resolution U-Net / U-Net++')
                predhr = predictloop(obj, ancpatches, batchsize=batchsize)

                print('| ---- Reconstructing HR image from patches..')
                predhr = ku.reconstructpatches(predhr, dissshape, stride)

                return [predhr]

        elif mod.startswith('2r'):
            print('| --- Predicting new patches, 2 resolution U-Net')
            predhr = obj.predict(ancpatches, batch_size=batchsize, verbose=1)[0]
            print('| ---- Reconstructing HR image from patches..')
            predhr = ku.reconstructpatches(predhr, dissshape, stride)
            return predhr

    else:
        print('Predict CNN - Unknown model')
