import numpy as np, random
from sklearn.feature_extraction.image import extract_patches_2d
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.backend import *
from tensorflow.keras import optimizers
from itertools import product

from models import adaptiveloss as aloss

from tensorflow.keras import backend as K
import tensorflow as tf

SEED = 42


# Customized model with GradNorm
class CustomModel(tf.keras.Model):
    def __init__(self, inputs, outputs, hubervalue, stdivalue, alpha, learning_rate, ntasks):
        super(CustomModel, self).__init__(inputs, outputs)
        self.sla = smoothLC1a(hubervalue, stdivalue)
        self.slb = smoothLC1b(hubervalue, stdivalue)
        self.slc = smoothLC1c(hubervalue, stdivalue)
        self.a = tf.constant(alpha, dtype=tf.float32)
        self.learning_rate = learning_rate
        self.ntasks = ntasks

        self.gnweights = [tf.Variable(1.0, trainable=True),
                          tf.Variable(1.0, trainable=True),
                          tf.Variable(1.0, trainable=True)]
        self.firstloss = [tf.Variable(np.nan, trainable=False),
                          tf.Variable(np.nan, trainable=False),
                          tf.Variable(np.nan, trainable=False)]
        self.gotloss = 0

        if ntasks == 4:
            self.mseg = mean_squared_error_geary()
            self.gnweights.append(tf.Variable(1.0, trainable=True))
            self.firstloss.append(tf.Variable(np.nan, trainable=False))


    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            with tf.GradientTape(persistent=True) as tape2:
                y_pred = self(x, training=True)  # Forward pass

                # Compute the loss values
                loss_t1 = self.sla(y, y_pred)
                loss_t2 = self.slb(y, y_pred)
                loss_t3 = self.slc(y, y_pred)
                if self.ntasks == 4: loss_t4 = self.mseg(y, y_pred)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)  # Must do

                # Weight loss values
                l1 = tf.multiply(loss_t1, self.gnweights[0])
                l2 = tf.multiply(loss_t2, self.gnweights[1])
                l3 = tf.multiply(loss_t3, self.gnweights[2])
                if self.ntasks == 4: l4 = tf.multiply(loss_t4, self.gnweights[3])

                lossesv = [l1, l2, l3]
                if self.ntasks == 4: lossesv.append(l4)
                loss = tf.add_n(lossesv)


            if self.gotloss == 0:
                self.firstloss = [loss_t1, loss_t2, loss_t3]
                if self.ntasks == 4: self.firstloss.append(loss_t4)
                self.gotloss = 1

            # Compute gradients
            trainable_vars = self.trainable_variables
            nweights = len(self.gnweights)
            model_vars = trainable_vars[:len(trainable_vars) - nweights]
            model_vars_gn = model_vars[76:]
            ws_vars = trainable_vars[len(trainable_vars) - nweights:]

            # Compute GW(i)
            G1 = tape2.gradient(l1, model_vars_gn)
            G2 = tape2.gradient(l2, model_vars_gn)
            G3 = tape2.gradient(l3, model_vars_gn)
            if self.ntasks == 4: G4 = tape2.gradient(l4, model_vars_gn)
            G1 = [tf.norm(gw, ord=2) for gw in G1]
            G2 = [tf.norm(gw, ord=2) for gw in G2]
            G3 = [tf.norm(gw, ord=2) for gw in G3]
            if self.ntasks == 4: G4 = [tf.norm(gw, ord=2) for gw in G4]

            # Compute -GW
            gradientsv = [G1, G2, G3]
            if self.ntasks == 4: gradientsv.append(G4)
            G_avg = tf.divide(tf.add_n(gradientsv), len(gradientsv))

            # Compute ri(t)
            l_hat_1 = tf.divide(l1, self.firstloss[0])
            l_hat_2 = tf.divide(l2, self.firstloss[1])
            l_hat_3 = tf.divide(l3, self.firstloss[2])
            if self.ntasks == 4: l_hat_4 = tf.divide(l4, self.firstloss[3])

            lhatv = [l_hat_1, l_hat_2, l_hat_3]
            if self.ntasks == 4: lhatv.append(l_hat_4)
            l_hat_avg = tf.divide(tf.add_n(lhatv), len(lhatv))
            inv_rate_1 = tf.divide(l_hat_1, l_hat_avg)
            inv_rate_2 = tf.divide(l_hat_2, l_hat_avg)
            inv_rate_3 = tf.divide(l_hat_3, l_hat_avg)
            if self.ntasks == 4: inv_rate_4 = tf.divide(l_hat_4, l_hat_avg)

            # Compute constant target
            CT1 = tf.multiply(G_avg, tf.pow(inv_rate_1, self.a))
            CT2 = tf.multiply(G_avg, tf.pow(inv_rate_2, self.a))
            CT3 = tf.multiply(G_avg, tf.pow(inv_rate_3, self.a))
            if self.ntasks == 4: CT4 = tf.multiply(G_avg, tf.pow(inv_rate_4, self.a))
            CT1 = tf.stop_gradient(tf.identity(CT1))
            CT2 = tf.stop_gradient(tf.identity(CT2))
            CT3 = tf.stop_gradient(tf.identity(CT3))
            if self.ntasks == 4: CT3 = tf.stop_gradient(tf.identity(CT4))

            # Compute Lgrad
            redsumv = [tf.reduce_sum(tf.abs(tf.subtract(G1, CT1))),
                        tf.reduce_sum(tf.abs(tf.subtract(G2, CT2))),
                        tf.reduce_sum(tf.abs(tf.subtract(G3, CT3)))]
            if self.ntasks == 4: redsumv.append(tf.reduce_sum(tf.abs(tf.subtract(G4, CT4))))
            loss_gradnorm = tf.add_n(redsumv)

        # Gradnorm optimization step
        Gws = tape.gradient(loss_gradnorm, ws_vars)

        # Compute standard gradients
        gradients = tape.gradient(loss, model_vars_gn)

        # Update model weights and task weights
        G_all = gradients + Gws
        vars_all = model_vars_gn + ws_vars
        self.optimizer.apply_gradients(zip(G_all, vars_all))

        # Normalize weights
        gnweightsv = [self.gnweights[0], self.gnweights[1], self.gnweights[2]]
        if self.ntasks == 4: gnweightsv.append(self.gnweights[3])
        coef = tf.divide(len(gnweightsv), tf.add_n(gnweightsv))
        self.gnweights[0] = tf.multiply(self.gnweights[0], coef)
        self.gnweights[1] = tf.multiply(self.gnweights[1], coef)
        self.gnweights[2] = tf.multiply(self.gnweights[2], coef)
        if self.ntasks == 4: self.gnweights[3] = tf.multiply(self.gnweights[3], coef)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        del tape, tape2

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


# Customized model, with adaptive loss function
class CustomModelAdaptive(tf.keras.Model):
    def __init__(self, inputs, outputs):
        super(CustomModelAdaptive, self).__init__(inputs, outputs)
        self.aux_l1 = aloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.aux_l2 = aloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
        self.sladapt = smoothLC1Adapt(self.aux_l1, self.aux_l2)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)  # Forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses) # Must do
            loss = self.sladapt(y, y_pred)

        # Compute gradients
        model_vars = self.trainable_variables
        loss_vars = tf.unstack(self.aux_l1.trainable_variables + self.aux_l2.trainable_variables)
        trainable_vars = list(model_vars) + list(loss_vars)
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def custom_activation(x):
    return K.elu(x)


def averageinst(y_pred, i):
    y_pred[0] = tf.math.add(y_pred[0][:,:,:,i], y_pred[1][:,:,:,i])
    y_pred[0] = tf.math.divide(y_pred[0], tf.cast(2, tf.float32))
    return y_pred

def averagepreds(y_pred):
    i = tf.constant(0)
    y_pred = tf.while_loop(tf.less(i, len(y_pred[0][-1])), averageinst(y_pred, i))
    return y_pred[0]


# Huber loss, 1 prediction
def smoothL1(hubervalue = 0.5, stdivalue = 0.01):
    def sl1(y_true, y_pred):
        numfinite = tf.math.count_nonzero(tf.math.is_finite(y_true))
        mask = tf.where(tf.math.is_nan(y_true), K.constant(0), K.constant(1))
        y_true = tf.math.multiply_no_nan(y_true, mask)
        y_pred = tf.math.multiply_no_nan(y_pred, mask)
        y_pred = tf.where(tf.math.is_nan(y_pred), K.constant(0), y_pred)

        # Huber Loss
        HUBER_DELTA = hubervalue
        x = K.abs(y_true - y_pred)
        x = tf.where(x < HUBER_DELTA, 1.5 * x ** 2, HUBER_DELTA * (x - 1.5 * HUBER_DELTA))
        lossv = tf.math.divide_no_nan(K.sum(x), tf.cast(numfinite, tf.float32))

        ## MAE
        # err = y_pred - y_true
        # lossv = tf.math.divide_no_nan(K.sum(err), tf.cast(numfinite, tf.float32))

        ## RMSE
        # se = K.square(y_pred - y_true)
        # mse = tf.math.divide_no_nan(K.sum(se), tf.cast(numfinite, tf.float32))
        # lossv = K.sqrt(mse)

        return lossv
    return sl1


# Huber loss, 2 predictions
def smoothLC1(hubervalue = 0.5, stdivalue = 40):
    def slc1(y_true, y_pred):
        numfinite = tf.math.count_nonzero(tf.math.is_finite(y_true))
        mask = tf.where(tf.math.is_nan(y_true), K.constant(0), K.constant(1))
        y_true = tf.math.multiply_no_nan(y_true, mask)
        y_pred = tf.math.multiply_no_nan(y_pred, mask)
        y_pred = tf.where(tf.math.is_nan(y_pred), K.constant(0), y_pred)

        # DEFINE DELTA AND AVERAGE PREDICTIONS
        HUBER_DELTA = hubervalue
        y_pred_avg_two_channels = K.expand_dims(K.mean(y_pred, axis=-1), -1)

        # DIFFERENCE BETWEEN PRED AND TRUE
        x1 = K.abs(y_true - y_pred_avg_two_channels)
        x1 = tf.where(x1 < HUBER_DELTA, 0.5 * x1 ** 2, HUBER_DELTA * (x1 - 0.5 * HUBER_DELTA))
        x1 = tf.math.divide_no_nan(K.sum(x1), tf.cast(numfinite, tf.float32))

        # DIFFERENCE BETWEEN DIFFERENT PREDICTIONS
        x2 = K.abs(y_pred[:, :, :, 0] - y_pred[:, :, :, 1])
        x2 = tf.where(x2 < HUBER_DELTA, 0.5 * x2 ** 2, HUBER_DELTA * (x2 - 0.5 * HUBER_DELTA))
        x2 = tf.math.divide_no_nan(K.sum(x2), tf.cast(numfinite, tf.float32))

        # INVERSE STANDARD DEVIATION
        x3 = 1 / (1 + tf.keras.backend.std(y_pred_avg_two_channels))
        cx3 = stdivalue
        x3 = cx3 * x3

        # Sum of all components
        sl = x1 + x2 + x3
        return sl
    return slc1


# Huber loss, adaptive loss function
def smoothLC1Adapt(aux_l1, aux_l2):
    def slc1adapt(y_true, y_pred):
        numfinite = tf.math.count_nonzero(tf.math.is_finite(y_true))
        mask = tf.where(tf.math.is_nan(y_true), K.constant(0), K.constant(1))
        y_true = tf.math.multiply_no_nan(y_true, mask)
        y_pred = tf.math.multiply_no_nan(y_pred, mask)
        y_pred = tf.where(tf.math.is_nan(y_pred), K.constant(0), y_pred)

        # Huber Loss
        y_pred_avg_two_channels = K.expand_dims(K.mean(y_pred, axis=-1), -1)

        # DIFFERENCE BETWEEN PRED AND TRUE
        x1 = aux_l1(y_true - y_pred_avg_two_channels)
        x1 = tf.math.divide_no_nan(K.sum(x1), tf.cast(numfinite, tf.float32))

        # DIFFERENCE BETWEEN DIFFERENT PREDICTIONS
        x2 = aux_l2(K.expand_dims(y_pred[:, :, :, 0], -1) - K.expand_dims(y_pred[:, :, :, 1], -1))
        x2 = tf.math.divide_no_nan(K.sum(x2), tf.cast(numfinite, tf.float32))

        # INVERSE STANDARD DEVIATION
        x3 = 1 / (1 + tf.keras.backend.std(y_pred_avg_two_channels))
        cx3 = 40
        x3 = cx3 * x3

        sl = x1 + x2 + x3
        return sl
    return slc1adapt


# Huber loss, including difference to previous iterations
def smoothLC1diffiters(hubervalue = 0.5, stdivalue = 0.01):
    def slc1diffiters(y_true, y_pred):
        y_true_act = y_true[:,:,:,0:1] # Only 0

        numfinite = tf.math.count_nonzero(tf.math.is_finite(y_true_act))
        mask = tf.where(tf.math.is_nan(y_true_act), K.constant(0), K.constant(1))
        y_true_act = tf.math.multiply_no_nan(y_true_act, mask)
        y_pred = tf.math.multiply_no_nan(y_pred, mask)
        y_pred = tf.where(tf.math.is_nan(y_pred), K.constant(0), y_pred)

        # Huber Loss
        HUBER_DELTA = hubervalue
        y_pred_avg_two_channels = K.expand_dims(K.mean(y_pred, axis=-1), -1)
        x1 = K.abs(y_true_act - y_pred_avg_two_channels)
        x1 = tf.where(x1 < HUBER_DELTA, 0.5 * x1 ** 2, HUBER_DELTA * (x1 - 0.5 * HUBER_DELTA))
        x1 = tf.math.divide_no_nan(K.sum(x1), tf.cast(numfinite, tf.float32))

        x2 = K.abs(y_pred[:, :, :, 0] - y_pred[:, :, :, 1])
        x2 = tf.where(x2 < HUBER_DELTA, 0.5 * x2 ** 2, HUBER_DELTA * (x2 - 0.5 * HUBER_DELTA))
        x2 = tf.math.divide_no_nan(K.sum(x2), tf.cast(numfinite, tf.float32))

        y_true_prev = y_true[:,:,:,1:2] # Only 0 and 1
        mask2 = tf.where(tf.math.is_nan(y_true_prev), K.constant(0), K.constant(1))
        y_true_prev = tf.math.multiply_no_nan(y_true_prev, mask)
        y_true_prev = tf.math.multiply_no_nan(y_true_prev, mask2)
        x3 = K.abs(y_true_prev - y_pred_avg_two_channels)
        x3 = tf.where(x3 < HUBER_DELTA, 0.5 * x3 ** 2, HUBER_DELTA * (x3 - 0.5 * HUBER_DELTA))
        x3 = tf.math.divide_no_nan(K.sum(x3), tf.cast(numfinite, tf.float32))

        x4 = 1 / (1 + tf.keras.backend.std(y_pred_avg_two_channels))
        cx4 = 40
        x4 = cx4 * x4

        sl = x1 + x2 + x3 + x4
        return sl
    return slc1diffiters


# Smooth loss multiple predictions, first part
def smoothLC1a(hubervalue = 0.5, stdivalue = 0.01):
    def slc1a(y_true, y_pred):
        y_true = y_true[:,:,:,0:1] # Only 0
        y_pred = y_pred[:,:,:,0:2] # Only 0 and 1
        numfinite = tf.math.count_nonzero(tf.math.is_finite(y_true))
        mask = tf.where(tf.math.is_nan(y_true), K.constant(0), K.constant(1))
        y_true = tf.math.multiply_no_nan(y_true, mask)
        y_pred = tf.math.multiply_no_nan(y_pred, mask)
        y_pred = tf.where(tf.math.is_nan(y_pred), K.constant(0), y_pred)

        # Huber Loss
        HUBER_DELTA = hubervalue
        y_pred_avg_two_channels = K.expand_dims(K.mean(y_pred, axis=-1), -1)
        x1 = K.abs(y_true - y_pred_avg_two_channels)
        x1 = tf.where(x1 < HUBER_DELTA, 0.5 * x1 ** 2, HUBER_DELTA * (x1 - 0.5 * HUBER_DELTA))
        sl = K.sum(x1)

        return tf.math.divide_no_nan(sl, tf.cast(numfinite, tf.float32))
    return slc1a


# Smooth loss multiple predictions, second part
def smoothLC1b(hubervalue = 0.5, stdivalue = 0.01):
    def slc1b(y_true, y_pred):
        y_true = y_true[:,:,:,0:1] # Only 0
        y_pred = y_pred[:,:,:,0:2] # Only 0 and 1
        numfinite = tf.math.count_nonzero(tf.math.is_finite(y_true))
        mask = tf.where(tf.math.is_nan(y_true), K.constant(0), K.constant(1))
        y_pred = tf.math.multiply_no_nan(y_pred, mask)
        y_pred = tf.where(tf.math.is_nan(y_pred), K.constant(0), y_pred)

        # Huber Loss
        HUBER_DELTA = hubervalue
        x2 = K.abs(y_pred[:, :, :, 0] - y_pred[:, :, :, 1])
        x2 = tf.where(x2 < HUBER_DELTA, 0.5 * x2 ** 2, HUBER_DELTA * (x2 - 0.5 * HUBER_DELTA))
        sl = K.sum(x2)

        return tf.math.divide_no_nan(sl, tf.cast(numfinite, tf.float32))
    return slc1b


# Smooth loss multiple predictions, third part
def smoothLC1c(hubervalue = 0.5, stdivalue = 0.01):
    def slc1c(y_true, y_pred):
        y_true = y_true[:,:,:,0:1] # Only 0
        y_pred = y_pred[:,:,:,0:2] # Only 0 and 1
        numfinite = tf.math.count_nonzero(tf.math.is_finite(y_true))
        mask = tf.where(tf.math.is_nan(y_true), K.constant(0), K.constant(1))
        y_pred = tf.math.multiply_no_nan(y_pred, mask)
        y_pred = tf.where(tf.math.is_nan(y_pred), K.constant(0), y_pred)

        # Huber Loss
        y_pred_avg_two_channels = K.expand_dims(K.mean(y_pred, axis=-1), -1)
        sl = 1 / (1 + tf.keras.backend.std(y_pred_avg_two_channels))
        csl = 40
        sl = csl * sl

        return tf.math.divide_no_nan(sl, tf.cast(numfinite, tf.float32))
    return slc1c


def mean_squared_error_geary():
    def msegeary(y_true, y_pred):
        return K.sqrt(K.mean(K.square((y_true[:,:,:,1] - y_pred[:,:,:,2]))))
    return msegeary


def unet(inputs, filters=[2,4,8,16,32], dropout=0.5, dsc=0):
    conv00 = unet_convblock(filters[0], inputs, dsc, downup=0)  # X00

    pool10 = unet_downsampling(conv00)
    conv10 = unet_convblock(filters[1], pool10, dsc, downup=0)  # X10

    pool20 = unet_downsampling(conv10)
    conv20 = unet_convblock(filters[2], pool20, dsc, downup=0)  # X20

    pool30 = unet_downsampling(conv20)
    conv30 = unet_convblock(filters[3], pool30, dsc, downup=0)  # X30
    drop30 = Dropout(dropout)(conv30)

    pool40 = unet_downsampling(drop30)
    conv40 = unet_convblock(filters[4], pool40)  # X40
    drop40 = Dropout(dropout)(conv40)

    up40 = unet_upsampling(filters[3], drop40)
    merge31 = concatenate([drop30, up40], axis=3)
    conv31 = unet_convblock(filters[3], merge31, dsc, downup=1)  # X31

    up31 = unet_upsampling(filters[2], conv31)
    merge22 = concatenate([conv20, up31], axis=3)
    conv22 = unet_convblock(filters[2], merge22, dsc, downup=1)  # X22

    up22 = unet_upsampling(filters[1], conv22)
    merge13 = concatenate([conv10, up22], axis=3)
    conv13 = unet_convblock(filters[1], merge13, dsc, downup=1)  # X13

    up13 = unet_upsampling(filters[0], conv13)
    merge04 = concatenate([conv00, up13], axis=3)
    conv04 = unet_convblock(filters[0], merge04, dsc, downup=1)  # X04

    # New
    aux = concatenate([conv04, inputs], axis=3)
    # aux = conv04

    output = Conv2D(1, 1, activation='linear')(aux)

    return output


def squeeze_excite_block(tensor, ratio=16):
    filters = tensor.shape[-1]
    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D()(tensor)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = multiply([tensor, se])
    return x


def unet_convblock(filters, inputs, dsc=0, downup=0):
    nfilters2ndconv = 2*filters if downup == 0 else filters/2
    if dsc == 0:
        conv = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    elif dsc == 1:
        conv = Conv2D(filters, 1, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv = SeparableConv2D(4 * filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
        conv = squeeze_excite_block(conv)
        conv = Conv2D(filters, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
        conv = concatenate([conv, inputs], axis=3)
        conv = Conv2D(nfilters2ndconv, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    elif dsc == 2:
        conv = Conv2D(filters, 1, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv = SeparableConv2D(4*filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
        conv = squeeze_excite_block(conv)
        conv = Conv2D(nfilters2ndconv, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    else:
        print("Error - Conv Block")
    return conv


def unet_downsampling(inputs):
    down = MaxPooling2D(pool_size=(2, 2))(inputs)
    return down


def unet_upsampling(filters, inputs):
    up = Conv2D(filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(inputs))
    return up


def unetplusplus(inputs, filters=[2,4,8,16,32], dropout=0.5, dsc=0):
    conv00 = unet_convblock(filters[0], inputs, dsc, downup=0) # X00

    pool10 = unet_downsampling(conv00)
    conv10 = unet_convblock(filters[1], pool10, dsc, downup=0) # X10

    pool20 = unet_downsampling(conv10)
    conv20 = unet_convblock(filters[2], pool20, dsc, downup=0) # X20

    pool30 = unet_downsampling(conv20)
    conv30 = unet_convblock(filters[3], pool30, dsc, downup=0) # X30
    drop30 = Dropout(dropout)(conv30)

    pool40 = unet_downsampling(drop30)
    conv40 = unet_convblock(filters[4], pool40) # X40
    drop40 = Dropout(dropout)(conv40)

    up10 = unet_upsampling(filters[0], conv10)
    merge01 = concatenate([conv00, up10], axis=3)
    conv01 = unet_convblock(filters[0], merge01, dsc, downup=1) # X01

    up20 = unet_upsampling(filters[1], conv20)
    merge11 = concatenate([conv10, up20], axis=3)
    conv11 = unet_convblock(filters[1], merge11, dsc, downup=1) # X11

    up30 = unet_upsampling(filters[2], conv30)
    merge21 = concatenate([conv20, up30], axis=3)
    conv21 = unet_convblock(filters[2], merge21, dsc, downup=1) # X21

    up40 = unet_upsampling(filters[3], drop40)
    merge31 = concatenate([drop30, up40], axis=3)
    conv31 = unet_convblock(filters[3], merge31, dsc, downup=1) # X31

    up11 = unet_upsampling(filters[0], conv11)
    merge02 = concatenate([conv01, up11, conv00], axis=3)
    conv02 = unet_convblock(filters[0], merge02, dsc, downup=1) # X02

    up21 = unet_upsampling(filters[1], conv21)
    merge12 = concatenate([conv11, up21, conv10], axis=3)
    conv12 = unet_convblock(filters[1], merge12, dsc, downup=1) # X12

    up31 = unet_upsampling(filters[2], conv31)
    merge22 = concatenate([conv21, up31, conv20], axis=3)
    conv22 = unet_convblock(filters[2], merge22, dsc, downup=1) # X22

    up12 = unet_upsampling(filters[0], conv12)
    merge03 = concatenate([conv02, up12, conv01, conv00], axis=3)
    conv03 = unet_convblock(filters[0], merge03, dsc, downup=1) # X03

    up22 = unet_upsampling(filters[1], conv22)
    merge13 = concatenate([conv12, up22, conv11, conv10], axis=3)
    conv13 = unet_convblock(filters[1], merge13, dsc, downup=1) # X13

    up13 = unet_upsampling(filters[0], conv13)
    merge04 = concatenate([conv03, up13, conv02, conv01, conv00], axis=3)
    conv04 = unet_convblock(filters[0], merge04, dsc, downup=1) # X04


    # New
    aux = concatenate([conv04, conv03, conv02, conv01, conv00, inputs], axis=3)
    # aux = concatenate([conv04, conv03, conv02, conv01, conv00], axis=3)

    output = Conv2D(1, 1, activation='linear')(aux)

    return output


def compilecnnmodel(cnnmod, shape, lrate, dropout=0.5, filters=[8, 16, 32, 64, 128], lweights=[0.5, 0.5],
                    hubervalue=1, stdivalue=40, previters=False, lossparams=[2,0,0,False]):
    tf.random.set_seed(SEED)

    if cnnmod == 'cnnlm':
        shape = [7]
        mod = Sequential()
        mod.add(Dense(units=1, input_shape=shape))
        mod.add(Activation('linear'))
        mod.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=lrate))

    elif cnnmod == 'lenet':
        mod = Sequential()
        mod.add(Conv2D(filters=filters[0], kernel_size=3, padding='same', input_shape=shape, activation='relu'))
        mod.add(MaxPooling2D(pool_size=(2, 2)))
        mod.add(Conv2D(filters=filters[1], kernel_size=3, padding='same'))
        mod.add(Activation('relu'))
        mod.add(MaxPooling2D(pool_size=(2, 2)))
        mod.add(Dropout(rate=0.1))
        mod.add(Flatten())
        mod.add(Dense(units=filters[2]))
        mod.add(Activation('relu'))
        mod.add(Dense(units=1))
        mod.add(Activation('linear'))
        mod.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=lrate))

    elif cnnmod == 'vgg':
        mod = Sequential()
        mod.add(Conv2D(filters[0], 3, activation='relu', padding='same', input_shape=shape, name='block1_conv1'))
        mod.add(Conv2D(filters[0], 3, activation='relu', padding='same', name='block1_conv2'))
        mod.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
        mod.add(Conv2D(filters[1], 3, activation='relu', padding='same', name='block2_conv1'))
        mod.add(Conv2D(filters[1], 3, activation='relu', padding='same', name='block2_conv2'))
        mod.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
        mod.add(Conv2D(filters[2], 3, activation='relu', padding='same', name='block3_conv1'))
        mod.add(Conv2D(filters[2], 3, activation='relu', padding='same', name='block3_conv2'))
        mod.add(Conv2D(filters[2], 3, activation='relu', padding='same', name='block3_conv3'))
        mod.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block4_conv1'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block4_conv2'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block4_conv3'))
        mod.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block5_conv1'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block5_conv2'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block5_conv3'))
        mod.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
        mod.add(Flatten(name='flatten'))
        mod.add(Dense(filters[4], activation='relu', name='fc1'))
        mod.add(Dense(filters[4], activation='relu', name='fc2'))
        mod.add(Dense(units=1, activation='linear', name='predictions'))
        mod.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=lrate))

    elif cnnmod == 'uenc':
        inputs = Input(shape)
        conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
        flat1 = Flatten()(drop5)
        dens1 = Dense(units=1, activation='linear', name='predictions')(flat1)
        mod = Model(inputs=inputs, outputs=dens1)
        mod.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=lrate))

    elif cnnmod == 'unet' or cnnmod == 'unetplusplus':
        inputs = Input(shape)

        # 1: Standard U-Net
        # 2: Original prediction + prediction derived from rotated input
        # 3: Original + Rotated + Geary
        npredictions = lossparams[0]
        usegradnorm = lossparams[1] # 0 (no gradnorm), 1 (with gradnorm)
        dsc = lossparams[2] # Type of encoder in U-Net++ (0/1/2)
        adaptive = lossparams[3]


        if npredictions >= 2:
            # Random transformation to apply
            randomint = K.constant(random.randint(0, 5))

            def t0(inputs): return Lambda(lambda x: K.reverse(x, axes=1))(inputs) # Horizontal flip
            def t1(inputs): return Lambda(lambda x: K.reverse(x, axes=2))(inputs) # Vertical flip
            def t2(inputs): return Lambda(lambda x: K.reverse(x, axes=(1,2)))(inputs) # Horizontal and vertical flip
            def t3(inputs): return Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(inputs) # Transpose
            def t4(inputs): return Lambda(lambda x: K.reverse(x, axes=1))(t3(inputs))  # Rotate clockwise
            def t5(inputs): return Lambda(lambda x: K.reverse(x, axes=2))(t3(inputs))  # Rotate counter clockwise
            input_b = tf.case([(tf.equal(randomint, K.constant(0)), lambda: t0(inputs)),
                               (tf.equal(randomint, K.constant(1)), lambda: t1(inputs)),
                               (tf.equal(randomint, K.constant(2)), lambda: t2(inputs)),
                               (tf.equal(randomint, K.constant(3)), lambda: t3(inputs)),
                               (tf.equal(randomint, K.constant(4)), lambda: t4(inputs)),
                               (tf.equal(randomint, K.constant(5)), lambda: t4(inputs))],
                              exclusive=True)

            # Contrastive
            if cnnmod == 'unet':
                processed_a = unet(inputs, filters, dropout, dsc)
                processed_b = unet(input_b, filters, dropout, dsc)
            elif cnnmod == 'unetplusplus':
                processed_a = unetplusplus(inputs, filters, dropout, dsc)
                processed_b = unetplusplus(input_b, filters, dropout, dsc)

            processed_b = tf.case([(tf.equal(randomint, K.constant(0)), lambda: t0(processed_b)),
                                   (tf.equal(randomint, K.constant(1)), lambda: t1(processed_b)),
                                   (tf.equal(randomint, K.constant(2)), lambda: t2(processed_b)),
                                   (tf.equal(randomint, K.constant(3)), lambda: t3(processed_b)),
                                   (tf.equal(randomint, K.constant(4)), lambda: t5(processed_b)),
                                   (tf.equal(randomint, K.constant(5)), lambda: t4(processed_b))],
                                 exclusive=True)

        if npredictions == 1:
            if cnnmod == 'unet':
                result = unet(inputs, filters, dropout)
            elif cnnmod == 'unetplusplus':
                result = unetplusplus(inputs, filters, dropout)

        elif npredictions == 2:
            result = Concatenate()([processed_a, processed_b])

        elif npredictions == 3:
            gearyout1 = Conv2D(1, 3, activation='linear', padding='same')(processed_a)
            gearyout2 = Conv2D(1, 3, activation='linear', padding='same')(processed_b)
            gearyout = Average()([gearyout1, gearyout2])
            result = Concatenate()([processed_a, processed_b, gearyout])

        else:
            print("Error - Compute model result")


        if usegradnorm == 0:
            if npredictions == 1:
                mod = Model(inputs=inputs, outputs=result)
                sl1 = smoothL1(hubervalue=hubervalue, stdivalue=stdivalue)
                mod.compile(loss=sl1, optimizer=optimizers.Adam(learning_rate=lrate))

            elif npredictions == 2:
                if adaptive: # Without previous iterations
                    mod = CustomModelAdaptive(inputs=inputs, outputs=result)
                    aux_l1 = aloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
                    aux_l2 = aloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
                    slc1adapt = smoothLC1Adapt(aux_l1, aux_l2)
                    mod.compile(loss=slc1adapt, optimizer=optimizers.Adam(learning_rate=lrate),
                                run_eagerly=False)

                else:
                    mod = Model(inputs=inputs, outputs=result)
                    if previters:
                        slc1diffiters = smoothLC1diffiters(hubervalue=hubervalue, stdivalue=stdivalue)
                        mod.compile(loss=slc1diffiters, optimizer=optimizers.Adam(learning_rate=lrate))
                    else:
                        slc1 = smoothLC1(hubervalue=hubervalue, stdivalue=stdivalue)
                        mod.compile(loss=slc1, optimizer=optimizers.Adam(learning_rate=lrate))

        elif usegradnorm == 1:
            if npredictions >= 2:
                slc1a = smoothLC1a(hubervalue=hubervalue, stdivalue=stdivalue)
                slc1b = smoothLC1b(hubervalue=hubervalue, stdivalue=stdivalue)
                slc1c = smoothLC1c(hubervalue=hubervalue, stdivalue=stdivalue)

            if npredictions == 2:
                mod = CustomModel(inputs=inputs, outputs=result,
                                  hubervalue=hubervalue, stdivalue=stdivalue,
                                  alpha=1, learning_rate=lrate, ntasks = 3)

                mod.compile(loss=[slc1a, slc1b, slc1c],
                            loss_weights=[1, 1, 1],
                            optimizer=optimizers.Adam(learning_rate=lrate),
                            run_eagerly=False)

            elif npredictions == 3:
                mod = CustomModel(inputs=inputs, outputs=result,
                                  hubervalue=hubervalue, stdivalue=stdivalue,
                                  alpha=1, learning_rate=lrate, ntasks=4)

                mseg = mean_squared_error_geary()
                mod.compile(loss=[slc1a, slc1b, slc1c, mseg],
                            loss_weights=[1, 1, 1, 1],
                            optimizer=optimizers.Adam(learning_rate=lrate),
                            run_eagerly=False)

        else:
            print("Error w/ usegradnorm")


    elif cnnmod == '2runet':
        inputs = Input(shape)
        conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(filters[3], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(filters[2], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(filters[1], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(filters[0], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)


        # High resolution output
        outputhr = Conv2D(1, 1, activation='linear', name="highres")(conv9)

        # Sum of high resolution output
        avgpoolinghr = AveragePooling2D(pool_size=4)(outputhr)
        outputlr = Lambda(lambda x: x * 4, name="lowres")(avgpoolinghr)

        mod = Model(inputs=inputs, outputs=[outputhr, outputlr])
        mod.compile(loss=['mean_squared_error', 'mean_squared_error'],
                    loss_weights=lweights,
                    optimizer=optimizers.Adam(lr=lrate))

    return mod


def savemodel(model, path, includeoptimizer):
    return model.save(path, include_optimizer=includeoptimizer)


def loadmodel(path, hubervalue, stdivalue):
    adapt_l1 = aloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
    adapt_l2 = aloss.AdaptiveLossFunction(num_channels=1, float_dtype=np.float32)
    return tf.keras.models.load_model(path,  custom_objects={'sl1': smoothL1(hubervalue, stdivalue),
                                                                'slc1': smoothLC1(hubervalue, stdivalue),
                                                                'slc1a': smoothLC1a(hubervalue, stdivalue),
                                                                'slc1b': smoothLC1b(hubervalue, stdivalue),
                                                                'slc1c': smoothLC1c(hubervalue, stdivalue),
                                                                'msegeary': mean_squared_error_geary(),
                                                                'slc1diffiters': smoothLC1diffiters(hubervalue, stdivalue),
                                                                'slc1adapt': smoothLC1Adapt(adapt_l1, adapt_l2)})


def createpatches(X, patchsize, padding, stride=1, cstudy=None):
    if cstudy:
        try:
            fp = np.memmap(cstudy + '.dat', mode='r')
            print('Found .dat file')
            ninstances = int(fp.shape[0] / patchsize / patchsize / X.shape[2] / 4) # Divide by dimensions
            shapemmap = (ninstances, patchsize, patchsize, X.shape[2])
            fp = np.memmap(cstudy + '.dat', dtype='float32', mode='r', shape=shapemmap)
        except:
            print('Did not find .dat file')
            if padding:
                rowpad = int((patchsize - 1) / 2)
                colpad = int(round((patchsize - 1) / 2))
                newX = np.pad(X, ((rowpad, colpad), (rowpad, colpad), (0, 0)), 'constant', constant_values=(0, 0))
            else:
                newX = X
            newX[np.isnan(newX)] = -9999999
            patches = extract_patches_2d(newX, [16, 16])
            patches[patches == -9999999] = np.nan
            fp = np.memmap(cstudy + '.dat', dtype='float32', mode='w+', shape=patches.shape)
            fp[:] = patches[:]
            fp = fp.reshape(-1, patchsize, patchsize, X.shape[2])
        return fp
    else:
        if padding:
            rowpad = int((patchsize - 1) / 2)
            colpad = int(round((patchsize - 1) / 2))
            newX = np.pad(X, ((rowpad, colpad), (rowpad, colpad), (0, 0)), 'constant', constant_values=(0, 0))
        else:
            newX = X

        newX[np.isnan(newX)] = -9999999
        patches = extract_patches_2d(newX, [16, 16])
        patches[patches == -9999999] = np.nan
        patches = patches.reshape(-1, patchsize, patchsize, X.shape[2])
        return patches


def reconstructpatches(patches, image_size, stride):
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    mean = np.zeros(image_size)
    patch_count = np.zeros(image_size)
    n_h = int((i_h - p_h) / stride + 1)
    n_w = int((i_w - p_w) / stride + 1)
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        patch_count[i * stride:i * stride + p_h, j * stride:j * stride + p_w] += ~np.isnan(p)
        ctignore = np.isnan(p)
        p[ctignore] = 0
        mean[i * stride:i * stride + p_h, j * stride:j * stride + p_w] += p
        p[ctignore] = np.nan
    mean = np.divide(mean, patch_count, out=np.zeros_like(mean), where=patch_count != 0)

    return mean
