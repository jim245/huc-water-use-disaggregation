"""
CNN/random-forest training, prediction, and U-Net model utilities.
"""

import gc
import math
import os
from concurrent import futures
from typing import Any, Optional, Self

import numpy
import tensorflow
from keras import callbacks, layers, models, ops, optimizers, saving, utils
from numpy import random
from numpy.typing import NDArray
from sklearn import ensemble

# seeds TensorFlow (fit_cnn) and the random forest (fit_rf); numpy's global RNG is seeded in disaggregation.py
SEED = 42  # e.g. the answer to life, the universe, and everything


def predict(
    mod: ensemble.RandomForestRegressor,
    X: NDArray[numpy.number[Any]],
    cell_mask: Optional[NDArray[numpy.bool]] = None,
) -> list[NDArray[numpy.floating[Any]]]:
    """
    Predicts a raster of values from a fitted random forest model.

    Parameters:
        mod (ensemble.RandomForestRegressor): The fitted random forest model.
        X (NDArray[numpy.number[Any]]): The ancillary raster stack, shaped (height, width, channels).
        cell_mask (Optional[NDArray[numpy.bool]] = None): The cells to predict, shaped (height, width); other cells
            are NaN. All cells when None.

    Returns:
        predicted_maps (list[NDArray[numpy.floating[Any]]]): A single-element list containing the predicted raster,
            shaped (height, width).
    """

    newX = X.reshape((-1, X.shape[2]))
    if cell_mask is None:
        pred = predict_rows_deterministic(mod, newX)
    else:
        # each cell is predicted independently, so skipping cells doesn't change the others
        kept = cell_mask.ravel()
        pred = numpy.full(newX.shape[0], numpy.nan)
        pred[kept] = predict_rows_deterministic(mod, newX[kept])
    pred = pred.reshape(X.shape[0], X.shape[1])

    return [pred]


def predict_rows_deterministic(
    mod: ensemble.RandomForestRegressor, X: NDArray[numpy.number[Any]]
) -> NDArray[numpy.float64]:
    """
    Predicts rows with a random forest in parallel while keeping results bit-reproducible.

    The forest's own parallel predict (n_jobs != 1) adds tree outputs in thread completion order, so results
    vary in the last bits between runs. Instead, rows are split into chunks predicted on separate threads,
    and every row sums its trees in the same fixed order.

    Each chunk does what RandomForestRegressor.predict does with n_jobs=1 (add each tree's prediction in
    estimator order, then divide by the number of trees), so the result is bit-identical to it. The trees are
    called directly because sklearn's Parallel, nested inside threads, races on the global warning filters.

    Parameters:
        mod (ensemble.RandomForestRegressor): The fitted random forest model.
        X (NDArray[numpy.number[Any]]): The rows to predict, shaped (n_rows, n_features).

    Returns:
        pred (NDArray[numpy.float64]): The predictions, shaped (n_rows,).
    """

    thread_count = min(os.cpu_count() or 1, max(1, len(X)))
    # the forest's own input check (float32 conversion, NaN allowed), done once for all chunks
    X = mod._validate_X_predict(X)

    def predict_chunk(chunk: NDArray[numpy.float32]) -> NDArray[numpy.float64]:
        out = numpy.zeros(len(chunk), dtype=numpy.float64)
        for tree in mod.estimators_:
            out += tree.predict(chunk, check_input=False)
        out /= len(mod.estimators_)
        return out

    # a plain thread pool: tree prediction releases the GIL, and no sklearn Parallel runs inside the threads
    with futures.ThreadPoolExecutor(thread_count) as pool:
        chunk_predictions = list(
            pool.map(predict_chunk, numpy.array_split(X, thread_count))
        )

    return numpy.concatenate(chunk_predictions)


class DataGenerator(utils.Sequence):
    """
    Keras data generator that serves batches of ancillary/target patch pairs
    by index, zeroing NaNs in the ancillary patches of each batch as it is served.

    Parameters:
        Xdataset (PatchView): The ancillary variable patches, shaped (n_patches, patch_size, patch_size, channels).
        ydataset (PatchView): The target patches, shaped (n_patches, patch_size, patch_size, 1).
        idsamples (NDArray[numpy.intp]): The patch indices to draw batches from.
        batch_size (int): The number of patches per batch.
    """

    def __init__(
        self: Self,
        Xdataset: "PatchView",
        ydataset: "PatchView",
        idsamples: NDArray[numpy.intp],
        batch_size: int,
    ):
        # keras.utils.PyDataset defaults (single worker, no multiprocessing)
        super().__init__()
        self.X, self.y = Xdataset, ydataset
        self.batch_size = batch_size
        self.idsamples = idsamples

    def __len__(self: Self) -> int:
        """
        Reports the number of batches per epoch.

        Returns:
            n_batches (int): The number of batches per epoch.
        """
        return math.ceil(len(self.idsamples) / self.batch_size)

    def __getitem__(
        self: Self, idx: int
    ) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
        """
        Builds a single batch, zeroing NaNs in the ancillary patches.

        Parameters:
            idx (int): The batch index.

        Returns:
            batch (tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]): The (X, y) patch batch.
        """
        idsamplesbatch = self.idsamples[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_X = self.X[idsamplesbatch]
        batch_X[numpy.isnan(batch_X)] = 0
        batch_y = self.y[idsamplesbatch]
        return numpy.array(batch_X), numpy.array(batch_y)

    def on_epoch_end(self: Self):
        # force garbage collection between epochs to limit memory growth
        # (no backend.clear_session() here: it flushed TensorFlow's kernel cache every epoch, costing ~1 s per
        # epoch without changing results, and Keras global state only grows when models are created)
        gc.collect()


class LossHistory(callbacks.Callback):
    """
    Keras training callback that records the loss reported at the end of each epoch.
    """

    def on_train_begin(self: Self, logs: dict[str, Any] = dict()):
        """
        Resets the recorded loss history.

        Parameters:
            logs (dict[str, Any]): Unused; required by the Keras callback interface.
        """
        self.losses: list[Any] = []

    def on_epoch_end(self: Self, batch: int, logs: dict[str, Any] = dict()):
        """
        Appends the epoch's loss to the recorded history.

        Parameters:
            batch (int): Unused; required by the Keras callback interface.
            logs (dict[str, Any]): The epoch's logs; expected to contain a "loss" key.
        """
        self.losses.append(logs.get("loss"))


def fit_cnn(
    X: "PatchView",
    y: "PatchView",
    p: list[float],
    cnnobj: models.Model,
    epochs: int,
    batch_size: int,
) -> list[float]:
    """
    Fits the compiled CNN model on patches whose middle pixel has a known
    (non-NaN) target value, sampling a random subset for training.

    Parameters:
        X (PatchView): The ancillary variable patches, shaped (n_patches, patch_size, patch_size, channels).
        y (PatchView): The target patches, shaped (n_patches, patch_size, patch_size, 1).
        p (list[float]): The proportion of eligible patches to sample for training; only p[0] is used.
        cnnobj (models.Model): The compiled CNN model to fit.
        epochs (int): The number of epochs to fit for.
        batch_size (int): The number of patches per training batch.

    Returns:
        hislist (list[float]): A single-element list containing the final epoch's training loss.
    """

    tensorflow.random.set_seed(SEED)

    # "middle" pixel of each patch: rows truncate (int) while columns round half to even (round), so for an even
    # patch size (no single middle) this is (patch_size / 2 - 1, patch_size / 2), e.g. (7, 8) for 16 and (3, 4) for 8
    middle_row, middle_column = int((y.shape[1] - 1) / 2), int(
        round((y.shape[2] - 1) / 2)
    )

    # Train only with patches having middle pixel different from NaN
    relevantids = numpy.where(~numpy.isnan(y.pixel(middle_row, middle_column, 0)))[0]

    relevsamples = random.choice(
        len(relevantids), round(len(relevantids) * p[0]), replace=False
    )

    idsamples = relevantids[relevsamples]

    print(
        "Number of patches (all, valid middle pixel, sampled):",
        y.shape[0],
        len(relevantids),
        len(idsamples),
    )

    print("| --- Fit - 1 resolution U-Net Model", epochs)

    ## Training generator
    lhistory = LossHistory()
    training_generator = DataGenerator(X, y, idsamples, batch_size=batch_size)
    # verbose=2: one line per epoch instead of progress bars, which fill log files with escape codes
    cnnobj.fit(training_generator, epochs=epochs, callbacks=lhistory, verbose=2)
    hislist: list[float] = [lhistory.losses[-1]]

    return hislist


def load_model(path: str, huber_value: float, smoothness_weight: float) -> models.Model:
    """
    Loads a saved CNN model, registering the custom loss function (smoothLC1) needed to deserialize it.

    Parameters:
        path (str): The path to the saved Keras model file.
        huber_value (float): The Huber loss delta used by smoothLC1.
        smoothness_weight (float): The weight of smoothLC1's inverse-standard-deviation term.

    Returns:
        model (models.Model): The loaded Keras model.
    """

    return models.load_model(
        path,
        {"slc1": smoothLC1(huber_value, smoothness_weight)},
        safe_mode=False,
    )


class PatchView:
    """
    Overlapping square patches of a raster stack, served from a sliding-window view of the raster instead of
    copied, so no patch array is ever built (a full stride-1 patch array holds each cell up to patch_size² times).

    Patches are numbered in row-major order of their top-left corners, as in sklearn's extract_patches_2d: patch k
    has its top-left corner at (k // n_w, k % n_w). NaNs are kept as NaN. The raster must not be modified while the
    view is in use.

    Parameters:
        raster (NDArray): The raster stack, shaped (height, width, channels).
        patch_size (int): The side length of the square patches.
    """

    def __init__(self: Self, raster: NDArray, patch_size: int):
        self.raster = raster
        # shaped (n_h, n_w, channels, patch_size, patch_size)
        self.windows = numpy.lib.stride_tricks.sliding_window_view(
            raster, (patch_size, patch_size), axis=(0, 1)
        )
        self.n_h, self.n_w = self.windows.shape[:2]
        self.shape = (self.n_h * self.n_w, patch_size, patch_size, raster.shape[2])
        self.dtype = raster.dtype

    def __len__(self: Self) -> int:
        return self.shape[0]

    def __getitem__(self: Self, indices: NDArray[numpy.intp]) -> NDArray:
        """
        Copies out the requested patches.

        Parameters:
            indices (NDArray[numpy.intp]): The patch numbers to return.

        Returns:
            patches (NDArray): The patches, shaped (len(indices), patch_size, patch_size, channels).
        """
        rows, columns = numpy.divmod(numpy.asarray(indices), self.n_w)
        return numpy.ascontiguousarray(
            numpy.moveaxis(self.windows[rows, columns], 1, -1)
        )

    def pixel(self: Self, row: int, column: int, channel: int = 0) -> NDArray:
        """
        Returns the value at one position within every patch, without building the patches.

        Parameters:
            row (int): The row within the patch.
            column (int): The column within the patch.
            channel (int = 0): The channel.

        Returns:
            values (NDArray): The value in each patch, shaped (n_patches,).
        """
        return self.raster[
            row : row + self.n_h, column : column + self.n_w, channel
        ].ravel()


@saving.register_keras_serializable()
def horizontal_flip(x: tensorflow.Tensor) -> tensorflow.Tensor:
    """
    Reverses a batched image tensor along axis 2 (the width/column axis),
    i.e. flips column order left-to-right.

    Parameters:
        x (tensorflow.Tensor): A batched image tensor, shaped (batch, height, width, channels).

    Returns:
        flipped (tensorflow.Tensor): The tensor reversed along axis 2.
    """
    return tensorflow.reverse(x, axis=[2])


@saving.register_keras_serializable()
def vertical_flip(x: tensorflow.Tensor) -> tensorflow.Tensor:
    """
    Reverses a batched image tensor along axis 1 (the height/row axis),
    i.e. flips row order top-to-bottom.

    Parameters:
        x (tensorflow.Tensor): A batched image tensor, shaped (batch, height, width, channels).

    Returns:
        flipped (tensorflow.Tensor): The tensor reversed along axis 1.
    """
    return tensorflow.reverse(x, axis=[1])


@saving.register_keras_serializable()
def hv_flip(x: tensorflow.Tensor) -> tensorflow.Tensor:
    """
    Reverses a batched image tensor along both the height and width axes.

    Parameters:
        x (tensorflow.Tensor): A batched image tensor, shaped (batch, height, width, channels).

    Returns:
        flipped (tensorflow.Tensor): The tensor reversed along axes 1 and 2.
    """
    return tensorflow.reverse(x, axis=[1, 2])


@saving.register_keras_serializable()
def transpose(x: tensorflow.Tensor) -> tensorflow.Tensor:
    """
    Swaps the height and width axes of a batched image tensor.

    Parameters:
        x (tensorflow.Tensor): A batched image tensor, shaped (batch, height, width, channels).

    Returns:
        transposed (tensorflow.Tensor): The tensor with axes 1 and 2 swapped.
    """
    return tensorflow.transpose(x, perm=[0, 2, 1, 3])


@saving.register_keras_serializable()
def rotate_cw(x: tensorflow.Tensor) -> tensorflow.Tensor:
    """
    Rotates a batched square image tensor 90 degrees clockwise by
    transposing the height/width axes and then reversing axis 2.

    Parameters:
        x (tensorflow.Tensor): A batched square image tensor, shaped (batch, size, size, channels).

    Returns:
        rotated (tensorflow.Tensor): The rotated tensor.
    """
    return tensorflow.reverse(tensorflow.transpose(x, perm=[0, 2, 1, 3]), axis=[2])


@saving.register_keras_serializable()
def rotate_ccw(x: tensorflow.Tensor) -> tensorflow.Tensor:
    """
    Rotates a batched square image tensor 90 degrees counterclockwise by
    transposing the height/width axes and then reversing axis 1. The inverse
    of rotate_cw.

    Parameters:
        x (tensorflow.Tensor): A batched square image tensor, shaped (batch, size, size, channels).

    Returns:
        rotated (tensorflow.Tensor): The rotated tensor.
    """
    return tensorflow.reverse(tensorflow.transpose(x, perm=[0, 2, 1, 3]), axis=[1])


def squeeze_excite_block(tensor: Any, ratio: int = 16) -> Any:
    """
    Applies a squeeze-and-excitation block: pools the input to a per-channel
    descriptor, learns per-channel gates from it, and rescales the input by
    those gates.

    Parameters:
        tensor (Any): The input Keras functional-API tensor.
        ratio (int = 16): The channel reduction ratio for the squeeze step.

    Returns:
        x (Any): The input tensor rescaled by the learned channel gates.
    """
    filters = tensor.shape[-1]
    se_shape = (1, 1, filters)
    se = layers.GlobalAveragePooling2D()(tensor)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(
        filters // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=False,
    )(se)
    se = layers.Dense(
        filters, activation="sigmoid", kernel_initializer="he_normal", use_bias=False
    )(se)
    x = layers.multiply([tensor, se])
    return x


def unet_convblock(filters: int, inputs: Any, dsc: int = 0, downup: int = 0) -> Any:
    """
    Builds one U-Net convolutional block.

    Parameters:
        filters (int): The base number of filters for the block.
        inputs (Any): The input Keras functional-API tensor.
        dsc (int = 0): The convolutional block style: 0 (plain double 3x3 conv, ignores downup), or 1 (1x1 conv,
            separable 3x3 conv, squeeze-excite, 1x1 conv, concatenation with the input, then a 3x3 conv sized by
            downup). Production uses 1 (slightly better HUC-12 placement than 0 in holdout tests); a third style (2) was
            tested, didn't help, and was removed (FIXES.md, fix 33).
        downup (int = 0): For dsc 1, sizes the block's final conv at 2x filters when 0 (downsampling path) or
            filters // 2 otherwise (upsampling path). Ignored when dsc == 0.

    Returns:
        conv (Any): The block's output tensor.
    """
    nfilters2ndconv = 2 * filters if downup == 0 else filters // 2
    if dsc == 0:
        conv = layers.Conv2D(
            filters,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )(inputs)
        conv = layers.Conv2D(
            filters,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )(conv)
    elif dsc == 1:
        conv = layers.Conv2D(
            filters,
            1,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )(inputs)
        conv = layers.SeparableConv2D(
            4 * filters,
            3,
            activation="relu",
            padding="same",
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
        )(conv)
        conv = squeeze_excite_block(conv)
        conv = layers.Conv2D(
            filters,
            1,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )(conv)
        conv = layers.concatenate([conv, inputs], axis=3)
        conv = layers.Conv2D(
            nfilters2ndconv,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )(conv)
    else:
        raise ValueError(f"Unsupported dsc {dsc!r}: must be 0 or 1.")
    return conv


def unet_downsampling(inputs: Any) -> Any:
    """
    Halves the spatial resolution via 2x2 max pooling.

    Parameters:
        inputs (Any): The input Keras functional-API tensor.

    Returns:
        down (Any): The pooled tensor.
    """
    down = layers.MaxPooling2D(pool_size=(2, 2))(inputs)
    return down


def unet_upsampling(filters: int, inputs: Any) -> Any:
    """
    Doubles the spatial resolution via nearest-neighbor upsampling followed by a 2x2 convolution.

    Parameters:
        filters (int): The number of output filters for the convolution.
        inputs (Any): The input Keras functional-API tensor.

    Returns:
        up (Any): The upsampled tensor.
    """
    up = layers.Conv2D(
        filters, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(layers.UpSampling2D(size=(2, 2))(inputs))
    return up


def unet(
    inputs: Any,
    filters: list[int] = [2, 4, 8, 16, 32],
    dropout: float = 0.5,
    dsc: int = 0,
) -> Any:
    """
    Builds a 3-level U-Net (encoder down to filters[3], then back up),
    concatenating the input with the final decoder output before a 1x1
    linear output convolution. Three 2x2 poolings need patch_size divisible by 8 (the current patch_size is 8); the
    commented-out 4th level below (which would use filters[4]) needs patch_size divisible by 16.

    Parameters:
        inputs (Any): The model input Keras functional-API tensor.
        filters (list[int] = [2, 4, 8, 16, 32]): The filter counts per U-Net level; filters[4] is unused at the
            current (3-level) depth.
        dropout (float = 0.5): The dropout rate applied at the bottleneck.
        dsc (int = 0): The convolutional block style passed through to unet_convblock.

    Returns:
        output (Any): The single-channel linear output tensor.
    """
    conv00 = unet_convblock(filters[0], inputs, dsc, downup=0)  # X00

    pool10 = unet_downsampling(conv00)
    conv10 = unet_convblock(filters[1], pool10, dsc, downup=0)  # X10

    pool20 = unet_downsampling(conv10)
    conv20 = unet_convblock(filters[2], pool20, dsc, downup=0)  # X20

    pool30 = unet_downsampling(conv20)
    conv30 = unet_convblock(filters[3], pool30, dsc, downup=0)  # X30
    drop30 = layers.Dropout(dropout)(conv30)

    # optional 4th level (needs patch_size divisible by 16); to use it, uncomment these lines and comment out the two
    # 3-level lines below
    # pool40 = unet_downsampling(drop30)
    # conv40 = unet_convblock(filters[4], pool40)  # X40
    # drop40 = layers.Dropout(dropout)(conv40)

    # up40 = unet_upsampling(filters[3], drop40)
    # merge31 = layers.concatenate([drop30, up40], axis=3)
    # conv31 = unet_convblock(filters[3], merge31, dsc, downup=1)  # X31

    # up31 = unet_upsampling(filters[2], conv31)
    # merge22 = layers.concatenate([conv20, up31], axis=3)

    # 3-level path (current): upsample straight from the level-3 bottleneck
    up31 = unet_upsampling(filters[2], drop30)
    merge22 = layers.concatenate([conv20, up31], axis=3)

    conv22 = unet_convblock(filters[2], merge22, dsc, downup=1)  # X22

    up22 = unet_upsampling(filters[1], conv22)
    merge13 = layers.concatenate([conv10, up22], axis=3)
    conv13 = unet_convblock(filters[1], merge13, dsc, downup=1)  # X13

    up13 = unet_upsampling(filters[0], conv13)
    merge04 = layers.concatenate([conv00, up13], axis=3)
    conv04 = unet_convblock(filters[0], merge04, dsc, downup=1)  # X04

    # skip connection from the raw input to the output layer
    aux = layers.concatenate([conv04, inputs], axis=3)

    output = layers.Conv2D(1, 1, activation="linear")(aux)

    return output


def smoothLC1(huber_value: float = 0.5, smoothness_weight: float = 40):
    """
    Builds a Keras loss combining three terms: Huber loss between y_true and
    the averaged prediction, Huber loss between the two prediction channels
    (encourages agreement), and an inverse-standard-deviation term,
    smoothness_weight / (1 + std), on the averaged prediction. This is the loss
    compile_cnn_model uses.

    Despite the parameter name, the third term is not a smoothness penalty: it shrinks as the prediction's standard
    deviation grows, so it rewards spread. The std is taken over the whole batch, including pixels zeroed by the NaN
    mask. disaggregation.py sets smoothness_weight to 0, which turns the term off.

    Parameters:
        huber_value (float = 0.5): The Huber loss delta.
        smoothness_weight (float = 40): The weight of the inverse-standard-deviation term.

    Returns:
        slc1 (Callable): The Keras loss function (y_true, y_pred) -> scalar tensor.
    """

    def slc1(y_true, y_pred):
        """Sum of the true-vs-prediction Huber loss, the cross-channel Huber loss, and the weighted inverse-std term."""
        numfinite = tensorflow.math.count_nonzero(tensorflow.math.is_finite(y_true))
        mask = tensorflow.where(
            tensorflow.math.is_nan(y_true),
            tensorflow.constant(0.0),
            tensorflow.constant(1.0),
        )
        y_true = tensorflow.math.multiply_no_nan(y_true, mask)
        y_pred = tensorflow.math.multiply_no_nan(y_pred, mask)
        y_pred = tensorflow.where(
            tensorflow.math.is_nan(y_pred), tensorflow.constant(0.0), y_pred
        )

        # DEFINE DELTA AND AVERAGE PREDICTIONS
        HUBER_DELTA = huber_value
        y_pred_avg_two_channels = ops.expand_dims(ops.mean(y_pred, axis=-1), -1)

        # DIFFERENCE BETWEEN PRED AND TRUE
        x1 = ops.abs(y_true - y_pred_avg_two_channels)
        x1 = tensorflow.where(
            x1 < HUBER_DELTA, 0.5 * x1**2, HUBER_DELTA * (x1 - 0.5 * HUBER_DELTA)
        )
        x1 = tensorflow.math.divide_no_nan(
            ops.sum(x1), tensorflow.cast(numfinite, tensorflow.float32)
        )

        # DIFFERENCE BETWEEN DIFFERENT PREDICTIONS
        x2 = ops.abs(y_pred[:, :, :, 0] - y_pred[:, :, :, 1])
        x2 = tensorflow.where(
            x2 < HUBER_DELTA, 0.5 * x2**2, HUBER_DELTA * (x2 - 0.5 * HUBER_DELTA)
        )
        x2 = tensorflow.math.divide_no_nan(
            ops.sum(x2), tensorflow.cast(numfinite, tensorflow.float32)
        )

        # INVERSE STANDARD DEVIATION
        x3 = 1 / (1 + ops.std(y_pred_avg_two_channels))
        cx3 = smoothness_weight
        x3 = cx3 * x3

        # Sum of all components
        sl = x1 + x2 + x3
        return sl

    return slc1


def compile_cnn_model(
    shape: list[int],
    learning_rate: float,
    dropout: float = 0.5,
    filters: list[int] = [8, 16, 32, 64, 128],
    huber_value: float = 1,
    smoothness_weight: float = 40.0,
    loss_parameters: list[int | bool] = [2, 0, 1],
) -> models.Model:
    """
    Builds and compiles the CNN model used for spatial disaggregation.

    The model holds two separate U-Nets (no shared weights). Branch A sees the input patch; branch B sees a
    transformed copy, and its output is transformed back. The two outputs are concatenated as two channels and the
    model is compiled with smoothLC1 as the loss, which fits their average to the target and penalizes their
    disagreement; predictloop averages them.

    The papers describe one network applied to both inputs, but the authors' code (which the original code here
    matches) builds two, as this does. One shared U-Net was tested (irr and pws-tot 2000, 3 seeds each) and made no
    consistent difference to HUC-12 accuracy, so the two separate U-Nets are kept.

    The transform is chosen once, when the model is built, by numpy's randint(0, 6). numpy is seeded with 42 in
    disaggregation.py just before, so the draw is always index 3, the transpose, and the saved model contains only
    the transpose (the original code made the same draw with the same result). Drawing a random orientation per
    patch instead was tested (irr 2000) and made no measurable difference to HUC-12 accuracy, so the fixed transpose
    is kept.

    Parameters:
        shape (list[int]): The model input shape, e.g. [patch_size, patch_size, channels].
        learning_rate (float): The Adam optimizer learning rate.
        dropout (float = 0.5): The dropout rate passed to unet.
        filters (list[int] = [8, 16, 32, 64, 128]): The filter counts per U-Net level, passed to unet.
        huber_value (float = 1): The Huber loss delta passed to the compiled loss function.
        smoothness_weight (float = 40.0): The weight of the loss's inverse-standard-deviation term.
        loss_parameters (list[int | bool] = [2, 0, 1]): [npredictions, usegradnorm, dsc]. npredictions must be 2
            and usegradnorm 0 (the original code's other modes were removed); dsc is the convolutional block style
            passed to unet_convblock.

    Returns:
        mod (models.Model): The compiled Keras model.
    """

    # set a seed for reproducibility
    tensorflow.random.set_seed(SEED)

    npredictions = loss_parameters[0]  # 2: original prediction + prediction from the transformed input
    usegradnorm = loss_parameters[1]  # 0: plain compile with a single loss (no GradNorm)
    dsc = loss_parameters[2]  # convolutional block style (0/1), see unet_convblock

    if npredictions != 2 or usegradnorm != 0:
        raise ValueError(
            f"Unsupported loss_parameters {loss_parameters!r}: npredictions must be 2 and usegradnorm 0."
        )

    inputs = layers.Input(shape)

    if npredictions >= 2:
        # transform for branch B, drawn once at build time (always index 3, the transpose, with the seed set in
        # disaggregation.py); tensorflow.case on a constant builds only the chosen branch into the model
        random_int = tensorflow.constant(random.randint(0, 6))

        input_b = tensorflow.case(
            [
                (
                    tensorflow.equal(random_int, tensorflow.constant(0)),
                    lambda: layers.Lambda(horizontal_flip, output_shape=lambda s: s)(
                        inputs
                    ),
                ),
                (
                    tensorflow.equal(random_int, tensorflow.constant(1)),
                    lambda: layers.Lambda(vertical_flip, output_shape=lambda s: s)(
                        inputs
                    ),
                ),
                (
                    tensorflow.equal(random_int, tensorflow.constant(2)),
                    lambda: layers.Lambda(hv_flip, output_shape=lambda s: s)(inputs),
                ),
                (
                    tensorflow.equal(random_int, tensorflow.constant(3)),
                    lambda: layers.Lambda(
                        transpose, output_shape=lambda s: (s[0], s[2], s[1], s[3])
                    )(inputs),
                ),
                (
                    tensorflow.equal(random_int, tensorflow.constant(4)),
                    lambda: layers.Lambda(rotate_cw, output_shape=lambda s: s)(inputs),
                ),
                (
                    tensorflow.equal(random_int, tensorflow.constant(5)),
                    lambda: layers.Lambda(rotate_ccw, output_shape=lambda s: s)(inputs),
                ),
            ],
            exclusive=True,
        )

        processed_a = unet(inputs, filters, dropout, dsc)
        processed_b = unet(input_b, filters, dropout, dsc)

        # Undo the transform used to build input_b so processed_b is
        # realigned with processed_a before concatenation. horizontal_flip,
        # vertical_flip, hv_flip, and transpose are each their own inverse;
        # rotate_cw and rotate_ccw invert each other.
        processed_b = tensorflow.case(
            [
                (
                    tensorflow.equal(random_int, tensorflow.constant(0)),
                    lambda: layers.Lambda(horizontal_flip, output_shape=lambda s: s)(
                        processed_b
                    ),
                ),
                (
                    tensorflow.equal(random_int, tensorflow.constant(1)),
                    lambda: layers.Lambda(vertical_flip, output_shape=lambda s: s)(
                        processed_b
                    ),
                ),
                (
                    tensorflow.equal(random_int, tensorflow.constant(2)),
                    lambda: layers.Lambda(hv_flip, output_shape=lambda s: s)(
                        processed_b
                    ),
                ),
                (
                    tensorflow.equal(random_int, tensorflow.constant(3)),
                    lambda: layers.Lambda(
                        transpose, output_shape=lambda s: (s[0], s[2], s[1], s[3])
                    )(processed_b),
                ),
                (
                    tensorflow.equal(random_int, tensorflow.constant(4)),
                    lambda: layers.Lambda(rotate_ccw, output_shape=lambda s: s)(
                        processed_b
                    ),
                ),
                (
                    tensorflow.equal(random_int, tensorflow.constant(5)),
                    lambda: layers.Lambda(rotate_cw, output_shape=lambda s: s)(
                        processed_b
                    ),
                ),
            ],
            exclusive=True,
        )

    if npredictions == 2:
        result = layers.Concatenate()([processed_a, processed_b])

    if usegradnorm == 0:
        if npredictions == 2:
            mod = models.Model(inputs=inputs, outputs=result)
            slc1 = smoothLC1(huber_value=huber_value, smoothness_weight=smoothness_weight)
            mod.compile(
                loss=slc1, optimizer=optimizers.Adam(learning_rate=learning_rate)
            )

            # create the optimizer's state variables now, so a save before the first fit includes them and loading
            # it doesn't warn about (and skip) a variable-count mismatch; they start at zero either way
            mod.optimizer.build(mod.trainable_variables)

    return mod


def reconstruct_patches(
    patches: NDArray, image_size: tuple[int, int], stride: int
) -> NDArray:
    """
    Reconstructs a full raster from overlapping patches, averaging across overlapping patches while ignoring NaNs.

    Patches must be in row-major order of their top-left corners, as from PatchView or sklearn's extract_patches_2d.
    Cells covered by no valid (non-NaN) patch value are 0, not NaN.

    Parameters:
        patches (NDArray): The patches to reconstruct from, shaped (n_patches, patch_size, patch_size, ...).
        image_size (tuple[int, int]): The (height, width) of the reconstructed raster.
        stride (int): The step size (in pixels) between neighboring patches' top-left corners.

    Returns:
        mean (NDArray): The reconstructed raster, averaged over overlapping patches.
    """

    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    # extra trailing dims (e.g. channels) are carried through
    extra_dims = patches.shape[3:]
    full_shape = (i_h, i_w) + extra_dims
    mean = numpy.zeros(full_shape)
    patch_count = numpy.zeros(full_shape)
    n_h = int((i_h - p_h) / stride + 1)
    n_w = int((i_w - p_w) / stride + 1)

    # patches are in row-major order of their top-left corners, so view them as an (n_h, n_w) grid of patches
    patch_grid = patches[: n_h * n_w].reshape((n_h, n_w, p_h, p_w) + extra_dims)

    # for a fixed offset (di, dj) within the patch, every patch writes to a distinct cell, so each offset is one
    # vectorized add; offsets run in reverse so each cell accumulates patches in the same order as a per-patch loop
    for di in reversed(range(p_h)):
        rows = slice(di, di + stride * (n_h - 1) + 1, stride)
        for dj in reversed(range(p_w)):
            cols = slice(dj, dj + stride * (n_w - 1) + 1, stride)
            offset_values = patch_grid[:, :, di, dj]
            offset_valid = ~numpy.isnan(offset_values)
            patch_count[rows, cols] += offset_valid
            mean[rows, cols] += numpy.where(offset_valid, offset_values, 0)

    mean = numpy.divide(
        mean, patch_count, out=numpy.zeros_like(mean), where=patch_count != 0
    )

    return mean


def predictloop(
    cnn_model: models.Model,
    patches: "PatchView",
    batch_size: int,
    patch_mask: Optional[NDArray[numpy.bool]] = None,
) -> NDArray[numpy.float32]:
    """
    Predicts patches in manual batches (rather than via model.predict), so
    NaNs in the input can be zeroed before each batch, and averages the
    model's first two output channels into a single prediction channel.

    Parameters:
        cnn_model (models.Model): The compiled/loaded CNN model.
        patches (PatchView): The input patches, shaped (n_patches, patch_size, patch_size, channels).
        batch_size (int): The number of patches per prediction batch.
        patch_mask (Optional[NDArray[numpy.bool]] = None): The patches to predict, shaped (n_patches,); other
            patches are NaN. All patches when None.

    Returns:
        y_pred_probs (NDArray[numpy.float32]): The predicted patches, shaped (n_patches, patch_size, patch_size, 1).
    """

    # Custom batched prediction loop; skipped patches stay NaN, which reconstruct_patches ignores
    final_shape = [patches.shape[0], patches.shape[1], patches.shape[2], 1]
    y_pred_probs = numpy.full(final_shape, numpy.nan, dtype=numpy.float32)

    # indices of the patches to predict
    patch_indices = (
        numpy.arange(patches.shape[0])
        if patch_mask is None
        else numpy.flatnonzero(patch_mask)
    )
    batch_count = math.ceil(len(patch_indices) / batch_size)

    for index in range(batch_count):
        batch_patch_indices = patch_indices[
            index * batch_size : (index + 1) * batch_size
        ]

        # Replace NANs in patches by zero (fancy indexing copies, so the input patches are untouched)
        patchespred = patches[batch_patch_indices]
        patchespred[numpy.isnan(patchespred)] = 0

        # Mean (pred original, pred transformed)
        y_pred_probs[batch_patch_indices] = numpy.expand_dims(
            numpy.mean(cnn_model.predict_on_batch(patchespred)[:, :, :, :2], axis=3),
            axis=3,
        )

        if (index + 1) % 1000 == 0:
            print("»» Batch", index + 1, "/", batch_count, end="\r")

    return y_pred_probs


def fit_rf(X: NDArray, y: NDArray) -> ensemble.RandomForestRegressor:
    """
    Fits a random forest regressor.

    Parameters:
        X (NDArray): The training features, shaped (n_samples, n_features).
        y (NDArray): The training targets, shaped (n_samples,).

    Returns:
        mod (ensemble.RandomForestRegressor): The fitted model.
    """

    mod = ensemble.RandomForestRegressor(
        random_state=SEED, criterion="squared_error", n_jobs=-1
    )  # scikit-learn defaults (100 trees)
    mod = mod.fit(X, y)
    return mod


def fit(
    X: NDArray,
    y: NDArray,
    p: list[float],
    method: str,
) -> ensemble.RandomForestRegressor | None:
    """
    Samples a proportion of pixels with a valid (non-NaN) target and, if
    method ends with "rf", fits a random forest on the sample. CNN methods
    are handled separately via fit_cnn.

    Parameters:
        X (NDArray): The ancillary variable raster stack, shaped (height, width, channels).
        y (NDArray): The target raster, shaped (height, width) or (height, width, 1).
        p (list[float]): The proportion of valid pixels to sample for training; only p[0] is used.
        method (str): The fitting method; only methods ending in "rf" are handled here.

    Returns:
        mod (ensemble.RandomForestRegressor | None): The fitted random forest, or None when method doesn't end in
            "rf".
    """

    X = X.reshape((-1, X.shape[2]))
    y = numpy.ravel(y)

    relevantids = numpy.where(~numpy.isnan(y))[0]
    relevsamples = random.choice(
        len(relevantids), round(len(relevantids) * p[0]), replace=False
    )
    idsamples = relevantids[relevsamples]
    print(
        "| --- Number of cells (all, valid target, sampled):",
        y.shape[0],
        len(relevantids),
        len(idsamples),
    )

    X = X[idsamples, :]
    y = y[idsamples]

    if method.endswith("rf"):
        print("|| Fit: Random Forests")
        return fit_rf(X, y)
    else:
        return None


def predict_cnn(
    obj: models.Model,
    ancpatches: "PatchView",
    dissshape: tuple[int, int],
    batch_size: int,
    stride: int = 1,
    patch_mask: Optional[NDArray[numpy.bool]] = None,
) -> list[NDArray]:
    """
    Predicts and reconstructs a full raster from ancillary patches using the
    given CNN model's currently loaded weights.

    Parameters:
        obj (models.Model): The CNN model to predict with.
        ancpatches (PatchView): The ancillary variable patches to predict from.
        dissshape (tuple[int, int]): The (height, width) of the raster to reconstruct.
        batch_size (int): The number of patches per prediction batch.
        stride (int = 1): The step size (in pixels) between neighboring patches' top-left corners, passed to
            reconstruct_patches.
        patch_mask (Optional[NDArray[numpy.bool]] = None): The patches to predict; other patches don't contribute
            to the reconstruction. All patches when None.

    Returns:
        predhr (list[NDArray]): A single-element list containing the reconstructed raster.
    """

    print("| --- Predicting new patches, 1 resolution U-Net / U-Net++")
    predhr = predictloop(obj, ancpatches, batch_size=batch_size, patch_mask=patch_mask)

    print("| ---- Reconstructing HR image from patches..")
    predhr = reconstruct_patches(predhr, dissshape, stride)

    return [predhr]
