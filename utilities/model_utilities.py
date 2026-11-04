"""
The CNN and random forest models of stage 4 of the pipeline (see disaggregation.py).
"""

import gc
import math
import os
from collections.abc import Callable, Sequence
from concurrent import futures
from pathlib import Path
from typing import Any, Self

import numpy
import tensorflow
from keras import layers, models, optimizers, saving, utils
from numpy import random
from numpy.typing import NDArray
from sklearn import ensemble

SEED = 42


def predict_rf(
    forest: ensemble.RandomForestRegressor,
    ancillary: NDArray[numpy.floating[Any]],
    cell_mask: NDArray[numpy.bool],
) -> NDArray[numpy.float64]:
    """
    Predicts a raster with a fitted random forest.

    Parameters:
        forest (ensemble.RandomForestRegressor): The fitted random forest.
        ancillary (NDArray[numpy.floating[Any]]): The ancillary layers, shaped (height, width, layers).
        cell_mask (NDArray[numpy.bool]): The cells to predict, shaped (height, width). Other cells are NaN.

    Returns:
        predicted_map (NDArray[numpy.float64]): The predicted raster, shaped (height, width).
    """

    rows = ancillary.reshape((-1, ancillary.shape[2]))
    kept = cell_mask.ravel()
    predictions = numpy.full(rows.shape[0], numpy.nan)
    predictions[kept] = predict_rows_deterministic(forest, rows[kept])

    return predictions.reshape(ancillary.shape[0], ancillary.shape[1])


def predict_rows_deterministic(
    forest: ensemble.RandomForestRegressor, rows: NDArray[numpy.floating[Any]]
) -> NDArray[numpy.float64]:
    """
    Predicts rows with a random forest on parallel threads, with results that are the same in each run.

    The parallel predict of the forest (n_jobs != 1) adds the tree outputs in the order the threads finish, so the
    last bits change between runs. This function splits the rows into chunks, one for each thread. Each chunk does what
    RandomForestRegressor.predict does with n_jobs=1: it adds the tree predictions in estimator order, then divides by
    the number of trees. So the result is bit-identical to that predict. The trees are called directly, because the
    Parallel of scikit-learn, nested in threads, races on the global warning filters.

    Parameters:
        forest (ensemble.RandomForestRegressor): The fitted random forest.
        rows (NDArray[numpy.floating[Any]]): The rows to predict, shaped (n_rows, layers).

    Returns:
        predictions (NDArray[numpy.float64]): The predictions, shaped (n_rows,).
    """

    # a job can use fewer CPUs than the node has, and os.cpu_count() returns the node's count
    thread_count = min(len(os.sched_getaffinity(0)), max(1, len(rows)))
    # the input check that RandomForestRegressor.predict does (float32, NaN allowed). It is private in scikit-learn
    checked_rows = forest._validate_X_predict(rows)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]

    def predict_chunk(chunk: NDArray[numpy.float32]) -> NDArray[numpy.float64]:
        out = numpy.zeros(len(chunk), dtype=numpy.float64)
        for tree in forest.estimators_:
            out += tree.predict(chunk, check_input=False)
        out /= len(forest.estimators_)
        return out

    # tree prediction releases the GIL, and no Parallel of scikit-learn runs in the threads
    with futures.ThreadPoolExecutor(thread_count) as pool:
        chunk_predictions = list(pool.map(predict_chunk, numpy.array_split(checked_rows, thread_count)))  # pyright: ignore[reportUnknownArgumentType]

    return numpy.concatenate(chunk_predictions)


class DataGenerator(utils.Sequence):
    """
    Serves batches of ancillary and target patches to Keras, with NaN set to 0 in the ancillary patches.

    Parameters:
        ancillary_patches (PatchView): The ancillary patches.
        target_patches (PatchView): The target patches.
        sample_ids (NDArray[numpy.intp]): The patch numbers to serve, in order.
        batch_size (int): The number of patches in each batch.
    """

    def __init__(
        self: Self,
        ancillary_patches: "PatchView",
        target_patches: "PatchView",
        sample_ids: NDArray[numpy.intp],
        batch_size: int,
    ) -> None:
        super().__init__()
        self.ancillary_patches = ancillary_patches
        self.target_patches = target_patches
        self.sample_ids = sample_ids
        self.batch_size = batch_size

    def __len__(self: Self) -> int:
        """
        Returns the number of batches in each epoch.

        Returns:
            batch_count (int): The number of batches in each epoch.
        """
        return math.ceil(len(self.sample_ids) / self.batch_size)

    def __getitem__(self: Self, index: int) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
        """
        Builds one batch.

        Parameters:
            index (int): The batch number.

        Returns:
            batch (tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]): The ancillary and target
                patches.
        """
        batch_ids = self.sample_ids[index * self.batch_size : (index + 1) * self.batch_size]
        ancillary = self.ancillary_patches[batch_ids]
        ancillary[numpy.isnan(ancillary)] = 0
        return ancillary, self.target_patches[batch_ids]

    def on_epoch_end(self: Self) -> None:
        """
        Collects garbage after each epoch, to limit memory growth.
        """
        _ = gc.collect()


def fit_cnn(
    ancillary_patches: "PatchView",
    target_patches: "PatchView",
    sample_fraction: float,
    model: models.Model,
    epochs: int,
    batch_size: int,
) -> None:
    """
    Fits the compiled CNN on a random sample of the patches whose middle cell has a target value.

    Parameters:
        ancillary_patches (PatchView): The ancillary patches.
        target_patches (PatchView): The target patches, with NaN where the target has no value.
        sample_fraction (float): The fraction of the patches with a middle target value that the fit uses.
        model (models.Model): The compiled CNN.
        epochs (int): The number of epochs.
        batch_size (int): The number of patches in each batch.
    """

    tensorflow.random.set_seed(SEED)

    # an even patch size has no single middle cell. Rows truncate and columns round half to even, so the cell is
    # (patch_size / 2 - 1, patch_size / 2)
    middle_row = int((target_patches.shape[1] - 1) / 2)
    middle_column = int(round((target_patches.shape[2] - 1) / 2))

    candidates = numpy.where(~numpy.isnan(target_patches.pixel(middle_row, middle_column, 0)))[0]
    sample_ids = candidates[random.choice(len(candidates), round(len(candidates) * sample_fraction), replace=False)]

    print(
        "Number of patches (all, valid middle pixel, sampled):",
        target_patches.shape[0],
        len(candidates),
        len(sample_ids),
    )
    print("| --- Fit - 1 resolution U-Net Model", epochs)

    # verbose=2 prints one line for each epoch, not progress bars, which fill the log files with escape codes. Keras
    # takes an int, but its annotation allows only str
    _ = model.fit(
        DataGenerator(ancillary_patches, target_patches, sample_ids, batch_size=batch_size),
        epochs=epochs,
        verbose=2,  # pyright: ignore[reportArgumentType]
    )


def load_model(path: str | Path, huber_value: float) -> models.Model:
    """
    Loads a saved CNN and registers its loss function.

    Parameters:
        path (str | Path): The saved Keras model file.
        huber_value (float): The Huber delta of the loss (see cnn_loss).

    Returns:
        model (models.Model): The loaded model.
    """

    loss = cnn_loss(huber_value)
    model = models.load_model(path, {loss.__name__: loss}, safe_mode=False)
    if not isinstance(model, models.Model):
        raise TypeError(f"{path} doesn't hold a Keras model.")

    return model


class PatchView:
    """
    The overlapping square patches of a raster stack, with stride 1, served from a sliding-window view of the raster.

    No patch array is built, because a full array holds each cell up to patch_size² times. Patches are numbered in
    row-major order of their top-left corners, as in extract_patches_2d of scikit-learn. Patch k has its top-left
    corner at (k // n_w, k % n_w). NaN stays NaN. The raster must not change while the view is in use.

    Parameters:
        raster (NDArray[numpy.floating[Any]]): The raster stack, shaped (height, width, channels).
        patch_size (int): The side length of the square patches.
    """

    def __init__(self: Self, raster: NDArray[numpy.floating[Any]], patch_size: int) -> None:
        self.raster = raster
        # shaped (n_h, n_w, channels, patch_size, patch_size)
        self.windows = numpy.lib.stride_tricks.sliding_window_view(raster, (patch_size, patch_size), axis=(0, 1))
        self.n_h, self.n_w = self.windows.shape[:2]
        self.shape = (self.n_h * self.n_w, patch_size, patch_size, raster.shape[2])
        self.dtype = raster.dtype

    def __len__(self: Self) -> int:
        """
        Returns the number of patches.

        Returns:
            patch_count (int): The number of patches.
        """
        return self.shape[0]

    def __getitem__(self: Self, indices: NDArray[numpy.intp]) -> NDArray[numpy.floating[Any]]:
        """
        Copies out the requested patches.

        Parameters:
            indices (NDArray[numpy.intp]): The patch numbers.

        Returns:
            patches (NDArray[numpy.floating[Any]]): The patches, shaped (len(indices), patch_size, patch_size,
                channels).
        """
        rows, columns = numpy.divmod(numpy.asarray(indices), self.n_w)
        return numpy.ascontiguousarray(numpy.moveaxis(self.windows[rows, columns], 1, -1))

    def pixel(self: Self, row: int, column: int, channel: int = 0) -> NDArray[numpy.floating[Any]]:
        """
        Returns the value at one position in every patch, without building the patches.

        Parameters:
            row (int): The row in the patch.
            column (int): The column in the patch.
            channel (int = 0): The channel.

        Returns:
            values (NDArray[numpy.floating[Any]]): The value in each patch, shaped (n_patches,).
        """
        return self.raster[row : row + self.n_h, column : column + self.n_w, channel].ravel()


@saving.register_keras_serializable()
def horizontal_flip(x: tensorflow.Tensor) -> tensorflow.Tensor:
    """
    Reverses the column order of a batched image tensor, shaped (batch, height, width, channels).
    """
    return tensorflow.reverse(x, axis=[2])


@saving.register_keras_serializable()
def vertical_flip(x: tensorflow.Tensor) -> tensorflow.Tensor:
    """
    Reverses the row order of a batched image tensor, shaped (batch, height, width, channels).
    """
    return tensorflow.reverse(x, axis=[1])


@saving.register_keras_serializable()
def hv_flip(x: tensorflow.Tensor) -> tensorflow.Tensor:
    """
    Reverses the row and column order of a batched image tensor, shaped (batch, height, width, channels).
    """
    return tensorflow.reverse(x, axis=[1, 2])


@saving.register_keras_serializable()
def transpose(x: tensorflow.Tensor) -> tensorflow.Tensor:
    """
    Swaps the height and width axes of a batched image tensor, shaped (batch, height, width, channels).
    """
    return tensorflow.transpose(x, perm=[0, 2, 1, 3])


@saving.register_keras_serializable()
def rotate_cw(x: tensorflow.Tensor) -> tensorflow.Tensor:
    """
    Turns a batched square image tensor 90 degrees clockwise. The inverse of rotate_ccw.
    """
    return tensorflow.reverse(tensorflow.transpose(x, perm=[0, 2, 1, 3]), axis=[2])


@saving.register_keras_serializable()
def rotate_ccw(x: tensorflow.Tensor) -> tensorflow.Tensor:
    """
    Turns a batched square image tensor 90 degrees counterclockwise. The inverse of rotate_cw.
    """
    return tensorflow.reverse(tensorflow.transpose(x, perm=[0, 2, 1, 3]), axis=[1])


# the branch B transforms of compile_cnn_model: (transform, inverse, output shape)
TRANSFORMS: tuple[tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]], ...] = (
    (horizontal_flip, horizontal_flip, lambda s: s),
    (vertical_flip, vertical_flip, lambda s: s),
    (hv_flip, hv_flip, lambda s: s),
    (transpose, transpose, lambda s: (s[0], s[2], s[1], s[3])),
    (rotate_cw, rotate_ccw, lambda s: s),
    (rotate_ccw, rotate_cw, lambda s: s),
)


def squeeze_excite_block(tensor: Any, ratio: int = 16) -> Any:
    """
    Applies a squeeze-and-excitation block. It pools the input to one value for each channel. Then it learns a gate for
    each channel from those values, and multiplies the input by the gates.

    Parameters:
        tensor (Any): The input Keras tensor.
        ratio (int = 16): The channel reduction ratio of the squeeze step.

    Returns:
        x (Any): The input tensor multiplied by the channel gates.
    """
    filters = tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(tensor)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // ratio, activation="relu", kernel_initializer="he_normal", use_bias=False)(se)
    se = layers.Dense(filters, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(se)
    return layers.multiply([tensor, se])


def unet_convblock(filters: int, inputs: Any, dsc: int, downup: int = 0) -> Any:
    """
    Builds one U-Net convolution block.

    Parameters:
        filters (int): The base filter count of the block.
        inputs (Any): The input Keras tensor.
        dsc (int): The block style, 0 or 1. Style 0 is two 3x3 convolutions. Style 1 is a 1x1 convolution, a
            separable 3x3 convolution, a squeeze-and-excitation block and a 1x1 convolution. Then style 1 concatenates
            the input and applies a 3x3 convolution sized by downup.
        downup (int = 0): For dsc 1, the size of the last convolution: 2 × filters when 0 (downsampling path), and
            filters // 2 otherwise (upsampling path). dsc 0 ignores it.

    Returns:
        conv (Any): The output tensor of the block.
    """
    if dsc == 0:
        conv = layers.Conv2D(filters, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
        conv = layers.Conv2D(filters, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv)
    elif dsc == 1:
        conv = layers.Conv2D(filters, 1, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
        conv = layers.SeparableConv2D(
            4 * filters,
            3,
            activation="relu",
            padding="same",
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
        )(conv)
        conv = squeeze_excite_block(conv)
        conv = layers.Conv2D(filters, 1, activation="relu", padding="same", kernel_initializer="he_normal")(conv)
        conv = layers.concatenate([conv, inputs], axis=3)
        conv = layers.Conv2D(
            2 * filters if downup == 0 else filters // 2,
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
    Halves the resolution with 2x2 max pooling.

    Parameters:
        inputs (Any): The input Keras tensor.

    Returns:
        down (Any): The pooled tensor.
    """
    return layers.MaxPooling2D(pool_size=(2, 2))(inputs)


def unet_upsampling(filters: int, inputs: Any) -> Any:
    """
    Doubles the resolution with nearest-neighbor upsampling, then a 2x2 convolution.

    Parameters:
        filters (int): The filter count of the convolution.
        inputs (Any): The input Keras tensor.

    Returns:
        up (Any): The upsampled tensor.
    """
    return layers.Conv2D(filters, 2, activation="relu", padding="same", kernel_initializer="he_normal")(
        layers.UpSampling2D(size=(2, 2))(inputs)
    )


def unet(inputs: Any, filters: Sequence[int], dropout: float, dsc: int) -> Any:
    """
    Builds a 3-level U-Net. It concatenates the input with the last decoder output before a linear 1x1 output
    convolution.

    Parameters:
        inputs (Any): The input Keras tensor. Its patch size must be divisible by 8, because of the three 2x2
            pooling layers.
        filters (Sequence[int]): The filter counts of the levels. Only filters[0] to filters[3] are used.
        dropout (float): The dropout rate at the bottleneck.
        dsc (int): The block style (see unet_convblock).

    Returns:
        output (Any): The single-channel output tensor.
    """
    conv00 = unet_convblock(filters[0], inputs, dsc, downup=0)

    conv10 = unet_convblock(filters[1], unet_downsampling(conv00), dsc, downup=0)

    conv20 = unet_convblock(filters[2], unet_downsampling(conv10), dsc, downup=0)

    conv30 = unet_convblock(filters[3], unet_downsampling(conv20), dsc, downup=0)
    drop30 = layers.Dropout(dropout)(conv30)

    merge22 = layers.concatenate([conv20, unet_upsampling(filters[2], drop30)], axis=3)
    conv22 = unet_convblock(filters[2], merge22, dsc, downup=1)

    merge13 = layers.concatenate([conv10, unet_upsampling(filters[1], conv22)], axis=3)
    conv13 = unet_convblock(filters[1], merge13, dsc, downup=1)

    merge04 = layers.concatenate([conv00, unet_upsampling(filters[0], conv13)], axis=3)
    conv04 = unet_convblock(filters[0], merge04, dsc, downup=1)

    return layers.Conv2D(1, 1, activation="linear")(layers.concatenate([conv04, inputs], axis=3))


def cnn_loss(huber_value: float) -> Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor]:
    """
    Builds the CNN loss for the two output channels of compile_cnn_model.

    The loss is the sum of two Huber losses. The first is between the target and the mean of the two channels. The
    second is between the two channels, and it rewards agreement. Each term is divided by the number of valid target
    cells in the batch. Cells with a NaN target don't count.

    Parameters:
        huber_value (float): The Huber delta, in units of the targets.

    Returns:
        two_branch_huber (Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor]): The Keras loss
            function (y_true, y_pred) -> scalar tensor.
    """

    def huber(difference: tensorflow.Tensor) -> tensorflow.Tensor:
        return tensorflow.where(
            difference < huber_value, 0.5 * difference**2, huber_value * (difference - 0.5 * huber_value)
        )

    def two_branch_huber(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor) -> tensorflow.Tensor:
        """Returns the sum of the target-to-mean and the channel-to-channel Huber losses."""
        valid_count = tensorflow.cast(
            tensorflow.math.count_nonzero(tensorflow.math.is_finite(y_true)), tensorflow.float32
        )
        mask = tensorflow.where(tensorflow.math.is_nan(y_true), tensorflow.constant(0.0), tensorflow.constant(1.0))
        y_true = tensorflow.math.multiply_no_nan(y_true, mask)
        y_pred = tensorflow.math.multiply_no_nan(y_pred, mask)
        y_pred = tensorflow.where(tensorflow.math.is_nan(y_pred), tensorflow.constant(0.0), y_pred)

        y_pred_mean = tensorflow.expand_dims(tensorflow.reduce_mean(y_pred, axis=-1), -1)

        target_term = tensorflow.math.divide_no_nan(
            tensorflow.reduce_sum(huber(tensorflow.abs(y_true - y_pred_mean))), valid_count
        )
        agreement_term = tensorflow.math.divide_no_nan(
            tensorflow.reduce_sum(huber(tensorflow.abs(y_pred[:, :, :, 0] - y_pred[:, :, :, 1]))), valid_count
        )
        return target_term + agreement_term

    return two_branch_huber


def compile_cnn_model(
    shape: tuple[int, int, int],
    learning_rate: float,
    dropout: float,
    filters: Sequence[int],
    huber_value: float,
    dsc: int,
) -> models.Model:
    """
    Builds and compiles the CNN.

    The model holds two U-Nets with separate weights. Branch A sees the input patch. Branch B sees a transformed copy,
    and its output is transformed back. The two outputs are the two output channels. cnn_loss fits their mean to the
    target, and predict_patches uses their mean. The papers describe one network for both inputs. One shared U-Net
    made no consistent difference to HUC-12 accuracy in tests, so the two U-Nets stay separate.

    The transform is TRANSFORMS[random.randint(0, 6)], drawn from the global NumPy generator when the model is built.
    With the seed that disaggregation.run sets just before, the draw is 3, the transpose. The draw also moves the
    generator, so the later samples depend on it.

    Parameters:
        shape (tuple[int, int, int]): The input shape (patch_size, patch_size, layers).
        learning_rate (float): The Adam learning rate.
        dropout (float): The dropout rate at the U-Net bottleneck.
        filters (Sequence[int]): The filter counts of the U-Net levels.
        huber_value (float): The Huber delta of the loss (see cnn_loss).
        dsc (int): The block style (see unet_convblock).

    Returns:
        model (models.Model): The compiled model.
    """

    tensorflow.random.set_seed(SEED)

    inputs = layers.Input(shape)

    transform, inverse, output_shape = TRANSFORMS[random.randint(0, 6)]
    input_b = layers.Lambda(transform, output_shape=output_shape)(inputs)

    processed_a = unet(inputs, filters, dropout, dsc)
    processed_b = unet(input_b, filters, dropout, dsc)
    processed_b = layers.Lambda(inverse, output_shape=output_shape)(processed_b)

    model = models.Model(inputs=inputs, outputs=layers.Concatenate()([processed_a, processed_b]))
    model.compile(loss=cnn_loss(huber_value), optimizer=optimizers.Adam(learning_rate=learning_rate))

    # a save before the first fit must hold the optimizer variables. Without them, the load warns about a
    # variable-count mismatch and skips the optimizer state
    model.optimizer.build(model.trainable_variables)

    return model


def reconstruct_patches(patches: NDArray[numpy.floating[Any]], image_size: tuple[int, int]) -> NDArray[numpy.float64]:
    """
    Builds a raster from overlapping stride-1 patches. Each cell is the mean of the non-NaN values that cover it.

    The patches must be in row-major order of their top-left corners, as from PatchView. Cells that no valid patch
    value covers are 0, not NaN.

    Parameters:
        patches (NDArray[numpy.floating[Any]]): The patches, shaped (n_patches, patch_size, patch_size, ...).
        image_size (tuple[int, int]): The (height, width) of the raster.

    Returns:
        mean (NDArray[numpy.float64]): The raster, shaped (height, width, ...).
    """

    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    extra_dims = patches.shape[3:]
    full_shape = (i_h, i_w) + extra_dims
    mean = numpy.zeros(full_shape)
    patch_count = numpy.zeros(full_shape)
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1

    patch_grid = patches[: n_h * n_w].reshape((n_h, n_w, p_h, p_w) + extra_dims)

    # for one offset (di, dj) in the patch, every patch writes to a different cell, so each offset is one vectorized
    # add. The offsets run in reverse, so each cell adds its patches in the same order as a loop over the patches
    for di in reversed(range(p_h)):
        rows = slice(di, di + n_h)
        for dj in reversed(range(p_w)):
            cols = slice(dj, dj + n_w)
            offset_values = patch_grid[:, :, di, dj]
            offset_valid = ~numpy.isnan(offset_values)
            patch_count[rows, cols] += offset_valid
            mean[rows, cols] += numpy.where(offset_valid, offset_values, 0)

    return numpy.divide(mean, patch_count, out=numpy.zeros_like(mean), where=patch_count != 0)


def predict_patches(
    model: models.Model,
    patches: PatchView,
    batch_size: int,
    patch_mask: NDArray[numpy.bool],
) -> NDArray[numpy.float32]:
    """
    Predicts the patches in batches, with NaN set to 0 in each batch, and returns the mean of the two output channels.

    Parameters:
        model (models.Model): The CNN.
        patches (PatchView): The ancillary patches.
        batch_size (int): The number of patches in each batch.
        patch_mask (NDArray[numpy.bool]): The patches to predict, shaped (n_patches,). Other patches are NaN.

    Returns:
        predictions (NDArray[numpy.float32]): The predicted patches, shaped (n_patches, patch_size, patch_size, 1).
    """

    predictions = numpy.full([patches.shape[0], patches.shape[1], patches.shape[2], 1], numpy.nan, dtype=numpy.float32)

    patch_indices = numpy.flatnonzero(patch_mask)
    batch_count = math.ceil(len(patch_indices) / batch_size)

    for index in range(batch_count):
        batch_patch_indices = patch_indices[index * batch_size : (index + 1) * batch_size]

        # the indexing copies, so the raster under the view doesn't change
        batch = patches[batch_patch_indices]
        batch[numpy.isnan(batch)] = 0

        predictions[batch_patch_indices] = numpy.expand_dims(
            numpy.mean(model.predict_on_batch(batch)[:, :, :, :2], axis=3), axis=3
        )

        if (index + 1) % 1000 == 0:
            print("»» Batch", index + 1, "/", batch_count, end="\r")

    return predictions


def fit_rf(
    ancillary: NDArray[numpy.floating[Any]],
    targets: NDArray[numpy.floating[Any]],
    sample_fraction: float,
) -> ensemble.RandomForestRegressor:
    """
    Fits a random forest on a random sample of the cells that have a target value.

    The forest uses the scikit-learn defaults (100 trees), with all CPUs of the job.

    Parameters:
        ancillary (NDArray[numpy.floating[Any]]): The ancillary layers, shaped (height, width, layers).
        targets (NDArray[numpy.floating[Any]]): The target raster, shaped (height, width) or (height, width, 1), with
            NaN where the target has no value.
        sample_fraction (float): The fraction of the cells with a target value that the fit uses.

    Returns:
        forest (ensemble.RandomForestRegressor): The fitted random forest.
    """

    rows = ancillary.reshape((-1, ancillary.shape[2]))
    target_values = numpy.ravel(targets)

    candidates = numpy.where(~numpy.isnan(target_values))[0]
    sample_ids = candidates[random.choice(len(candidates), round(len(candidates) * sample_fraction), replace=False)]
    print(
        "| --- Number of cells (all, valid target, sampled):",
        target_values.shape[0],
        len(candidates),
        len(sample_ids),
    )

    print("|| Fit: Random Forests")
    forest = ensemble.RandomForestRegressor(random_state=SEED, criterion="squared_error", n_jobs=-1)
    return forest.fit(rows[sample_ids, :], target_values[sample_ids])


def predict_cnn(
    model: models.Model,
    ancillary_patches: PatchView,
    shape: tuple[int, int],
    batch_size: int,
    patch_mask: NDArray[numpy.bool],
) -> NDArray[numpy.float64]:
    """
    Predicts a raster with the CNN.

    Parameters:
        model (models.Model): The CNN.
        ancillary_patches (PatchView): The ancillary patches.
        shape (tuple[int, int]): The (height, width) of the raster.
        batch_size (int): The number of patches in each prediction batch.
        patch_mask (NDArray[numpy.bool]): The patches to predict. Other patches don't add to the raster.

    Returns:
        predicted_map (NDArray[numpy.float64]): The predicted raster, shaped (height, width, 1).
    """

    print("| --- Predicting new patches, 1 resolution U-Net / U-Net++")
    predictions = predict_patches(model, ancillary_patches, batch_size, patch_mask)

    print("| ---- Reconstructing HR image from patches..")
    return reconstruct_patches(predictions, shape)
