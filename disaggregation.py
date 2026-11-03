"""
Stage 4 of the pipeline: spatial disaggregation (see main.py).
"""

import os
import random
from pathlib import Path
from typing import Literal, cast

# TensorFlow reads these when it is first imported, so they must be set before the TensorFlow and Keras imports below.
# PYTHONHASHSEED isn't set here, because Python reads it only at interpreter start-up.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# isort: split

import keras
import numpy
import tensorflow
from numpy import random as numpy_random
from numpy import typing as numpy_typing

from utilities import model_utilities, raster_utilities

NEGLIGIBLE_FRACTION = 1e-6


def run(
    prepared_layers: list[numpy_typing.NDArray[numpy.float32]],
    initial_estimates: numpy_typing.NDArray[numpy.float32],
    zones: numpy_typing.NDArray[numpy.int32],
    grid: raster_utilities.Grid,
    *,
    case_study: str,
    year: int,
    huc_level: int,
    support_layer: int | None = None,
) -> numpy_typing.NDArray[numpy.float32]:
    """
    Refines the initial estimates with alternating CNN and random forest iterations (see disaggregate).

    Writes the result to output/huc<huc_level>/<case_study>/<year>/<case_study>_huc<huc_level>_<year>.tif, and the CNN
    model to model.keras in the same directory.

    Parameters:
        prepared_layers (list[numpy_typing.NDArray[numpy.float32]]): The ancillary layers, NaN as nodata.
        initial_estimates (numpy_typing.NDArray[numpy.float32]): The initial estimates
            (pycnophylactic_interpolation.run).
        zones (numpy_typing.NDArray[numpy.int32]): The huc_level zone index raster (raster_utilities.rasterize_zones),
            0 outside all HUCs.
        grid (raster_utilities.Grid): The grid of all the rasters.
        case_study (str): The case study of the run.
        year (int): The statistics year.
        huc_level (int): The HUC level of the statistics.
        support_layer (int | None = None): The index of the prepared layer whose cells > 0 can receive water. All
            cells can receive water when None.

    Returns:
        disaggregated_raster (numpy_typing.NDArray[numpy.float32]): The disaggregated raster, as written.
    """

    directory = raster_utilities.run_directory(case_study=case_study, year=year, huc_level=huc_level)

    ancillary_layers = numpy.dstack(tuple(prepared_layers))

    set_seed_and_growth(model_utilities.SEED)

    disaggregated_raster = disaggregate(
        ancillary_layers,
        initial_estimates,
        zones,
        model_path=directory / "model.keras",
        support_layer=support_layer,
    )

    # the loop works in float64. The file and all later stages use the float32 values
    disaggregated_raster = disaggregated_raster.squeeze(-1).astype(numpy.float32)

    raster_utilities.write_raster(directory / f"{case_study}_huc{huc_level}_{year}.tif", disaggregated_raster, grid)

    return disaggregated_raster


def disaggregate(
    ancillary_layers: numpy_typing.NDArray[numpy.float32],
    initial_estimates: numpy_typing.NDArray[numpy.float32],
    zones: numpy_typing.NDArray[numpy.int32],
    *,
    model_path: Path,
    support_layer: int | None = None,
    patch_size: int = 8,
    learning_rate: float = 0.001,
    dropout: float = 0.5,
    filters: tuple[int, ...] = (8, 16, 32, 64, 128),
    huber_value: float = 1.0,
    dsc: int = 1,
    max_iterations: int = 30,
    sample_fraction: float = 0.01,
    iteration_epochs: int = 3,
    batch_size: int = 512,
    min_iterations: int = 30,
    converge: float = 1.5,
    random_forest_missing_value: float = numpy.nan,
    cnn_target_scale: float | Literal["std"] = "std",
) -> numpy_typing.NDArray[numpy.float64]:
    """
    Refines the initial estimates with alternating CNN and random forest iterations.

    Each iteration fits a model that predicts the current map from the ancillary layers. Then it rescales the
    prediction so that each HUC keeps its total in the initial estimates. A HUC whose predicted sum is at most
    NEGLIGIBLE_FRACTION of its total keeps its values from the previous map, which hold its total. The result is the
    target of the next iteration. Odd iterations use the CNN, and even iterations use the random forest. Writes the CNN
    model to model_path when it builds the model and after each CNN iteration.

    Parameters:
        ancillary_layers (numpy_typing.NDArray[numpy.float32]): The ancillary layers on the grid of the initial
            estimates, shaped (height, width, layers), NaN as nodata.
        initial_estimates (numpy_typing.NDArray[numpy.float32]): The initial estimates. Their sum in each HUC is the
            total that each iteration keeps.
        zones (numpy_typing.NDArray[numpy.int32]): The zone index raster of the HUCs whose totals are kept, 0 outside
            all HUCs.
        model_path (Path): The CNN model file.
        support_layer (int | None = None): The index of the ancillary layer whose cells > 0 can receive water. The
            loop sets the predictions on other cells to 0 before each rescale, and doesn't compute them. All cells can
            receive water when None.
        patch_size (int = 8): The side length of the square CNN patches. It must be divisible by 8
            (model_utilities.unet).
        learning_rate (float = 0.001): The Adam learning rate.
        dropout (float = 0.5): The dropout rate at the U-Net bottleneck.
        filters (tuple[int, ...] = (8, 16, 32, 64, 128)): The filter count of each U-Net level (model_utilities.unet).
        huber_value (float = 1.0): The Huber delta of the CNN loss (model_utilities.cnn_loss), in units of the
            scaled targets (see cnn_target_scale).
        dsc (int = 1): The convolution block style of the U-Net (model_utilities.unet_convblock).
        max_iterations (int = 30): The maximum number of iterations.
        sample_fraction (float = 0.01): The fraction of the training candidates that each iteration samples. The CNN
            candidates are the patches with a valid middle cell. The random forest candidates are the valid cells.
        iteration_epochs (int = 3): The number of CNN training epochs in each CNN iteration.
        batch_size (int = 512): The number of patches in each CNN training and prediction batch.
        min_iterations (int = 30): The first iteration at which the loop checks for convergence.
        converge (float = 1.5): The convergence limit. The error of an iteration is the mean absolute change from the
            previous map. From min_iterations on, the loop keeps the map with the lowest error while each error is
            less than converge × the previous error. At the first error that isn't, the loop stops and returns the
            kept map, or the current map if it kept none. With min_iterations equal to max_iterations, the result is
            always the map of the last iteration.
        random_forest_missing_value (float = numpy.nan): The value that the random forest sees where an ancillary
            layer has no data. NaN uses the missing-value handling of scikit-learn. The CNN always sees 0.
        cnn_target_scale (float | Literal["std"] = "std"): The factor that multiplies the CNN training targets. The
            loop divides the predictions by the same factor. The loss terms depend on the units of the targets. "std"
            uses 1 / std(initial estimates), so that the CNN trains at unit spread.

    Returns:
        estimates (numpy_typing.NDArray[numpy.float64]): The disaggregated raster, shaped (height, width, 1).
    """

    shape = cast(tuple[int, int], initial_estimates.shape)

    if cnn_target_scale == "std":
        cnn_target_scale = 1.0 / float(numpy.nanstd(initial_estimates))
    cnn_target_scale = float(cnn_target_scale)

    zone_count = int(zones.max()) + 1
    zone_indices = zones.ravel()

    initial_valid = ~numpy.isnan(initial_estimates).ravel()
    zone_totals = numpy.bincount(
        zone_indices[initial_valid],
        weights=initial_estimates.ravel()[initial_valid],
        minlength=zone_count,
    )
    zone_has_totals = numpy.bincount(zone_indices[initial_valid], minlength=zone_count) > 0
    zone_has_totals[0] = False

    estimates = numpy.expand_dims(initial_estimates, axis=2).astype(numpy.float64)
    estimates_valid = ~numpy.isnan(estimates[..., 0])
    previous_estimates = estimates

    zone_mask = numpy.where(zones != 0, 1.0, numpy.nan)

    # only these cells keep their values after the masking in the loop, so only the patches that hold one of them
    # need a prediction
    kept_cells = (zones != 0) & estimates_valid

    supported = (
        None if support_layer is None else numpy.nan_to_num(ancillary_layers[..., support_layer], nan=0.0) > 0
    )
    prediction_cells = kept_cells if supported is None else kept_cells & supported
    kept_patches = (
        numpy.lib.stride_tricks.sliding_window_view(prediction_cells, (patch_size, patch_size))
        .any(axis=(2, 3))
        .ravel()
    )

    ancillary_layers = ancillary_layers * numpy.dstack([zone_mask] * ancillary_layers.shape[2])

    ancillary_patches = model_utilities.PatchView(ancillary_layers, patch_size)

    random_forest_ancillary = (
        ancillary_layers
        if numpy.isnan(random_forest_missing_value)
        else numpy.nan_to_num(ancillary_layers, nan=random_forest_missing_value)
    )

    cnn_model = model_utilities.compile_cnn_model(
        (patch_size, patch_size, ancillary_layers.shape[2]),
        learning_rate,
        dropout,
        filters,
        huber_value,
        dsc,
    )
    cnn_model.save(model_path, include_optimizer=True)

    previous_error = -numpy.inf
    minimum_error = numpy.inf
    lowest_error_estimates: numpy_typing.NDArray[numpy.float64] | None = None

    for iteration in range(1, max_iterations + 1):
        estimates = estimates * zone_mask[..., numpy.newaxis]

        if iteration % 2 == 1:
            cnn_targets = estimates if cnn_target_scale == 1.0 else estimates * cnn_target_scale
            target_patches = model_utilities.PatchView(cnn_targets, patch_size)

            cnn_model = model_utilities.load_model(model_path, huber_value)
            model_utilities.fit_cnn(
                ancillary_patches,
                target_patches,
                sample_fraction,
                cnn_model,
                iteration_epochs,
                batch_size,
            )
            cnn_model.save(model_path, include_optimizer=True)

            predicted_map = model_utilities.predict_cnn(
                cnn_model,
                ancillary_patches,
                shape,
                batch_size,
                patch_mask=kept_patches,
            )
            if cnn_target_scale != 1.0:
                predicted_map = predicted_map / cnn_target_scale

        else:
            forest = model_utilities.fit_rf(random_forest_ancillary, estimates, sample_fraction)
            predicted_map = numpy.expand_dims(
                model_utilities.predict_rf(forest, random_forest_ancillary, cell_mask=prediction_cells), axis=2
            )

        predicted_map = predicted_map * zone_mask[..., numpy.newaxis]
        predicted_map[predicted_map < 0] = 0
        predicted_map[~estimates_valid, :] = numpy.nan

        if supported is not None:
            predicted_map[kept_cells & ~supported] = 0.0

        predicted_valid = ~numpy.isnan(predicted_map).ravel()
        predicted_sums = numpy.bincount(
            zone_indices[predicted_valid],
            weights=predicted_map.ravel()[predicted_valid],
            minlength=zone_count,
        )
        zone_has_predictions = numpy.bincount(zone_indices[predicted_valid], minlength=zone_count) > 0

        rescale = zone_has_totals & zone_has_predictions & (predicted_sums > zone_totals * NEGLIGIBLE_FRACTION)
        zone_ratios = numpy.ones(zone_count)
        _ = numpy.divide(zone_totals, predicted_sums, out=zone_ratios, where=rescale)

        fallback_zones = zone_has_totals & (zone_totals > 0) & ~rescale
        estimates = numpy.where(
            fallback_zones[zones][..., numpy.newaxis],
            estimates,
            predicted_map * zone_ratios[zones][..., numpy.newaxis],
        )
        if fallback_zones.any():
            print(
                f"Disaggregation fallback: {int(fallback_zones.sum())} HUCs kept their previous values, "
                + f"{zone_totals[fallback_zones].sum():.4f} mgd"
            )

        error = numpy.nanmean(abs(estimates - previous_estimates))
        error_rate = (error / previous_error) if previous_error > 0 else numpy.inf
        previous_error = error

        if iteration >= min_iterations:
            if error_rate < converge:
                if error < minimum_error:
                    minimum_error = error
                    lowest_error_estimates = numpy.copy(estimates)
            else:
                if lowest_error_estimates is not None:
                    estimates = lowest_error_estimates
                break
        previous_estimates = estimates

    return estimates


def set_seed_and_growth(seed: int) -> None:
    """
    Seeds Python, NumPy and TensorFlow, clears the Keras session, and lets TensorFlow allocate GPU memory as needed.

    Parameters:
        seed (int): The seed.
    """

    random.seed(seed)
    numpy_random.seed(seed)
    keras.backend.clear_session()
    tensorflow.random.set_seed(seed)

    for gpu in tensorflow.config.list_physical_devices("GPU"):
        tensorflow.config.experimental.set_memory_growth(gpu, True)
