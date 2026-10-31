"""
Disaggregates water use statistics using ancillary raster weights.
"""

import os
import random
from os import path
from typing import Optional, cast

# TensorFlow reads these when it is first imported, so they must be set before the TensorFlow/Keras imports below
# (they used to be set inside run(), after the import, where they had no effect). PYTHONHASHSEED isn't set: Python
# only reads it at interpreter start-up.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# isort: split

import keras
import numpy
from numpy import random as numpy_random
from numpy import typing as numpy_typing
from tensorflow import config as tensorflow_config
from tensorflow import random as tensorflow_random

from utilities import model_utilities, raster_utilities

SEED = 42  # the answer to life, the universe, and everything


def run(
    prepared_layers: list[numpy_typing.NDArray[numpy.float32]],
    initial_estimates: numpy_typing.NDArray[numpy.float32],
    zones: numpy_typing.NDArray[numpy.int32],
    grid: raster_utilities.Grid,
    case_study: str,
    year: int,
    huc_level: int = 12,
    support_layer: Optional[int] = None,
) -> numpy_typing.NDArray[numpy.float32]:
    """
    Disaggregates water use statistics using ancillary raster weights.

    The result is written to output/huc<huc_level>/<case_study>/<year>/<case_study>_huc<huc_level>_<year>.tif.

    Parameters:
        prepared_layers (list[numpy_typing.NDArray[numpy.float32]]): The ancillary layers, NaN as nodata.
        initial_estimates (numpy_typing.NDArray[numpy.float32]): The initial estimates (the pycnophylactic output).
        zones (numpy_typing.NDArray[numpy.int32]): The huc_level zone index raster, 0 outside all HUCs, the same zones
            the initial estimates preserve totals over.
        grid (raster_utilities.Grid): The grid all of the rasters are on.
        case_study (str): The case study to disaggregate.
        year (int): The statistics year.
        huc_level (int = 12): The HUC level whose totals are preserved.
        support_layer (Optional[int] = None): Index into prepared_layers of the layer whose cells > 0 may receive water
            (0 = NLCD cultivated for irrigation). All cells when None.

    Returns:
        disaggregated_raster (numpy_typing.NDArray[numpy.float32]): The spatially disaggregated raster, as written.
    """

    ancillary_datasets = numpy.dstack(tuple(prepared_layers))

    set_seed_and_growth(SEED)

    disaggregated_raster = runDisaggregation(
        case_study=case_study,
        year=year,
        huc_level=huc_level,
        initial_estimates_data=initial_estimates,
        id_raster=zones,
        ancillary_datasets=ancillary_datasets,
        patch_size=8,
        learning_rate=0.001,
        dropout=0.5,
        filters=[8, 16, 32, 64, 128],
        huber_value=1,
        smoothness_weight=0,
        loss_parameters=[2, 0, 1],
        max_iterations=30,
        methods=["apcnn", "aprf"],
        p=[0.01],
        cnn_model="unet",
        iteration_epochs=3,
        batch_size=512,
        min_iterations=30,
        converge=1.5,
        support_layer=support_layer,
    )

    # the loop works in float64; the file and everything downstream use the float32 values
    disaggregated_raster = disaggregated_raster.squeeze(-1).astype(numpy.float32)

    raster_utilities.write_raster(
        path.join(
            "output",
            f"huc{huc_level}",
            case_study,
            str(year),
            f"{case_study}_huc{huc_level}_{year}.tif",
        ),
        disaggregated_raster,
        grid,
    )

    return disaggregated_raster


def runDisaggregation(
    case_study: str,
    year: int,
    huc_level: int,
    initial_estimates_data: numpy_typing.NDArray[numpy.float32],
    id_raster: numpy_typing.NDArray[numpy.int32],
    ancillary_datasets: numpy_typing.NDArray[numpy.float32],
    patch_size: int = 8,
    learning_rate: float = 0.001,
    dropout: float = 0.5,
    filters: list[int] = [8, 16, 32, 64, 128],
    huber_value: float = 1.0,
    smoothness_weight: float = 0,
    loss_parameters: list[int] = [2, 0, 1],
    max_iterations: int = 30,
    methods: list[str] = ["apcnn", "aprf"],
    p: list[float] = [0.01],
    cnn_model: str = "unet",
    iteration_epochs: int = 3,
    batch_size: int = 64,
    min_iterations: int = 30,
    converge: float = 2.0,
    random_forest_missing_value: float = numpy.nan,
    cnn_target_scale: float | str = "std",
    support_layer: Optional[int] = None,
) -> numpy_typing.NDArray[numpy.float64]:
    """
    Spatially disaggregates water use statistics using ancillary raster weights.

    Starting from the initial estimates, each iteration fits a model (CNN or random forest, alternating) that predicts
    the current map from the ancillary layers, then rescales the prediction so every HUC keeps its initial total. The
    rescaled prediction becomes the next iteration's target.

    Parameters:
        case_study (str): The case study, used for the output directory the CNN model is saved in.
        year (int): The year, used for the output directory the CNN model is saved in.
        huc_level (int): The HUC level whose totals are preserved, used for the output directory the CNN model is saved in.
        initial_estimates_data (numpy_typing.NDArray[numpy.float32]): The initial estimates (the pycnophylactic output). Their per-HUC sums are the totals each iteration is rescaled to.
        id_raster (numpy_typing.NDArray[numpy.int32]): The zone index raster of the HUCs whose totals are preserved, 0 outside all HUCs (raster_utilities.rasterize_zones).
        ancillary_datasets (numpy_typing.NDArray[numpy.float32]): The ancillary layers on the initial estimates grid, shaped (height, width, layers).
        patch_size (int = 8): The side length of the square patches the CNN trains and predicts on. The U-Net depth in model_utilities.unet assumes 8.
        learning_rate (float = 0.001): The Adam optimizer learning rate.
        dropout (float = 0.5): The dropout rate at the U-Net bottleneck.
        filters (list[int] = [8, 16, 32, 64, 128]): The filter counts per U-Net level; filters[4] is unused at patch size 8.
        huber_value (float = 1.0): The Huber delta of the CNN loss terms, in units of the scaled targets (see cnn_target_scale).
        smoothness_weight (float = 0): Weight of the loss's spread term, 40 / (1 + std(prediction)) before 2026-09. At this data's scale any weight rewards spread over accuracy (see FIXES.md, HUC-12 collapse), so it's 0.
        loss_parameters (list[int] = [2, 0, 1]): [npredictions, usegradnorm, dsc], passed to model_utilities.compile_cnn_model. Only npredictions == 2 and usegradnorm == 0 are implemented. dsc 1 (separable conv + squeeze-excite blocks) replaced 0 (plain double 3x3 convs) in 2026-10: slightly better HUC-12 placement in holdout tests (see FIXES.md, fix 33).
        max_iterations (int = 30): The maximum number of iterations.
        methods (list[str] = ["apcnn", "aprf"]): The two methods to alternate: odd iterations use methods[0], even iterations methods[1]. A method ending in "cnn" uses the CNN, any other the random forest.
        p (list[float] = [0.01]): The fraction of training candidates sampled each iteration (patches with a valid centre cell for the CNN, valid cells for the random forest). Only p[0] is used.
        cnn_model (str = "unet"): Unused; the CNN is always a U-Net.
        iteration_epochs (int = 3): The number of CNN training epochs per CNN iteration.
        batch_size (int = 64): The number of patches per CNN training and prediction batch.
        min_iterations (int = 30): The first iteration at which convergence is checked.
        converge (float = 2.0): The convergence threshold: an iteration converges while its error (mean absolute change from the previous map) is less than converge × the previous iteration's error.
        random_forest_missing_value (float = numpy.nan): The value the random forest sees where an ancillary layer is missing. NaN uses scikit-learn's native missing-value handling; -9999999 reproduces the pre-2026-09 behavior (see FIXES.md, fix 6). The CNN always sees 0.
        cnn_target_scale (float | str = "std"): Factor the CNN's training targets are multiplied by (its predictions are divided by it again). The loss terms depend on the target's units (see FIXES.md, HUC-12 collapse); "std" uses 1 / std(initial estimates) so the CNN trains at unit spread. 1.0 reproduces the pre-2026-09 behavior.
        support_layer (Optional[int] = None): Index of the ancillary layer whose cells > 0 may receive water (0 = NLCD cultivated for irrigation). Predictions elsewhere are set to 0 before each HUC-8 rescale, and aren't computed (see FIXES.md, fix 15). All cells when None.

    Returns:
        initial_estimates_data (numpy_typing.NDArray[numpy.float64]): The disaggregated raster, shaped (height, width, 1).
    """

    output_directory = path.join("output", f"huc{huc_level}", case_study, str(year))

    initial_estimates_shape = cast(tuple[int, int], initial_estimates_data.shape)

    # the CNN trains on targets at unit spread by default (see cnn_target_scale)
    if cnn_target_scale == "std":
        cnn_target_scale = 1.0 / float(numpy.nanstd(initial_estimates_data))
    cnn_target_scale = float(cnn_target_scale)

    # zones that cover no cell can't affect any per-zone sum, so the highest index present sets the array sizes
    zone_count = int(id_raster.max()) + 1
    zone_raster_indices = id_raster.ravel()

    # per-HUC initial estimate totals, which each iteration's predictions are rescaled to match
    initial_estimates_valid = ~numpy.isnan(initial_estimates_data).ravel()
    zone_totals = numpy.bincount(
        zone_raster_indices[initial_estimates_valid],
        weights=initial_estimates_data.ravel()[initial_estimates_valid],
        minlength=zone_count,
    )
    zone_has_totals = (
        numpy.bincount(
            zone_raster_indices[initial_estimates_valid], minlength=zone_count
        )
        > 0
    )
    zone_has_totals[0] = False

    initial_estimates_data = numpy.expand_dims(initial_estimates_data, axis=2)
    initial_estimates_valid_mask = ~numpy.isnan(initial_estimates_data[..., 0])
    previous_initial_estimates_data = initial_estimates_data

    initial_estimates_mask = numpy.where(id_raster != 0, 1.0, numpy.nan)

    # only cells inside a HUC with a valid initial estimate survive masking after each prediction, so only those
    # cells, and only the patches containing at least one of them, need to be predicted
    kept_cells = (id_raster != 0) & initial_estimates_valid_mask

    # cells that may receive water; predictions elsewhere are zeroed, so they needn't be computed
    supported = (
        None
        if support_layer is None
        else numpy.nan_to_num(ancillary_datasets[..., support_layer], nan=0.0) > 0
    )
    prediction_cells = kept_cells if supported is None else kept_cells & supported
    kept_patches = (
        numpy.lib.stride_tricks.sliding_window_view(
            prediction_cells, (patch_size, patch_size)
        )
        .any(axis=(2, 3))
        .ravel()
    )

    ancillary_variables_mask = numpy.dstack(
        [initial_estimates_mask] * ancillary_datasets.shape[2]
    )

    # mask the ancillary datasets with the variable mask
    ancillary_datasets = ancillary_datasets * ancillary_variables_mask

    # ancillary variable patches, read from a sliding-window view rather than copied (ancillary_datasets must not
    # be modified from here on)
    ancillary_patches = model_utilities.PatchView(ancillary_datasets, patch_size)

    # random forest input: missing ancillary values are left as NaN (scikit-learn's forests handle NaN natively) or
    # filled with random_forest_missing_value; cells outside HUCs are never trained on or predicted
    random_forest_ancillary = (
        ancillary_datasets
        if numpy.isnan(random_forest_missing_value)
        else numpy.nan_to_num(ancillary_datasets, nan=random_forest_missing_value)
    )

    # compile the model and save the initial weights
    compiled_cnn_model = model_utilities.compile_cnn_model(
        [patch_size, patch_size, ancillary_datasets.shape[2]],
        learning_rate,
        dropout,
        filters,
        huber_value,
        smoothness_weight,
        loss_parameters=loss_parameters,
    )

    compiled_cnn_model.save(
        path.join(output_directory, "model.keras"), include_optimizer=True
    )

    # set initial error values
    previous_error = -numpy.inf
    minimum_error = numpy.inf

    for iteration in range(1, max_iterations + 1):
        # alternate methods: odd iterations use methods[0], even iterations methods[1]
        iteration_method = methods[0] if (iteration % 2) == 1 else methods[1]

        if iteration_method.endswith("cnn"):
            initial_estimates_data = (
                initial_estimates_data * initial_estimates_mask[..., numpy.newaxis]
            )

            # the loss is unit-dependent, so the CNN can train on scaled targets (undone after prediction)
            cnn_targets = (
                initial_estimates_data
                if cnn_target_scale == 1.0
                else initial_estimates_data * cnn_target_scale
            )
            patches = model_utilities.PatchView(cnn_targets, patch_size)

            compiled_cnn_model = model_utilities.load_model(
                path.join(output_directory, "model.keras"),
                huber_value,
                smoothness_weight,
            )
            model_utilities.fit_cnn(
                ancillary_patches,
                patches,
                p,
                compiled_cnn_model,
                iteration_epochs,
                batch_size,
            )
            compiled_cnn_model.save(
                path.join(output_directory, "model.keras"),
                include_optimizer=True,
            )

            predicted_maps = model_utilities.predict_cnn(
                compiled_cnn_model,
                ancillary_patches,
                initial_estimates_shape,
                batch_size,
                patch_mask=kept_patches,
            )
            if cnn_target_scale != 1.0:
                predicted_maps = [
                    predicted_map / cnn_target_scale for predicted_map in predicted_maps
                ]

        else:
            initial_estimates_data = (
                initial_estimates_data * initial_estimates_mask[..., numpy.newaxis]
            )

            mod = model_utilities.fit(
                random_forest_ancillary, initial_estimates_data, p, iteration_method
            )

            predicted_maps = model_utilities.predict(
                mod, random_forest_ancillary, cell_mask=prediction_cells
            )

            # expand the predicted maps dimensions
            for i in range(len(predicted_maps)):
                predicted_maps[i] = numpy.expand_dims(predicted_maps[i], axis=2)

        for predicted_map in predicted_maps:
            predicted_map = predicted_map * initial_estimates_mask[..., numpy.newaxis]
            predicted_map[predicted_map < 0] = 0
            predicted_map[~initial_estimates_valid_mask, :] = numpy.nan

            # no water on cells that can't receive it (before the HUC-8 rescale)
            if supported is not None:
                predicted_map[kept_cells & ~supported] = 0.0

            # per-HUC predicted totals
            predicted_valid = ~numpy.isnan(predicted_map).ravel()
            predicted_sums = numpy.bincount(
                zone_raster_indices[predicted_valid],
                weights=predicted_map.ravel()[predicted_valid],
                minlength=zone_count,
            )
            zone_has_predictions = (
                numpy.bincount(
                    zone_raster_indices[predicted_valid], minlength=zone_count
                )
                > 0
            )

            # rescale each HUC's predictions to its initial total; HUCs with a negligible predicted total are left as is
            rescale = (
                zone_has_totals
                & zone_has_predictions
                & (predicted_sums > zone_totals * 1e-6)
            )
            polygon_ratios = numpy.ones(zone_count)
            numpy.divide(zone_totals, predicted_sums, out=polygon_ratios, where=rescale)

            predicted_map = (
                predicted_map * polygon_ratios[id_raster][..., numpy.newaxis]
            )

            initial_estimates_data = predicted_map

        # convergence error: mean absolute change from the previous map, and its ratio to the previous error
        error = numpy.nanmean(
            abs(initial_estimates_data - previous_initial_estimates_data)
        )
        error_rate = (error / previous_error) if previous_error > 0 else numpy.inf
        previous_error = error

        # from min_iterations on, keep the lowest-error map while the error ratio stays below converge; once it
        # doesn't, stop and return that map (or the current one if the first checked iteration already fails). If
        # max_iterations is reached without stopping, the last map is returned, so with min_iterations ==
        # max_iterations the result is always the last iteration's map
        if iteration >= min_iterations:
            if error_rate < converge:
                if error < minimum_error:
                    minimum_error = error
                    lowest_error_disaggregated_data = numpy.copy(initial_estimates_data)
            else:
                if iteration != min_iterations:
                    initial_estimates_data = lowest_error_disaggregated_data
                break
        previous_initial_estimates_data = initial_estimates_data

    return initial_estimates_data


def set_seed_and_growth(seed: int):
    """
    Seed randomness and allow memory growth.

    Parameters:
      seed (int): An initial seed.
    """

    # seed randomness
    random.seed(seed)
    numpy_random.seed(seed)
    keras.backend.clear_session()
    tensorflow_random.set_seed(seed)

    # allow dynamic memory allocation for all available GPUs
    for gpu in tensorflow_config.list_physical_devices("GPU"):
        tensorflow_config.experimental.set_memory_growth(gpu, True)
