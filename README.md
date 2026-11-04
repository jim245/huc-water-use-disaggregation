# Overview

This repository is a refactor of [spdisaggregation](https://github.com/joaomigl15/spdisaggregation), adapted for water use statistics and reorganized into a single pipeline driven by [main.py](./main.py). The pipeline takes water use totals per Hydrologic Unit Code (HUC) watershed. It spatially disaggregates them into gridded estimates across the conterminous United States, at 1000 m by default. It redistributes the HUC totals in three steps: dasymetric mapping against an ancillary weighting layer, pycnophylactic smoothing, and iterative refinement. The refinement alternates a U-Net convolutional neural network and a random forest. Both train on ancillary predictors: land cover, crops, terrain slope, precipitation, on-road emissions, economic output and population. Each step keeps the total of each HUC that has cells that can hold its water, and each refinement iteration rescales to these totals. A HUC with no such cells loses its total, and the evaluation reports that water as missing.

# Setup

Use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for package and environment management (recommended).

> The environment is for linux-64 with an NVIDIA GPU that CUDA 13 supports, with driver 580 or later. On a machine without a GPU, such as a login node, set `CONDA_OVERRIDE_CUDA=13.0` before you create the environment.

From the repository root, run `conda env create -f environment.yaml -p .conda` to create the environment in `.conda`, and `conda activate ./.conda` to activate it.

Run `.conda/bin/pre-commit install` once to install the [pre-commit hooks](./.pre-commit-config.yaml). They lint, type-check, spell-check and format each commit, according to the tool settings in [`pyproject.toml`](./pyproject.toml).

## Folder Structure

All inputs are in `input/`. The pipeline downloads the HUC boundaries, the NLCD rasters and the state boundaries. Put all other datasets at the paths in [Datasets Used](#datasets-used) before the run. The pipeline also writes its caches to `input/`.

The pipeline writes its outputs to `output/huc{huc_level}/{case_study}/{year}/{case_study}_huc{huc_level}_{year}.tif` in million gallons per day per cell. Each case study's yearly metrics are in `output/huc{huc_level}/{case_study}/metrics.csv`.

# Workflow

> Stages 1-3 are CPU-only, and may trip GPU-idle watchdogs.

1. Data preparation:
   - Download the [HUC boundaries](./utilities/download_huc_boundaries.py), the [NLCD rasters](./utilities/download_nlcd_rasters.py) and the [state boundaries](./utilities/figure_utilities.py), and [generate the template raster](./utilities/generate_template_raster.py).
   - Associate per-HUC water use statistics with HUC boundary polygons [[1]](./utilities/merge_irr_statistics.py)[[2]](./utilities/merge_pws_statistics.py)[[3]](./utilities/merge_pwscu_statistics.py). The three scripts share [statistics_utilities.py](./utilities/statistics_utilities.py).
   - Generate the ancillary layers of the case study [[1]](./utilities/generate_irr_rasters.py)[[2]](./utilities/generate_pws_rasters.py). Both scripts use [raster_utilities.py](./utilities/raster_utilities.py).
2. Spread each HUC total over its cells with [dasymetric mapping](./dasymetric_mapping.py).
3. Smooth the dasymetric estimates with [pycnophylactic interpolation](./pycnophylactic_interpolation.py). The result is the initial estimates.
4. Iteratively refine the initial estimates using the [disaggregation](./disaggregation.py) regression model.
5. [Evaluate](./evaluation.py) results by comparing predicted HUC totals with actuals. A run below HUC-12 also gets a holdout score against the HUC-12 statistics, which it never used, written to `output/huc{huc_level}/{case_study}/holdout_metrics.csv`.
6. [Render](./utilities/figure_utilities.py) the disaggregated raster to `output/huc{huc_level}/{case_study}/{year}/{case_study}_huc{huc_level}_{year}.png`.

The [main.py](./main.py) orchestrator script can be invoked as follows:

```text
usage: main.py [-h] --case_study {irr,pws-cu,pws-gw,pws-sw,pws-tot} --year YEAR [--huc2s HUC2 [HUC2 ...]] [--huc_level {2,4,6,8,10,12}] [--crs CRS] [--cell_size CELL_SIZE] [--detail_huc2s [HUC2 ...]] [--prepare_only]

Runs the spatial disaggregation pipeline for one case study and one year.

options:
  -h, --help            show this help message and exit
  --case_study {irr,pws-cu,pws-gw,pws-sw,pws-tot}
                        The case study to run. It selects the statistics and the ancillary layers.
  --year YEAR           The year to run. It selects the statistics and the ancillary layers. The years of each case study: irr 2000-2020, pws-cu 2009-2020, pws-gw 2000-2020, pws-sw 2000-2020, pws-tot 2000-2020.
  --huc2s HUC2 [HUC2 ...]
                        The HUC-2 watersheds of the run, from 1 to 18, without zero-padding (for example "1", not "01"). The default is all 18 (CONUS).
  --huc_level {2,4,6,8,10,12}
                        The HUC level of all stages. The source statistics stop at level 12. The project tested only levels 8 and 12.
  --crs CRS             The coordinate reference system of all spatial operations.
  --cell_size CELL_SIZE
                        The output cell size, in CRS units.
  --detail_huc2s [HUC2 ...]
                        The HUC-2 watersheds that the detail view of the figure zooms in on. They must be in --huc2s, so a run without 14 and 15 must set this option. Pass no values to omit the detail view.
  --prepare_only        Run only stage 1 (data preparation), and keep its downloads, statistics and caches in input/ for later runs. It needs no GPU.
```

Run each case study and year twice: `--huc_level 12` for the product, and `--huc_level 8` for its holdout score. Near-perfect scores in `metrics.csv` only show that the run keeps the HUC totals. The holdout score measures how well the method places water from HUC-8 down to HUC-12. It is a proxy, and it does not measure the error of individual cells.

The level 12 and level 8 runs of a case study and year both write the HUC-12 statistics. When these statistics don't exist yet, finish the level 12 run before you start the level 8 run. Don't run two years of one case study at the same HUC level at the same time, because both runs update the same `metrics.csv`.

Example:

```sh
python main.py --case_study pws-cu --year 2009 --huc2s 14 15 --huc_level 12 --crs EPSG:5070 --cell_size 1000
```

# Datasets Used

Paths are relative to `input/`. The pipeline downloads the datasets marked "Downloaded". When a raster has no file for the year, the pipeline extrapolates one from the years that do exist.

## Boundaries

| Dataset                                                                                                                                                      | Path                                    | Use                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------- | ---------------------- |
| [Watershed Boundary Dataset](https://www.usgs.gov/national-hydrography/watershed-boundary-dataset)                                                           | `HUC/{huc2}/Shape/WBDHU{huc_level}.shp` | All. Downloaded        |
| [Census Cartographic Boundary Files](https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html) (2025 states at 1:500,000) | `USA/cb_2025_us_state_500k.shp`         | The figure. Downloaded |

## Water Use Statistics

| Dataset                                                                                                | Path                                               | Use                                |
| ------------------------------------------------------------------------------------------------------ | -------------------------------------------------- | ---------------------------------- |
| [Irrigation](https://www.sciencebase.gov/catalog/item/639a11b9d34e0de3a1f0ddde)                        | `statistics/Irr_CU_HUC12_Tot_annual_2000_2020.csv` | irr, 2000-2020                     |
| [Public Water Supply](https://www.sciencebase.gov/catalog/item/65f86970d34e97daac9ff4c8)               | `statistics/PS_HUC12_{GW,SW,Tot}_2000_2020.csv`    | pws-gw, pws-sw, pws-tot, 2000-2020 |
| [Public Water Supply (Consumptive)](https://www.sciencebase.gov/catalog/item/65f869add34e97daac9ff4d2) | `statistics/ps_mon_CU_ETr.csv`                     | pws-cu, 2009-2020                  |

## Ancillary Inputs

| Dataset                                                                                                                                                    | Path                                                                    | Use                  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | -------------------- |
| [USDA Cropland Data Layer](https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php) (CDL)                                                       | `CDL/{year}/{year}_30m_cdls.tif`, from 2008. Years before 2008 use 2008 | irr                  |
| [DARTE On-road CO2 Emissions](https://data.nasa.gov/dataset/darte-annual-on-road-co2-emissions-on-a-1-km-grid-conterminous-usa-v2-1980-2017-4d5f4) (DARTE) | `DARTE/onroad_{year}.tif`                                               | pws                  |
| [Gross Domestic Product](https://www.nature.com/articles/s41597-022-01322-5#code-availability) (GDP)                                                       | `GDP/{year}GDP.tif`                                                     | pws                  |
| [Terrain Slope](https://www.landfire.gov/topographic/slope) (LandFire)                                                                                     | `LandFire/LF2020_SlpD_CONUS/Tif/LF2020_SlpD_CONUS.tif`                  | irr, all years       |
| [MRLC Annual NLCD Land Cover](https://www.mrlc.gov/data) (NLCD, Collection 1 version 1)                                                                    | `NLCD/{year}/Annual_NLCD_LndCov_{year}_CU_C1V1.tif`                     | irr, pws. Downloaded |
| [Precipitation](https://prism.oregonstate.edu/data/) (PRISM)                                                                                               | `PRISM/{year}/prism_ppt_us_30s_{year}.tif`                              | irr                  |
| [WorldPop Population Counts](https://hub.worldpop.org/geodata/listing?id=74) (WorldPop)                                                                    | `WorldPop/usa_ppp_{year}_1km_Aggregated.tif`                            | pws                  |
| [Public-Supply Water Service Areas](https://www.sciencebase.gov/catalog/item/608035dcd34e8564d6835790) (WSA_v1)                                            | `WSA_v1/WSA_v1.shp`                                                     | pws mask             |

# Citation

This pipeline builds on the self-training disaggregation method from [spdisaggregation](https://github.com/joaomigl15/spdisaggregation). If you use it, please cite:

> Monteiro, J., Martins, B., Costa, M., & Pires, J. M. (2021). Geospatial Data Disaggregation through Self-Trained Encoder–Decoder Convolutional Models. _ISPRS International Journal of Geo-Information_, 10(9), 619. https://doi.org/10.3390/ijgi10090619

# License

Pending
