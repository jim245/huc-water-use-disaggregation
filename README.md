# Overview

This repository is a refactor of [spdisaggregation](https://github.com/joaomigl15/spdisaggregation), adapted for water use statistics and reorganized into a single pipeline driven by [main.py](./main.py). The pipeline spatially disaggregates coarse water use statistics, reported as totals per Hydrologic Unit Code (HUC) watershed, into fine-resolution (1000m by default) gridded estimates across the conterminous United States. Reported HUC totals are redistributed in three stages: dasymetric mapping against an ancillary weighting layer, pycnophylactic smoothing, and iterative refinement with an alternating U-Net convolutional neural network and random forest trained on ancillary land cover, socioeconomic, and climate predictors. Every stage preserves mass: the cells within each HUC always sum to that HUC's reported total, with a rescaling step after each refinement iteration.

# Setup

[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) is recommended for package/environment management.

> If you are working in an environment without direct GPU access, you must set the `CONDA_OVERRIDE_CUDA` environment variable to `12.9` BEFORE installing Tensorflow, in order to force CUDA compatibility.

From the repository root, run `conda env create -f environment.yaml` to create the `WATER_DISAGGREGATION` environment with all required packages, and `conda activate WATER_DISAGGREGATION` to activate it.

## Folder Structure

Ancillary datasets are generally organized as `input/{DATASET}/{year}/` if there is more than 1 file per year. If there is only 1 file per year, they can be found in `input/{DATASET}/`.

Outputs are placed in `output/huc{huc_level}/{case_study}/{year}/{case_study}_huc{huc_level}_{year}.tif` in million gallons per day per cell. Each case study's yearly metrics are in `output/huc{huc_level}/{case_study}/metrics.csv`.

# Workflow

> Steps 0-4 are CPU-only, and may trip GPU-idle watchdogs.

0. Download any necessary datasets and [generate the template raster](./utilities/generate_template_raster.py).
1. Associate per-HUC water use statistics with HUC boundary polygons. [[1]](./utilities/merge_irr_statistics.py)[[2]](./utilities/merge_pws_statistics.py)[[3]](./utilities/merge_pwscu_statistics.py)
2. Generate relevant ancillary dataset rasters. [[1]](./utilities/generate_irr_rasters.py)[[2]](./utilities/generate_pws_rasters.py)
3. Compute initial estimates with [dasymetric mapping](./dasymetric_mapping.py).
4. Smooth initial estimates with [pycnophylactic interpolation](./pycnophylactic_interpolation.py).
5. Iteratively refine initial estimates using the [disaggregation](./disaggregation.py) regression model.
6. [Evaluate](./evaluation.py) results by comparing predicted HUC totals with actuals.
7. [Render](./utilities/figure_utilities.py) the disaggregated raster to `output/huc{huc_level}/{case_study}/{year}/{case_study}_huc{huc_level}_{year}.png`.

The [main.py](./main.py) orchestrator script can be invoked as follows:

```text
usage: main.py [-h] [--case_study {irr,pws-cu,pws-gw,pws-sw,pws-tot}] [--year YEAR] [--huc2s HUC2S [HUC2S ...]] [--huc_level {2,4,6,8,10,12,14,16}] [--crs CRS] [--cell_size CELL_SIZE] [--bottom_panel_huc2s [BOTTOM_PANEL_HUC2S ...]]

options:
  -h, --help            show this help message and exit
  --case_study {irr,pws-cu,pws-gw,pws-sw,pws-tot}
                        The case study to run, determines input statistics and ancillary data layers.
  --year YEAR           The year that determines statistics and ancillary data layers.
  --huc2s HUC2S [HUC2S ...]
                        The HUC-2 watersheds used for all spatial operations and calculations.
  --huc_level {2,4,6,8,10,12,14,16}
                        The HUC scale used for statistics, initial estimates and spatial disaggregation.
  --crs CRS             The coordinate reference system used in all spatial operations.
  --cell_size CELL_SIZE
                        The desired output cell size (in CRS units).
  --bottom_panel_huc2s [BOTTOM_PANEL_HUC2S ...]
                        The HUC-2 watersheds shown in the figure's bottom panel, any subset of --huc2s; pass none to omit it.
```

Example:

```sh
python main.py --case_study pws-cu --year 2009 --huc2s 14 15 --huc_level 12 --crs EPSG:5070 --cell_size 1000
```

# Datasets Used

## Boundaries

- [Watershed Boundary Dataset](https://www.usgs.gov/national-hydrography/watershed-boundary-dataset)

## Water Use Statistics

- [Irrigation](https://www.sciencebase.gov/catalog/item/639a11b9d34e0de3a1f0ddde) (irr, 2000-2020)
- [Public Water Supply](https://www.sciencebase.gov/catalog/item/65f86970d34e97daac9ff4c8) (pws-[gw|sw|tot], 2000-2020)
- [Public Water Supply (Consumptive)](https://www.sciencebase.gov/catalog/item/65f869add34e97daac9ff4d2) (pws-cu, 2009-2020)

## Ancillary Inputs

- [USDA Cropland Data Layer](https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php) (CDL)
- [DARTE On-road CO2 Emissions](https://data.nasa.gov/dataset/darte-annual-on-road-co2-emissions-on-a-1-km-grid-conterminous-usa-v2-1980-2017-4d5f4) (DARTE)
- [Gross Domestic Product](https://www.nature.com/articles/s41597-022-01322-5#code-availability) (GDP)
- [Terrain Slope](https://www.landfire.gov/topographic/slope) (LandFire)
- [MRLC Land Cover](https://www.mrlc.gov/data) (NLCD)
- [Precipitation](https://prism.oregonstate.edu/data/) (PRISM)
- [WorldPop Population Counts](https://hub.worldpop.org/geodata/listing?id=74) (WorldPop)
- [Public-Supply Water Service Areas](https://www.sciencebase.gov/catalog/item/608035dcd34e8564d6835790) (WSA_v1)

# Citation

This pipeline builds on the self-training disaggregation method from [spdisaggregation](https://github.com/joaomigl15/spdisaggregation). If you use it, please cite:

> Monteiro, J., Martins, B., Costa, M., & Pires, J. M. (2021). Geospatial Data Disaggregation through Self-Trained Encoder–Decoder Convolutional Models. _ISPRS International Journal of Geo-Information_, 10(9), 619. https://doi.org/10.3390/ijgi10090619

# License

Pending
