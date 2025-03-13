# SmaAt-UNet
Code for the Paper "SmaAt-UNet: Precipitation Nowcasting using a Small Attention-UNet Architecture" [Arxiv-link](https://arxiv.org/abs/2007.04417), [Elsevier-link](https://www.sciencedirect.com/science/article/pii/S0167865521000556?via%3Dihub)

![SmaAt-UNet](SmaAt-UNet.png)

The proposed SmaAt-UNet can be found in the model-folder under [SmaAt_UNet](models/SmaAt_UNet.py) (Not Included in the current project due to time constrains, will be modified and re-included in the future).

**>>>IMPORTANT<<<**

The original Code from the paper can be found in this branch: https://github.com/HansBambel/SmaAt-UNet/tree/snapshot-paper

The current master branch has since upgraded packages and was refactored. Since the exact package-versions differ the experiments may not be 100% reproducible.

If you have problems running the code, feel free to open an issue here on Github.

## Installing dependencies
This project is using [rye](https://rye-up.com/) as dependency management. Therefore, installing the required dependencies is as easy as this:
```shell
rye sync --no-lock
```

In any case a [requirements.txt](requirements.txt) is also added by converting the lock-file from rye:
```shell
sed '/^-e/d' requirements.lock > requirements.txt
sed '/^-e/d' requirements-dev.lock > requirements-dev.txt
```

Basically, only the following requirements are needed:
```
tqdm
torch
lightning
tensorboard
torchsummary
h5py
numpy
```

---
For the paper we used the [Lightning](https://github.com/Lightning-AI/lightning) -module (PL) which simplifies the training process and allows easy additions of loggers and checkpoint creations.
In order to use PL we created the model [UNetDS_Attention](models/unet_precip_regression_lightning.py) whose parent inherits from the pl.LightningModule. This model is the same as the pure PyTorch SmaAt-UNet implementation with the added PL functions.

### Training

For training on the precipitation task we used the [train_precip_lightning.py](train_precip_lightning.py) file.
The training will place a checkpoint file for every model in the `default_save_path` `lightning/precip_regression`.
After finishing training place the best models (probably the ones with the lowest validation loss) that you want to compare in another folder in `checkpoints/comparison`.

### Testing
The script [calc_metrics_test_set.py](calc_metrics_test_set.py) will use all the models placed in `checkpoints/comparison` to calculate the MSEs and other metrics such as Precision, Recall, Accuracy, F1, CSI, FAR, HSS.
The results will get saved in a json in the same folder as the models.

The metrics of the persistence model (for now) are only calculated using the script [test_precip_lightning.py](test_precip_lightning.py). This script will also use all models in that folder and calculate the test-losses for the models in addition to the persistence model.
It will get handled by [this issue](https://github.com/HansBambel/SmaAt-UNet/issues/28).

### Precipitation dataset
The dataset consists of precipitation maps in 1-hour intervals consiting of Precipitation data in April and October of 2019 and 2020 in Northern Central Vietnam hich is the peak monsoon season.

Use the [tiff_to_h5.py](tiff_to_h5.py) to compile the dataset into a single h5 file with preprocessing

Use the [create_dataset.py](create_datasets.py) to create the two datasets used from the original dataset.

The dataset is already normalized using a [Min-Max normalization](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)).
In order to revert this you need to multiply the images by 260.0; this results in the images showing the mm/h.

### Citation
```
@article{TREBING2021,
title = {SmaAt-UNet: Precipitation Nowcasting using a Small Attention-UNet Architecture},
journal = {Pattern Recognition Letters},
year = {2021},
issn = {0167-8655},
doi = {https://doi.org/10.1016/j.patrec.2021.01.036},
url = {https://www.sciencedirect.com/science/article/pii/S0167865521000556},
author = {Kevin Trebing and Tomasz Staǹczyk and Siamak Mehrkanoon},
keywords = {Domain adaptation, neural networks, kernel methods, coupling regularization},
abstract = {Weather forecasting is dominated by numerical weather prediction that tries to model accurately the physical properties of the atmosphere. A downside of numerical weather prediction is that it is lacking the ability for short-term forecasts using the latest available information. By using a data-driven neural network approach we show that it is possible to produce an accurate precipitation nowcast. To this end, we propose SmaAt-UNet, an efficient convolutional neural networks-based on the well known UNet architecture equipped with attention modules and depthwise-separable convolutions. We evaluate our approaches on a real-life datasets using precipitation maps from the region of the Netherlands and binary images of cloud coverage of France. The experimental results show that in terms of prediction performance, the proposed model is comparable to other examined models while only using a quarter of the trainable parameters.}
}
```
