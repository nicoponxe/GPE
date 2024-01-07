# Gait Percentage Prediction ML
Gait Percent Estimation during Walking and Running using Sagittal Shank or Thigh Angles

## Setup

This doesn't require any setup except to have the Matlab files stored in the `Datasets/` folder.

This folder containes Matlab files that were provided by the university.

### ENV variables

There's one environment variable introduced called `PATH_TO_MAT` which is the path to the
Matlab files. You have to set that up (or change it in the `Makefile`) so the code works
correctly.

## Requirements

### Python Libraries

The python libraries (and its versions) are specified in the `requirements.txt`.

You have to install them by running:

```
pip3 install -r requirements.txt
```

### GNU make [not mandatory]

If you want you can also install GNU `make`. This would help you out to run the different
options without writing the command options.

In further sections only the `make` commands are going to be shown.

## HOW-TO use this section

If you haven't installed `make` as a dependency you will have to check the `Makefile` in order
to understand how to run the code.

To get help just run `make` or `make help`

## Steps

1. `make train`: This trains the model and generates the `xgb_model.pkl5` file that stores the model.
1. `make test`: This tests the generated model.
