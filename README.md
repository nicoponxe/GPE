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

The `make train` and `make test` commands are not working at the moment.

# The Input Files

The Matlab files are stored in the `Datasets/` folder. These files contain the gait % data for
21 subjects walking at 3 different speeds.

Since the gait % is from 1 to 100, the number of rows in each file is 2100.

The combination of these 3 files give us a total of 6300 rows, which are used to train the model
without data augmentation.


# K-FOLD

## Introduction

K-fold cross-validation is a technique used in machine learning for assessing the performance
and generalizability of predictive models. It is particularly useful when the available dataset
is limited in size like this one.

This is a breakdown of how k-fold cross-validation works:

- *Divide the Dataset*: The entire dataset is divided into 'k' equal-sized folds or subset.

- *Model Training and Validation Loop*: The process then involves a series of iterations,
   where in each iteration, a different fold is held out as the test set, and the remaining k-1
   folds are combined to form the training set. This means that for each iteration.

This results on

- *Reduced Bias*: Since each observation gets to be in the test set exactly once and in the
  training set k-1 times, the method reduces bias associated with random sampling.

- *Utilization of Data*: It allows every observation to be used for both training and testing,
  which is particularly beneficial when the dataset is small.

- *Reliable Estimate*: Provides a more reliable estimate of the model's performance on unseen
  data compared to using a single train/test split.

## K-Fold approach for this

The choice of 'k' can significantly affect the variance and bias of the estimation.

In this particular case, the k is equal to 1 person. This means we remove 1 person from the dataset,
THEN we agument and finally we test with the person.

On each step we generate a graphic with the identity function for the percentage and the prediction
made for that person originally removed from the dataset.

In total we have 21 "people" because we have 7 people divided into 3 different speeds.

## Results on K-Fold

The results on the K-Fold were not what we were expecting, therefore we have to
make some adjustments to the dataset or hyperparameters in order to get better results.

If we take a look at the RMSE and R2, the result is far from exceptional.

Below you will find the result. We have to reduce the RMSE and improve the R2 for all the
people.

```
Person,Train RMSE,Test RMSE,Train R2,Test R2
1,1.8166923409604707,19.791540261072015,0.9960391586418189,0.5299069116043991
2,1.8535959014347558,9.58166942503152,0.9958766063416553,0.889818915126812
3,1.7997239296899756,15.738160930027078,0.9961128038126628,0.7027426229109741
4,1.9944555699646755,13.305789637761787,0.9952260989852227,0.7875259071294702
5,1.800937234360717,8.142340700002556,0.9961075608495568,0.9204347889889972
6,2.040271871495994,10.438791248534299,0.995004249253384,0.8692248872121497
7,1.6340511403600801,12.746045947693352,0.9967955317980053,0.8050264778869366
8,1.691119113835892,9.975362329844456,0.9965677961509978,0.8805786335293371
9,1.8628191140083068,8.025696142448655,0.9958354694851309,0.9226981115260543
10,1.837336543312519,5.582011201880024,0.9959486281747476,0.9626056416946726
11,1.7443453433496139,15.206062836849219,0.9963483460223637,0.7225030339055423
12,1.9862059004010657,13.63917197183349,0.995265509896444,0.7767452600333055
13,1.7868343702023215,9.689056418229983,0.9961682843485913,0.887335356404868
14,1.8659568350474316,13.017404832279862,0.9958214282505128,0.796636269345979
15,1.949890187145972,4.619172634798135,0.9954370576154478,0.9743933323371522
16,1.7963590659099902,9.148510020355321,0.9961273256601536,0.8995556728562355
17,1.7634810909134402,13.634514619038375,0.9962677881091998,0.776897703094184
18,1.9301436208902283,11.847715996432505,0.9955290076240465,0.831541104911944
19,1.8728950235104314,10.344482723274924,0.9957902961067026,0.8715771703424742
20,1.8433507172773245,8.141864937450142,0.9959220619659324,0.9204440868170657
21,2.007381525344491,4.885394949275559,0.9951640196960163,0.9713566350910205
```

# Model Improvement

## Ideas

The model seems to work reasonably well. However we need to improve the model to make the
RMSE smaller and R2 a bit bigger across all the people in the datasets.

We'll try different techniques to improve the model, like:

- *Data Preprocessing*
    - *Normalization / Standardization*: Scale the numerical values to ensure there's no value that dominates the others.
    - *Smooth Input Data*: Since the input data is not smooth one suggestion was to use Kalman
       filter or Butterworth filter to smooth out the curve before training the model.

- *Hyperparameter Tuning*: Change model hyperparameters:
    - `max_depth`
    - `learning_rate` (prevents overfitting)
    - `n_estimators` (number of boosting rounds)
    - `subsample`
    - `colsample_bytree`
    - `gamma`

- *Check if the model is Overfitting*
    - Increase `reg_alpha` (to add regularization)
    - Increase `reg_lambda` (to add regularization)
    - Decrease `max_depth` (to make the model less complex)
    - Increase `min_child_weight` (to reduce overfit)

- *Early Stopping*
    - Prevent overfitting by stopping the training process if the model's performance doesn't
      improve after a number of rounds.

- *Dynamic Learning Rate*
    - Adjust the learning rate dynamically during training. Start higher and reduce it when the model approaches to the minimum.

# Normalization & Standardization

- *MIN/MAX*: We have normalized the data using a min/max function to normalize the data.

- *KALMAN FILTER*: Also we have smoothed out the curve before the training by appling a kalman filter.

