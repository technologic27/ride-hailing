# Driver Allocation
==============================

  1. Analyze allocation experiments
  2. Build and evaluate an ML model

# Open Source Libraries

This project relies on a number of open source libraries/APIs. Why reinvent the wheel!

* Pandas
* Sklearn
* Seaborn

# Installation

Requires Python3.+ and virtual environment to run.

Change directory to the project directory.

```sh
cd path/to/project_directory
```
Set-up virtual environment for the first time.

```sh
$ make setup
```
Activate virtual environment.

```sh
$ source driver-allocation/bin/activate
```
Install requirements in Linux-based environment.

```sh
sudo pip3 install -r requirements.txt
```
Create clean dataset for metrics generation and modelling. Do this only once! 

```sh
$ make data
```
Generate results for analyzing allocation experiments.

```sh
$ make metrics
```
Train the ML model.

```sh
$ make train
```
Make predictions and generate order_id to driver_id allocations.

```sh
$ make predict
```
If virtual environment `driver-allocation` has been created before, activate virtual environment using the following command. Skip installation of requirements.

```sh
$ source driver-allocation/bin/activate
```
To run jupyter notebooks. 

```sh
$ jupyter notebooks
```

# File Submission Information
Model Information
---
I have modelled the driver allocation problem as a classification model. Where positive is successful completion of the trip and negative otherwise. I used a random forest classifier. The parameters of the model are `src/models/model_rfc.json`. The random forest classifier model artefact is `models/model_rfc.json`. The model prediction results and accuracy scores are `models/rfc_results.json`. On a high level, the model is underfitting and requires more features for better performance. A quick overview of the model results is as follows:

1. `accuracy rate of train`: 0.6626012492817557
2. `accuracy rate of test`: 0.6545863426026035

Metrics Submission
---
Files that have been created for submission are in the reports directory.
1. `reports/metrics.csv`
2. `reports/order_driver_match.csv`

Deep-dive Analysis of Model, Feature Importance
---
Results of the model are in the notebooks directory.
1. `notebooks/model_results.ipynb`

Notebooks Exploring Features and Raw Data
---
Exploratory data analysis of raw data and features are in the notebooks directory.
1. `notebooks/eda_explore_raw_data.ipynb`
2. `notebooks/eda_data_for_modelling.ipynb`
3. `notebooks/eda_feature_engineering.ipynb`

The raw data is submitted with the project found in the data/raw directory

Data Details
---
Raw data is found in `data/raw`. Cleaned for the interim data is found in `data\interim` and data used for metric generatio and modelling is found in `data/processed`. Only need to run `make data` once to generate the data. If the data has been generated, skip `make data` step.


### Todos

 - Write pytests
 - Create more features
 - Test on another ML model and compare results

License
----
