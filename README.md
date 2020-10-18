# Modeling pipeline for prod2vec training with Python, Keras, Sagemaker and Airflow

### Project scope

This project contains a pipeline, orchestrated with Airflow, for training product vectors with Amazon Sagemaker.  The output of the pipeline is a set of trained product vectors that can be utilized in further modelling or analysis.

### Data utilized

The data utilizes Instacart data uploaded on to kaggle that contains customer orders and all items contained within the orders along with various mapping tables allowing for product ID's to be mapped to descriptions.  The data can be downloaded from [here](
https://www.kaggle.com/c/instacart-market-basket-analysis/data)

Specifically, the 'orders.csv' and 'products.csv' data sets were used in this project.

### Infrastructure

![](Img/prod2vec_infrastructure.PNG)
