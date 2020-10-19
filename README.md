# Modeling pipeline for prod2vec training with Python, Keras, Sagemaker and Airflow

### Project scope

This project contains a pipeline, orchestrated with Airflow, for training product vectors with Amazon Sagemaker.  The output of the pipeline is a set of trained product vectors that can be utilized in further modelling or analysis.

### Data utilized

The data utilizes Instacart data uploaded on to kaggle that contains customer orders and all items contained within the orders along with various mapping tables allowing for product ID's to be mapped to descriptions.  The data can be downloaded from [here](
https://www.kaggle.com/c/instacart-market-basket-analysis/data)

Specifically, the 'orders.csv' and 'products.csv' data sets were used in this project.

### Infrastructure

The infrastructure utilized to train the model is shown in the diagram below:

![](Img/prod2vec_infrastructure.PNG)

PyCharm was utilized as the IDE and code was automatically deployed to an ec2 instance with Airflow installed with a Postgres RDS instance.  Data was stored in an s3 bucket, models were tuned and trained utilizing Amazon Sagemaker.

### Airflow Orchestration

As menitoned above, Airflow was utilized to orchestrate and automate the pipeline.  The diagram below shows the DAG and tasks:

![](Img/prod2vec_airflow.PNG)

**pre_process_data:**  Reads the orders data and pre-processes it into lists of target and context pairs for ingestion into the model (more details in the modelling section below)  
**run_data_quality_checks:** Runs basic data quality checks on the pre-processed data prior to modelling e.g. missing value checks, data integrity checks  
**branching:** Task determines whether to run the hyperparameter tuning - this can be skipped if hyperparameters are known to speed up training  
**model_training:** Trains the model using provided hyperparameters  
**model_tuning:** Tunes the number of embeddings and learning rate with Bayesian hyperparameter tuning techniques  
**tuning_analysis:** Obtains a csv with the best training jobs from the tuning and some plots of the tuning job showing the hyperparameter search  
**post_processing:** Obtains the embedding layer from the model   

### prod2vec model details  

The model is based on a word2vec model developed with Keras explained in the following [blog](https://adventuresinmachinelearning.com/word2vec-keras-tutorial/).  

The model architecture is shown in the illustration below:

![](Img/model.PNG)

The model takes as input a target and context item pair which is passed through an embedding layer.  The size of the embedding layer can be tuned, in the model run in this project it was tuned between 100 and 500 nodes. The dot product between the context and target embeddings is calculated and the resulting vector is passed to a single node sigmoid layer where the target / context paired as classified as being a genuine pair (the item was found within the context) or a negative sample (the item was not found within the context). 
