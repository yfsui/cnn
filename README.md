# CNN Model for A Sentiment Classification Task

**Environment Variables**
* --train "./train" 
* --validation "./validation" 
* --eval "./eval" 
* --model_output_dir "./" 
* --config_file "./training_config.json" 
* --num_epoch 10


***model-output***
* `sentiment_model.h5`: model output 


***model_training***
* `sentiment_model_cnn.py`: code for cnn model


***glue-job***
* `glue_job_script.py`: script for Glue ETL job
* `glue-job-output`: outputs of Glue ETL job (three feature sets for train, dev, eval)


***datasets***
* `training.sample.csv`: full dataset
* `train.csv`: training dataset
* `dev.csv`: validation dataset
* `eval.csv`: test dataset


Link to dictionary on S3: https://e4577-cloud.s3.amazonaws.com/dictionary/glove.txt 
