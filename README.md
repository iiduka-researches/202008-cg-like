# 202008-cg-like


## Usage
### Preparation
#### Criteo dataset
1. Please download dataset from [Kaggle Display Advertising Challenge Dataset](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)
1. Please put train.txt under dataset/data/criteo/


### Execution
To conduct experiments, please execute the following: 
```shell script
python main.py <dataset name> <model name>
```
or 
```shell script
pipenv run python main.py <dataset name> <model name>
```

### ISC
To execute the command on the Ikuta Fast Scalar Computation (ISC) Server, please execute the following:  
```shell script
pipenv shell
pip install --cache-dir <dir> torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```

### Utils
- To observe experiment processes by LINE Notify, you need to make or edit utils/line/token.json
  ```json
  {
    "url": "https://notify-api.line.me/api/notify",
    "token": "ACCESS_TOKEN"
  }
  ```
  You can get an access token from https://notify-bot.line.me/

- To send CSV by Gmail, you need to make or edit utils/gmail/account.json  
  ```json
  {
    "service_account": "GOOGLE_ACCOUNT",
    "password": "PASSWORD"
  }
  ```
  
  and utils/gmail/client_secret.json 
  - https://developers.google.com/gmail/api/  
  - https://console.cloud.google.com/
