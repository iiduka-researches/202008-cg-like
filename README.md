# 202008-cg-like


## Usage
To conduct experiments, please execute the following: 
```shell script
python main.py
```

or 

```shell script
pipenv run python main.py
```

- To observe experiment processes by LINE Notify, you need to make or edit utils/line/token.json
  ```json
  {
    "url": "https://notify-api.line.me/api/notify",
    "token": "ACCESS_TOKEN"
  }
  ```

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

- To execute the command on the Ikuta Fast Scalar Computation (ISC) Server, please execute the following:  
  ```shell script
  pipenv shell
  pip install --cache-dir <dir> torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
  ```

