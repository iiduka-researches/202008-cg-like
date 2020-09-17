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
