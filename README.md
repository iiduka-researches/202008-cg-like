# 202008-cg-like


## Usage
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

- To execute the command on the Ikuta Fast Scalar Computation (ISC) Server, please execute the following:  
  ```shell script
  pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
  ```

