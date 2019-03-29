# AutoMachineLearning


three api for u

## upload_dataset
upload_data_file_url = http://119.3.102.208:8279/upload_file
Method : POST
Parameter Form:
> * files = {'file': ('files', strings, "application/octet-stream")}

> * strings is the binary data with dict-form, which own given keys and values
> * {'X_train':X_train, 'y_train':y_train,'X_test':X_test, 'y_test':y_test}
        
requests.post(upload_data_file_url, files=files)

return data_id  or Error Info


## check_dataset

url= http://119.3.102.208:8279/check_file/
> * dataid is the code get from upload_dataset API
Method : GET
Sample:
requests.get(url + dataid)

return dict

## auto_ml

base_url = http://119.3.102.208:8279/
url = base_url  + 'AutoML/' + dataid
> * dataid is the code get from upload_dataset API

params = {'regressor': 'Null',
          'preprocessing': [],
          'max_evals': 5,
         'trial_timeout': 10,
          'seed': None}
Method : POST
Sample:
>> * r = requests.post(url, params=params)


return Model parameter



