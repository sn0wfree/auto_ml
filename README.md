# AutoMachineLearning


* Three API for u currently

## Upload_dataset
* Url : http://119.3.102.208:8279/upload_file
* Method : POST
* Parameter Form:
> * files = {'file': ('files', strings, "application/octet-stream")}
> * strings is the binary data with dict-form, which own given keys and values
>> * strings = pickle.dumps({'X_train':X_train, 'y_train':y_train,'X_test':X_test, 'y_test':y_test},2)
* Return data_id  or Error Info


Sample:     
> * requests.post(url, files=files)




## Check_dataset

* Url : http://119.3.102.208:8279/check_file/
* Method : GET
> * **dataid** is the code get from upload_dataset API
* Parameter Form:
>> * None

* Return dict

Sample:
> * requests.get(url + dataid)


## Auto_ml

* base_url = http://119.3.102.208:8279/
* url = base_url  + 'AutoML/' + dataid
* Method : POST
> * **dataid**  is the code get from upload_dataset API
* Parameter Form:
>> * params = {'regressor': 'Null','preprocessing': 'Null', 'max_evals': 5,'trial_timeout': 10,'seed': 'Null'}
* Return Model parameter


Sample:
>> * r = requests.post(url, params=params)






