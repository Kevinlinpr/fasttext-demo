import fasttext

train_data = 'cooking.valid'
valid_data = 'cooking.valid'
model_file_path = './model_cooking.bin'

if __name__ == '__main__':
    model = fasttext.train_supervised(input=train_data, 
                                      autotuneValidationFile=valid_data, 
                                      lr=1.0,
                                      autotuneDuration=600)
    valid_result = model.test(valid_data)
    print(valid_result)
    if valid_result[1] > 0.99 and valid_result[2] > 0.99:
        model.save_model(model_file_path)
        print('Training up to standard!')
    else:
        print('Retrain!')
    
