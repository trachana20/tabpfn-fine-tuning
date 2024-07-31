import config_utils

def set_augmentator(train_X, 
                    train_y,
                    test_X,
                    val_X,
                    target_col_name,
                    device,
                    flag,
                    gan_epochs):
    '''
    Set augmentation type
    '''
    
    if flag == "gan":
        return config_utils.apply_gans_augmentation(train_X, 
                                       train_y,
                                       target_col_name,
                                       device,
                                       gan_epochs)
    elif flag == "rag":
        return config_utils.apply_rag_augmentation(train_X, 
                                      train_y,
                                      test_X,
                                      val_X,
                                      target_col_name,
                                      device,
                                      gan_epochs,
                                      )
    else:
        return None,None,train_X, train_y
    
def set_model_config(model,flag):

    '''
    Set model config type
    '''    
    if flag == "linformer":
        return config_utils.apply_linformer(model)
    elif flag == "performer":
        return config_utils.apply_performer(model)
    elif flag == "lora":
        return config_utils.apply_lora(model)
    else:
        return model
    