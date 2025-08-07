future = pd.DataFrame(fb_df['ds'])

    # future


    # future = m.make_future_dataframe(periods=2160,freq='h')

    
    temp_merged['LR_Result'] = LR_model_load.predict(X)
#     temp_merged['LR_Result'] = new_results.predict(X)
    
    temp_merged['LR_percentage_err'] = (temp_merged['LR_Result']-temp_merged['final_usage'])/temp_merged['final_usage']
    temp_merged['LR_abs_percentage_err'] = abs((temp_merged['LR_Result']-temp_merged['final_usage'])/temp_merged['final_usage'])
    temp_merged['LR_RMSE'] = np.sqrt((temp_merged['LR_Result']-temp_merged['final_usage'])**2)

    temp_merged['yhat_percentage_err'] = (temp_merged['yhat']-temp_merged['final_usage'])/temp_merged['final_usage']
    temp_merged['yhat_abs_percentage_err'] = abs((temp_merged['yhat']-temp_merged['final_usage'])/temp_merged['final_usage'])
    
    y_test_cal = temp_merged[['final_usage']]

    X_test_cal = column_selection(temp_merged)



    X_train = X_test_cal[:int(X_test_cal.shape[0]*0.9)]
    X_test = X_test_cal[int(X_test_cal.shape[0]*0.9):]
    y_train = y_test_cal[:int(X_test_cal.shape[0]*0.9)]
    y_test = y_test_cal[int(X_test_cal.shape[0]*0.9):]

    # X_train, X_test, y_train, y_test = train_test_split(
    # X_test_cal, y_test_cal, test_size=0.2, random_state=42)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  
        'objective': 'regression', 
        'metric': {'mape'},  
        'num_leaves': 200,   
        'learning_rate': 0.05,  
        'feature_fraction': 0.7, 
        'bagging_fraction': 0.7, 
        'bagging_freq': 50,  
        'verbose':1  
    }

    print('Start training...')
    model = lgb.train(params,lgb_train,num_boost_round=2000,valid_sets=lgb_eval, early_stopping_rounds=300)
    # model = lgb.train(params,lgb_train,num_boost_round=2000)


    print('Start predicting...')
    y_hat = model.predict(X_test, num_iteration=model.best_iteration)


    print('MAPE for load data : ')
    temp_merged['GBDT_FV_Result'] = model.predict(X_test_cal)
    temp_merged['GBDT_FV_percentage_err'] = (temp_merged['GBDT_FV_Result']-temp_merged['final_usage'])/temp_merged['final_usage']
    temp_merged['GBDT_FV_abs_percentage_err'] = abs((temp_merged['GBDT_FV_Result']-temp_merged['final_usage'])/temp_merged['final_usage'])
    y_test['valid'] = model.predict(X_test)
    y_test['GBDT_FV_abs_percentage_err'] = abs((y_test['valid']-y_test['final_usage'])/y_test['final_usage'])
    y_test['GBDT_FV_RMSE'] = np.sqrt((y_test['valid']-y_test['final_usage'])**2)

    
    y_test_cal = temp_merged[['final_usage']]

    X_test_cal = column_selection(temp_merged)


    lgb_train = lgb.Dataset(X_test_cal, y_test_cal)
    # lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  
        'objective': 'regression', 
        'metric': {'mape'},  
        'num_leaves': 200,   # test with reduced value
        'learning_rate': 0.05,  
        'feature_fraction': 0.7, # test with reduced value
        'bagging_fraction': 0.7, 
        'bagging_freq': 50,  
        'verbose':1  
    }

    print('Start training...')
    model = lgb.train(params,lgb_train,num_boost_round=2000)


    print('Start predicting...')
    # y_hat = model.predict(X_test, num_iteration=model.best_iteration)

    
    
    key_GBDT = '/'+(SP_ID_List[0])+'_flat_GBDT.pickle'
    GBDT_model_save = save_model_to_s3(model,bucket,key_GBDT)
    gbm_pickle = read_joblib('s3://'+bucket+key_GBDT)
    temp_merged['GBDT_Result'] = gbm_pickle.predict(X_test_cal)

    temp_merged['GBDT_percentage_err'] = (temp_merged['GBDT_Result']-temp_merged['final_usage'])/temp_merged['final_usage']
    temp_merged['GBDT_abs_percentage_err'] = abs((temp_merged['GBDT_Result']-temp_merged['final_usage'])/temp_merged['final_usage'])
    temp_merged['GBDT_RMSE'] = np.sqrt((temp_merged['GBDT_Result']-temp_merged['final_usage'])**2)

    
    df = pd.DataFrame([])
    df = df.append(temp_merged.describe()['yhat_abs_percentage_err'][1:2])
    df = df.append(temp_merged.describe()['LR_abs_percentage_err'][1:2])
    df = df.append(temp_merged.describe()['LR_percentage_err'][1:2])
    df = df.append(temp_merged.describe()['LR_RMSE'][1:2])
    df = df.append(temp_merged.describe()['GBDT_RMSE'][1:2])
    df = df.append(temp_merged.describe()['GBDT_abs_percentage_err'][1:2])
    df = df.append(temp_merged.describe()['GBDT_percentage_err'][1:2])
    df = df.append(y_test.describe()['GBDT_FV_abs_percentage_err'][1:2])

    df = df.T
    df['feeder'] = SCE_SP['CircuitID'][i]
    # df['feeder_size'] = SCE_SP['id'][i]
    
    result_summary = result_summary.append(df)
#     result_summary.to_csv('summary_temp.csv')

# result_summary.to_csv('summary_test_V3.csv')

