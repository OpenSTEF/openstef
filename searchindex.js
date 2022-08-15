Search.setIndex({docnames:["code_modules","concepts","dashboard","index","modules","openstef","openstef.data_classes","openstef.feature_engineering","openstef.metrics","openstef.model","openstef.model.metamodels","openstef.model.regressors","openstef.model_selection","openstef.monitoring","openstef.pipeline","openstef.postprocessing","openstef.preprocessing","openstef.tasks","openstef.tasks.utils","openstef.validation","quickstart","tutorials"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.index":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,sphinx:56},filenames:["code_modules.rst","concepts.rst","dashboard.rst","index.rst","modules.rst","openstef.rst","openstef.data_classes.rst","openstef.feature_engineering.rst","openstef.metrics.rst","openstef.model.rst","openstef.model.metamodels.rst","openstef.model.regressors.rst","openstef.model_selection.rst","openstef.monitoring.rst","openstef.pipeline.rst","openstef.postprocessing.rst","openstef.preprocessing.rst","openstef.tasks.rst","openstef.tasks.utils.rst","openstef.validation.rst","quickstart.rst","tutorials.rst"],objects:{"":{openstef:[5,0,0,"-"]},"openstef.data_classes":{model_specifications:[6,0,0,"-"],prediction_job:[6,0,0,"-"],split_function:[6,0,0,"-"]},"openstef.data_classes.model_specifications":{ModelSpecificationDataClass:[6,1,1,""]},"openstef.data_classes.model_specifications.ModelSpecificationDataClass":{feature_modules:[6,2,1,""],feature_names:[6,2,1,""],hyper_params:[6,2,1,""],id:[6,2,1,""]},"openstef.data_classes.prediction_job":{PredictionJobDataClass:[6,1,1,""]},"openstef.data_classes.prediction_job.PredictionJobDataClass":{Config:[6,1,1,""],backtest_split_func:[6,2,1,""],completeness_treshold:[6,2,1,""],default_modelspecs:[6,2,1,""],depends_on:[6,2,1,""],description:[6,2,1,""],flatliner_treshold:[6,2,1,""],forecast_type:[6,2,1,""],get:[6,3,1,""],horizon_minutes:[6,2,1,""],hub_height:[6,2,1,""],id:[6,2,1,""],lat:[6,2,1,""],lon:[6,2,1,""],minimal_table_length:[6,2,1,""],model:[6,2,1,""],n_turbines:[6,2,1,""],name:[6,2,1,""],quantiles:[6,2,1,""],resolution_minutes:[6,2,1,""],save_train_forecasts:[6,2,1,""],sid:[6,2,1,""],train_components:[6,2,1,""],train_horizons_minutes:[6,2,1,""],train_split_func:[6,2,1,""],turbine_type:[6,2,1,""]},"openstef.data_classes.prediction_job.PredictionJobDataClass.Config":{smart_union:[6,2,1,""]},"openstef.data_classes.split_function":{SplitFuncDataClass:[6,1,1,""]},"openstef.data_classes.split_function.SplitFuncDataClass":{"function":[6,2,1,""],arguments:[6,2,1,""],load:[6,3,1,""]},"openstef.enums":{ForecastType:[5,1,1,""],MLModelType:[5,1,1,""],TracyJobResult:[5,1,1,""]},"openstef.enums.ForecastType":{BASECASE:[5,2,1,""],DEMAND:[5,2,1,""],SOLAR:[5,2,1,""],WIND:[5,2,1,""]},"openstef.enums.MLModelType":{LGB:[5,2,1,""],LINEAR:[5,2,1,""],ProLoaf:[5,2,1,""],XGB:[5,2,1,""],XGB_QUANTILE:[5,2,1,""]},"openstef.enums.TracyJobResult":{FAILED:[5,2,1,""],SUCCESS:[5,2,1,""],UNKNOWN:[5,2,1,""]},"openstef.exceptions":{InputDataInsufficientError:[5,4,1,""],InputDataInvalidError:[5,4,1,""],InputDataWrongColumnOrderError:[5,4,1,""],ModelWithoutStDev:[5,4,1,""],NoPredictedLoadError:[5,4,1,""],NoRealisedLoadError:[5,4,1,""],OldModelHigherScoreError:[5,4,1,""]},"openstef.feature_engineering":{apply_features:[7,0,0,"-"],feature_adder:[7,0,0,"-"],general:[7,0,0,"-"],historic_features:[7,0,0,"-"],holiday_features:[7,0,0,"-"],lag_features:[7,0,0,"-"],weather_features:[7,0,0,"-"]},"openstef.feature_engineering.apply_features":{apply_features:[7,5,1,""]},"openstef.feature_engineering.feature_adder":{FeatureAdder:[7,1,1,""],FeatureDispatcher:[7,1,1,""],ParsedFeature:[7,1,1,""],adders_from_module:[7,5,1,""],adders_from_modules:[7,5,1,""]},"openstef.feature_engineering.feature_adder.FeatureAdder":{apply_features:[7,3,1,""],name:[7,3,1,""],parse_feature_name:[7,3,1,""],required_features:[7,3,1,""]},"openstef.feature_engineering.feature_adder.FeatureDispatcher":{apply_features:[7,3,1,""],dispatch_features:[7,3,1,""]},"openstef.feature_engineering.feature_adder.ParsedFeature":{name:[7,2,1,""],params:[7,2,1,""]},"openstef.feature_engineering.general":{add_missing_feature_columns:[7,5,1,""],enforce_feature_order:[7,5,1,""],remove_non_requested_feature_columns:[7,5,1,""]},"openstef.feature_engineering.historic_features":{add_historic_load_as_a_feature:[7,5,1,""]},"openstef.feature_engineering.holiday_features":{check_for_bridge_day:[7,5,1,""],generate_holiday_feature_functions:[7,5,1,""]},"openstef.feature_engineering.lag_features":{extract_lag_features:[7,5,1,""],generate_lag_feature_functions:[7,5,1,""],generate_non_trivial_lag_times:[7,5,1,""],generate_trivial_lag_features:[7,5,1,""]},"openstef.feature_engineering.weather_features":{add_additional_solar_features:[7,5,1,""],add_additional_wind_features:[7,5,1,""],add_humidity_features:[7,5,1,""],calc_air_density:[7,5,1,""],calc_dewpoint:[7,5,1,""],calc_saturation_pressure:[7,5,1,""],calc_vapour_pressure:[7,5,1,""],calculate_dni:[7,5,1,""],calculate_gti:[7,5,1,""],calculate_windspeed_at_hubheight:[7,5,1,""],calculate_windturbine_power_output:[7,5,1,""],humidity_calculations:[7,5,1,""]},"openstef.metrics":{figure:[8,0,0,"-"],metrics:[8,0,0,"-"],reporter:[8,0,0,"-"]},"openstef.metrics.figure":{convert_to_base64_data_uri:[8,5,1,""],plot_data_series:[8,5,1,""],plot_feature_importance:[8,5,1,""]},"openstef.metrics.metrics":{bias:[8,5,1,""],frac_in_stdev:[8,5,1,""],franks_skill_score:[8,5,1,""],franks_skill_score_peaks:[8,5,1,""],get_eval_metric_function:[8,5,1,""],mae:[8,5,1,""],nsme:[8,5,1,""],r_mae:[8,5,1,""],r_mae_highest:[8,5,1,""],r_mae_lowest:[8,5,1,""],r_mne_highest:[8,5,1,""],r_mpe_highest:[8,5,1,""],rmse:[8,5,1,""],skill_score:[8,5,1,""],skill_score_positive_peaks:[8,5,1,""],xgb_quantile_eval:[8,5,1,""],xgb_quantile_obj:[8,5,1,""]},"openstef.metrics.reporter":{Report:[8,1,1,""],Reporter:[8,1,1,""]},"openstef.metrics.reporter.Reporter":{generate_report:[8,3,1,""],get_metrics:[8,3,1,""],write_report_to_disk:[8,3,1,""]},"openstef.model":{basecase:[9,0,0,"-"],confidence_interval_applicator:[9,0,0,"-"],fallback:[9,0,0,"-"],metamodels:[10,0,0,"-"],model_creator:[9,0,0,"-"],objective:[9,0,0,"-"],objective_creator:[9,0,0,"-"],regressors:[11,0,0,"-"],serializer:[9,0,0,"-"],standard_deviation_generator:[9,0,0,"-"]},"openstef.model.basecase":{BaseCaseModel:[9,1,1,""]},"openstef.model.basecase.BaseCaseModel":{can_predict_quantiles:[9,3,1,""],fit:[9,3,1,""],make_basecase_forecast:[9,3,1,""],predict:[9,3,1,""]},"openstef.model.confidence_interval_applicator":{ConfidenceIntervalApplicator:[9,1,1,""]},"openstef.model.confidence_interval_applicator.ConfidenceIntervalApplicator":{add_confidence_interval:[9,3,1,""]},"openstef.model.fallback":{generate_fallback:[9,5,1,""]},"openstef.model.metamodels":{grouped_regressor:[10,0,0,"-"],missing_values_handler:[10,0,0,"-"]},"openstef.model.metamodels.grouped_regressor":{GroupedRegressor:[10,1,1,""]},"openstef.model.metamodels.grouped_regressor.GroupedRegressor":{fit:[10,3,1,""],grouped_compute:[10,3,1,""],predict:[10,3,1,""]},"openstef.model.metamodels.missing_values_handler":{MissingValuesHandler:[10,1,1,""]},"openstef.model.metamodels.missing_values_handler.MissingValuesHandler":{fit:[10,3,1,""],predict:[10,3,1,""]},"openstef.model.model_creator":{ModelCreator:[9,1,1,""]},"openstef.model.model_creator.ModelCreator":{MODEL_CONSTRUCTORS:[9,2,1,""],create_model:[9,3,1,""]},"openstef.model.objective":{LGBRegressorObjective:[9,1,1,""],LinearRegressorObjective:[9,1,1,""],ProLoafRegressorObjective:[9,1,1,""],RegressorObjective:[9,1,1,""],XGBQuantileRegressorObjective:[9,1,1,""],XGBRegressorObjective:[9,1,1,""]},"openstef.model.objective.LGBRegressorObjective":{get_params:[9,3,1,""],get_pruning_callback:[9,3,1,""]},"openstef.model.objective.LinearRegressorObjective":{get_params:[9,3,1,""]},"openstef.model.objective.ProLoafRegressorObjective":{get_params:[9,3,1,""],get_pruning_callback:[9,3,1,""]},"openstef.model.objective.RegressorObjective":{create_report:[9,3,1,""],get_default_values:[9,3,1,""],get_params:[9,3,1,""],get_pruning_callback:[9,3,1,""],get_trial_track:[9,3,1,""]},"openstef.model.objective.XGBQuantileRegressorObjective":{get_params:[9,3,1,""],get_pruning_callback:[9,3,1,""]},"openstef.model.objective.XGBRegressorObjective":{get_default_values:[9,3,1,""],get_params:[9,3,1,""],get_pruning_callback:[9,3,1,""]},"openstef.model.objective_creator":{ObjectiveCreator:[9,1,1,""]},"openstef.model.objective_creator.ObjectiveCreator":{OBJECTIVES:[9,2,1,""],create_objective:[9,3,1,""]},"openstef.model.regressors":{custom_regressor:[11,0,0,"-"],lgbm:[11,0,0,"-"],linear:[11,0,0,"-"],regressor:[11,0,0,"-"],xgb:[11,0,0,"-"],xgb_quantile:[11,0,0,"-"]},"openstef.model.regressors.custom_regressor":{CustomOpenstfRegressor:[11,1,1,""],create_custom_objective:[11,5,1,""],is_custom_type:[11,5,1,""],load_custom_model:[11,5,1,""]},"openstef.model.regressors.custom_regressor.CustomOpenstfRegressor":{objective:[11,3,1,""],valid_kwargs:[11,3,1,""]},"openstef.model.regressors.lgbm":{LGBMOpenstfRegressor:[11,1,1,""]},"openstef.model.regressors.lgbm.LGBMOpenstfRegressor":{can_predict_quantiles:[11,3,1,""],feature_names:[11,3,1,""],gain_importance_name:[11,2,1,""],weight_importance_name:[11,2,1,""]},"openstef.model.regressors.linear":{LinearOpenstfRegressor:[11,1,1,""],LinearRegressor:[11,1,1,""]},"openstef.model.regressors.linear.LinearOpenstfRegressor":{can_predict_quantiles:[11,3,1,""],feature_names:[11,3,1,""],fit:[11,3,1,""]},"openstef.model.regressors.regressor":{OpenstfRegressor:[11,1,1,""]},"openstef.model.regressors.regressor.OpenstfRegressor":{can_predict_quantiles:[11,3,1,""],feature_names:[11,3,1,""],fit:[11,3,1,""],predict:[11,3,1,""],score:[11,3,1,""],set_feature_importance:[11,3,1,""]},"openstef.model.regressors.xgb":{XGBOpenstfRegressor:[11,1,1,""]},"openstef.model.regressors.xgb.XGBOpenstfRegressor":{can_predict_quantiles:[11,3,1,""],feature_names:[11,3,1,""],gain_importance_name:[11,2,1,""],weight_importance_name:[11,2,1,""]},"openstef.model.regressors.xgb_quantile":{XGBQuantileOpenstfRegressor:[11,1,1,""]},"openstef.model.regressors.xgb_quantile.XGBQuantileOpenstfRegressor":{can_predict_quantiles:[11,3,1,""],feature_names:[11,3,1,""],fit:[11,3,1,""],get_feature_importances_from_booster:[11,3,1,""],predict:[11,3,1,""]},"openstef.model.serializer":{MLflowSerializer:[9,1,1,""]},"openstef.model.serializer.MLflowSerializer":{get_model_age:[9,3,1,""],load_model:[9,3,1,""],remove_old_models:[9,3,1,""],save_model:[9,3,1,""]},"openstef.model.standard_deviation_generator":{StandardDeviationGenerator:[9,1,1,""]},"openstef.model.standard_deviation_generator.StandardDeviationGenerator":{generate_standard_deviation_data:[9,3,1,""]},"openstef.model_selection":{model_selection:[12,0,0,"-"]},"openstef.model_selection.model_selection":{backtest_split_default:[12,5,1,""],group_kfold:[12,5,1,""],random_sample:[12,5,1,""],sample_indices_train_val:[12,5,1,""],split_data_train_validation_test:[12,5,1,""]},"openstef.monitoring":{performance_meter:[13,0,0,"-"],teams:[13,0,0,"-"]},"openstef.monitoring.performance_meter":{PerformanceMeter:[13,1,1,""]},"openstef.monitoring.performance_meter.PerformanceMeter":{checkpoint:[13,3,1,""],complete_level:[13,3,1,""],start_level:[13,3,1,""]},"openstef.monitoring.teams":{build_sql_query_string:[13,5,1,""],format_message:[13,5,1,""],get_card_section:[13,5,1,""],post_teams:[13,5,1,""]},"openstef.pipeline":{create_component_forecast:[14,0,0,"-"],utils:[14,0,0,"-"]},"openstef.pipeline.create_component_forecast":{create_components_forecast_pipeline:[14,5,1,""]},"openstef.pipeline.utils":{generate_forecast_datetime_range:[14,5,1,""]},"openstef.postprocessing":{postprocessing:[15,0,0,"-"]},"openstef.postprocessing.postprocessing":{add_components_base_case_forecast:[15,5,1,""],add_prediction_job_properties_to_forecast:[15,5,1,""],calculate_wind_power:[15,5,1,""],normalize_and_convert_weather_data_for_splitting:[15,5,1,""],post_process_wind_solar:[15,5,1,""],split_forecast_in_components:[15,5,1,""]},"openstef.preprocessing":{preprocessing:[16,0,0,"-"]},"openstef.preprocessing.preprocessing":{replace_repeated_values_with_nan:[16,5,1,""]},"openstef.tasks":{calculate_kpi:[17,0,0,"-"],create_components_forecast:[17,0,0,"-"],create_solar_forecast:[17,0,0,"-"],create_wind_forecast:[17,0,0,"-"],split_forecast:[17,0,0,"-"],utils:[18,0,0,"-"]},"openstef.tasks.calculate_kpi":{calc_kpi_for_specific_pid:[17,5,1,""],check_kpi_task:[17,5,1,""],main:[17,5,1,""],set_incomplete_kpi_to_nan:[17,5,1,""]},"openstef.tasks.create_components_forecast":{create_components_forecast_task:[17,5,1,""],main:[17,5,1,""]},"openstef.tasks.create_solar_forecast":{apply_fit_insol:[17,5,1,""],apply_persistence:[17,5,1,""],calc_norm:[17,5,1,""],combine_forecasts:[17,5,1,""],fides:[17,5,1,""],main:[17,5,1,""],make_solar_prediction_pj:[17,5,1,""]},"openstef.tasks.create_wind_forecast":{main:[17,5,1,""],make_wind_forecast_pj:[17,5,1,""]},"openstef.tasks.split_forecast":{convert_coefdict_to_coefsdf:[17,5,1,""],determine_invalid_coefs:[17,5,1,""],find_components:[17,5,1,""],main:[17,5,1,""],split_forecast_task:[17,5,1,""]},"openstef.tasks.utils":{dependencies:[18,0,0,"-"],predictionjobloop:[18,0,0,"-"],taskcontext:[18,0,0,"-"]},"openstef.tasks.utils.dependencies":{build_graph_structure:[18,5,1,""],build_nx_graph:[18,5,1,""],find_groups:[18,5,1,""],has_dependencies:[18,5,1,""]},"openstef.tasks.utils.predictionjobloop":{PredictionJobException:[18,4,1,""],PredictionJobLoop:[18,1,1,""]},"openstef.tasks.utils.predictionjobloop.PredictionJobLoop":{map:[18,3,1,""]},"openstef.tasks.utils.taskcontext":{TaskContext:[18,1,1,""]},"openstef.validation":{validation:[19,0,0,"-"]},"openstef.validation.validation":{calc_completeness:[19,5,1,""],check_data_for_each_trafo:[19,5,1,""],drop_target_na:[19,5,1,""],find_nonzero_flatliner:[19,5,1,""],find_zero_flatliner:[19,5,1,""],is_data_sufficient:[19,5,1,""],validate:[19,5,1,""]},openstef:{data_classes:[6,0,0,"-"],enums:[5,0,0,"-"],exceptions:[5,0,0,"-"],feature_engineering:[7,0,0,"-"],metrics:[8,0,0,"-"],model:[9,0,0,"-"],model_selection:[12,0,0,"-"],monitoring:[13,0,0,"-"],pipeline:[14,0,0,"-"],postprocessing:[15,0,0,"-"],preprocessing:[16,0,0,"-"],tasks:[17,0,0,"-"],validation:[19,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","exception","Python exception"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:exception","5":"py:function"},terms:{"06769d701c1d9c9acb9a66f2f9d7a6c7":8,"100m":7,"10m":7,"14d":9,"15t":[7,17],"1mwp":[7,15],"24h":17,"2nd":17,"abstract":[7,11],"case":[7,8,11,15],"class":[5,6,7,8,9,10,11,12,13,18],"default":[6,7,8,9,10,11,12,13,15],"enum":[0,4,9,15,17],"final":[7,10],"float":[6,7,8,9,10,11,12,19],"function":[6,7,8,9,10,11,12,13,14,15,17,18,19,21],"import":[7,8,10,11,15,17,19],"int":[5,6,7,8,9,10,11,12,16,17,18,19],"new":[1,5,7,13,17],"null":14,"public":3,"return":[6,7,8,9,10,11,12,13,14,15,16,17,18,19],"short":3,"static":[8,9,11],"true":[6,8,12,13,17,18,19],AWS:1,But:7,For:[7,10,11,12,13],One:13,THe:9,The:[1,7,8,9,10,11,12,13,14,15,17,18,19],There:8,These:[7,17],Use:9,Using:7,_figur:8,_frozen:9,abc:7,about:[3,7,8,9],abov:[16,17,19],abs:17,absolut:[8,17],abstractmethod:11,accept:12,accord:[7,8,12],accuraci:7,achiev:17,action:[1,13],actual:17,add:[7,9,17],add_additional_solar_featur:7,add_additional_wind_featur:7,add_components_base_case_forecast:15,add_confidence_interv:9,add_historic_load_as_a_featur:7,add_humidity_featur:7,add_missing_feature_column:7,add_prediction_job_properties_to_forecast:15,add_to_df:17,added:[7,9],adder:7,adders_from_modul:7,addit:[7,17],addtodf:17,adher:11,adit:7,after:[11,19],age:9,agenda:7,aggreg:10,ago:7,ahead:17,air:7,algorithm1:17,algorithm_typ:15,algorithmn:17,algtyp:14,alia:7,all:[1,3,7,9,10,11,17,18,19],all_forecast:17,all_peak:12,allforecast:17,allow:6,along:[10,11],alpha:11,alphabet:7,alreadi:18,also:[7,8,9,13],altern:17,alwai:15,amount:[8,19],ani:[6,7,9,11,13,17,18],anyth:19,api:[1,11],appear:19,append:[7,13],appli:[7,10,12,17,18],applic:21,apply_featur:[0,4,5],apply_fit_insol:17,apply_persist:17,apx:17,arg:[9,15,18],argument:[6,9,12,18],arrai:[7,8,10,11,12,19],artifact_fold:9,assign:[7,12],associ:7,assum:[7,12,15,17],assumpt:9,atmospher:7,attribut:[6,10,11,17],authent:13,autocorrel:7,autom:11,automat:[7,8,15,18],avail:[3,9,11,15,17,19,21],avoid:12,back:[9,12],back_test:12,backend:10,backtest_split_default:12,backtest_split_func:6,base:[1,5,6,7,8,9,10,11,13,14,15,17,18],base_estim:10,basecas:[0,4,5,8,15,17],basecase_forecast:15,basecasemodel:9,baseestim:[9,10,11],basemodel:6,basic:7,becaus:8,been:[11,18],befor:[9,19],between:[7,12,14,17,18,19],bia:[8,17],big:19,blob:[11,21],block:[7,19],bool:[6,9,11,12,13,17,18,19],booster:11,boosting_typ:11,bouwvak:7,bridge_dai:7,bridgedai:7,brugdagen:7,build:[1,8,13,18],build_graph_structur:18,build_nx_graph:18,build_sql_query_str:13,built:12,button:13,calc_air_dens:7,calc_complet:19,calc_dewpoint:7,calc_kpi_for_specific_pid:17,calc_norm:17,calc_saturation_pressur:7,calc_vapour_pressur:7,calcul:[7,8,15,17,19],calculate_dni:7,calculate_gti:7,calculate_kpi:[0,4,5],calculate_wind_pow:15,calculate_windspeed_at_hubheight:7,calculate_windturbine_power_output:7,call:[1,9,17],callabl:[6,10,11,18],can:[1,8,10,11,12,13,15,17,18,19,21],can_predict_quantil:[9,11],card:13,cardsect:13,cari:[15,17],carri:17,certain:1,chain:10,charg:7,check:[1,7,17,19],check_data_for_each_trafo:19,check_for_bridge_dai:7,check_kpi_task:17,checkpoint:13,choos:1,cicd:1,clarif:17,class_weight:11,classmethod:[9,10,11],clean:[12,19],cloud:12,cluster:14,code:[3,17],code_of_conduct:3,coef:15,coefdict:17,coeffici:[13,15,17],coefficients_df:13,coefici:17,coefieicnt:14,col:19,collect:7,colnam:17,color:13,colsample_bytre:11,colum:17,column:[5,7,8,9,10,11,14,15,16,17,19],column_nam:16,com:[7,8,9,11,13,21],comat:9,combin:[1,17],combination_coef:17,combinationcoef:17,combine_forecast:17,common:11,compat:[9,19],compens:19,complet:[7,10,13,17,19],complete_level:13,completeness_threshold:19,completeness_treshold:6,complex:13,compon:[3,14,15,17],comput:[1,7,8,10,18],concept:3,concret:11,concurr:10,condit:8,confid:9,confidence_interval_appl:[0,4,5],confidencegener:9,confidenceintervalappl:9,config:[1,6,17,18],configur:[1,9],connect:17,connector:13,conrrespond:10,consid:19,consist:7,constant:[10,11],construct:9,constructor:9,consumpt:[15,17],contain:[1,3,7,8,10,13,14,17,19],contect:17,content:[0,4],content_typ:8,context:[17,18],continu:19,contribut:17,convert:[8,10,11,15,17],convert_coefdict_to_coefsdf:17,convert_to_base64_data_uri:8,copi:12,core:[7,8,9,10,11,12,13,14,15,16,17,19],correct:[7,19],correctli:15,correspond:[1,10,12],could:[13,19],count:19,countri:7,creat:[1,7,8,9,13,14,17],create_basecase_forecast:[0,4,5],create_component_forecast:[0,4,5],create_components_forecast:[0,4,5],create_components_forecast_pipelin:14,create_components_forecast_task:17,create_custom_object:11,create_forecast:[0,4,5],create_model:9,create_object:9,create_report:9,create_solar_forecast:[0,4,5],create_wind_forecast:[0,4,5],creation:7,cron:[1,17],cronjob:17,cross:12,csv:7,current:[1,9,13],curv:[7,15],custom:[5,8,14,17],custom_model_path:11,custom_regressor:[5,9],customopenstfregressor:11,dai:[7,9,12,17],daili:9,dashboard:[1,21],data:[1,3,5,7,8,9,10,11,12,14,15,16,17,19,21],data_:12,data_class:[0,4,5,7,9,14,15,17,18],data_series_figur:8,data_uri:8,databas:[1,17,18,21],datafram:[1,7,8,9,10,11,12,13,14,15,16,17,19],dataframegroupbi:10,dataset:[7,12,14,17],datatim:[5,17],date:[7,12],date_rang:[7,17],datetim:[5,7,9,14,15,17,19],datetimeindex:[7,19],datfram:13,days_list:7,dbc:1,debug_pid:18,dec:7,default_modelspec:6,defenit:9,defin:[7,10,11],definit:11,degener:8,degre:7,delta:19,demand:5,densiti:7,depend:[5,9,17],depends_on:6,deploi:1,deriv:[7,8,9],deriven:7,describ:[7,8,15],descript:[6,14,17,19],desir:[9,11,12,16],detail:3,detect:[7,15,19],determin:[7,9,14,15,17],determine_invalid_coef:17,develop:8,deviat:[5,9],dew_point:7,dewpoint:7,df_re:10,dict:[6,7,8,9,10,11,13,14,15,17],dictionari:[7,8,9,13,17],dictionnari:[7,10],dictonari:17,differ:[12,19],digraph:18,direct:[7,17,18],directli:[1,17],disk:[1,8],dispatch_featur:7,distribut:[8,9],dmatrix:8,dmlc:[8,11],dni:7,dni_convert:7,doc:[8,13],document:[7,21],doe:[7,17,18],domest:17,don:7,drop_target_na:19,dto:7,dtype:[10,11],ducth:7,due:12,dure:[7,9,18,19],dutch_holidays_2020:7,each:[7,8,10,11,12,13,15,17,18,19],earlier:14,east:7,edg:18,edgetyp:18,effect:[7,19],effici:[8,17],either:[8,15],els:9,empti:7,end:[14,19],end_tim:17,enddat:17,energi:[1,3,15,17],enforc:7,enforce_feature_ord:7,engin:[1,7],enhanc:13,enough:[8,19],ensur:[12,15],enter:13,entri:17,enumer:5,equal:[7,8,15,17,19],error:[7,8,9,17],especi:7,estim:[8,9,10,12,15],estimators_:10,eval_metr:[9,11],eval_set:10,evalu:[1,8],evenli:12,eventu:7,everi:17,everyth:7,exactli:17,exampl:[7,9,10,14,17],except:[0,1,4,18],exclus:10,execut:[17,19],exist:[7,8],expect:17,experi:9,experiment_nam:9,extra:[1,7,15],extract_lag_featur:7,extrapol:7,extrem:[9,12],extreme_dai:9,face:7,fact:13,factori:9,fail:[5,9],failsav:7,fall:9,fallback:[0,4,5,13],fallback_strategi:9,fals:[9,11,12,17,18,19],featur:[1,7,8,10,11,12,13,17],feature_1:7,feature_add:[0,4,5],feature_appl:[0,4,5],feature_engin:[0,4,5],feature_funct:7,feature_import:8,feature_importance_:11,feature_importance_figur:8,feature_importances_:10,feature_m:7,feature_modul:6,feature_nam:[6,7,10,11],feature_names_:10,featureadd:7,featuredispatch:7,featurefunct:7,featurelist:7,feautur:11,fetch:1,fetcher:1,fide:17,field:[7,10],figur:[0,4,5,17],file:[7,8],fill:10,fill_valu:[10,11],filter:7,find:18,find_compon:17,find_group:18,find_nonzero_flatlin:19,find_zero_flatlin:19,first:[7,8,14,17,19],fit:[9,10,11,17],flatlin:[1,19],flatliner_load_threshold:19,flatliner_threshold:19,flatliner_treshold:6,flatliner_window:19,fly:7,fold:12,folder:17,follow:[7,12,13,17,18],folow:17,forecaopenstefitinsol:17,forecast:[1,3,7,8,9,14,15,17,21],forecast_data:14,forecast_input:9,forecast_input_data:9,forecast_oth:[14,15],forecast_qu:15,forecast_solar:14,forecast_typ:[6,15],forecast_wind_on_shor:14,forecasttyp:[5,15],form:7,format_messag:13,found:[5,7,17,18,21],frac_in_stdev:8,fraction:[12,19],frame:[7,8,9,10,11,12,13,14,15,16,17,19],franks_skill_scor:8,franks_skill_score_peak:8,freq:[7,17],frequent:[10,11],from:[1,6,7,9,10,11,12,13,15,17],fromheight:7,frozentri:9,fulli:[9,19],func:10,funciton:7,furthermor:7,futur:9,gain:11,gain_importance_nam:11,gamma:11,gbdt:11,gener:[0,4,5,8,9,10,12,14,15],generate_fallback:9,generate_forecast_datetime_rang:14,generate_holiday_feature_funct:7,generate_lag_feature_funct:7,generate_lag_funct:7,generate_non_trivial_lag_tim:7,generate_report:8,generate_standard_deviation_data:9,generate_trivial_lag_featur:7,get:[1,3,6,9,11,12,13,17],get_card_sect:13,get_default_valu:9,get_eval_metric_funct:8,get_feature_importances_from_boost:11,get_metr:8,get_model_ag:9,get_param:9,get_pruning_callback:9,get_trial_track:9,gist:8,gistcom:8,github:[1,7,8,9,11,21],give:[1,9],given:[5,6,7,8,12,13,14,15,17,18],glmd:15,global:[7,10],goe:17,grafana:8,graph:18,graph_obj:8,graph_object:8,grid:1,group:[9,10,12,18],group_column:10,group_kfold:12,group_r:10,groupbi:10,grouped_comput:10,grouped_regressor:[5,9],groupedregressor:10,gti:7,h_ahead:7,handl:[7,10,11],has:[5,7,11,14],has_depend:18,have:[5,6,7,11,12,17,18],header:8,height:7,height_treshold:7,help:1,hemelvaart:7,herfstvakantienoord:7,herstvakanti:7,hessian:8,higher:5,highest:8,highli:18,histor:[1,7,9,17],historic_featur:[0,4,5],historic_load:7,hold:17,holidai:7,holiday_featur:[0,4,5],holiday_funct:7,holiday_nam:7,holidayfunciton:7,homogenis:19,horizon:[7,8,9],horizon_minut:6,hour:[7,17,19],hours_delta:17,hoursdelta:17,household:17,how:[8,17,19,21],howev:[8,9],hpa:7,http:[3,7,8,9,11,13,21],hub:7,hub_height:[6,7],hubheigh:7,hubheight:7,humid:7,humidity_calcul:7,humidity_conversion_formulas_b210973en:7,humidti:7,hyper_param:6,hyperparamet:8,hyperparameter_optimization_onli:9,hyperparameter_valu:9,idea:9,ideal:7,identifi:19,ids:18,ignor:[17,19],imag:13,implement:[7,9,11,12],importance_typ:11,improv:[7,19],imput:[10,11],imputation_strategi:[10,11],imputer_:10,includ:[1,7,9,12,17,21],incomplet:19,incorpor:9,ind:19,independ:17,index:[7,8,9,10,15,17,19],indic:[7,8,11,12,14,17,19],inf:17,influx:1,info:[7,8],inform:[1,7,13,18],initi:9,inner:13,input:[1,5,7,9,10,11,12,14,16,17,19],input_data:[7,9,11,12,14],input_split_funct:17,inputdatainsufficienterror:5,inputdatainvaliderror:5,inputdatawrongcolumnordererror:5,insert:[13,17],insid:9,insol:17,instan:6,instanc:[6,9,10],instead:8,insuffici:5,integ:[10,11],intend:[7,12],interfac:[1,3,7,11,21],interv:9,invalid:[5,9,13,17],invalid_coeffici:13,irradi:7,is_custom_typ:11,is_data_suffici:19,issu:[8,9],iter:[12,18],its:[6,7],itself:1,jenkin:1,job:[1,7,9,10,14,15,17,18,19],joblib:10,jupyt:21,just:[1,6,19],k8s:17,keep:13,kei:[6,7,10,13,17],kerst:7,kerstvakanti:7,keyword:[9,11,18],knmi:17,known:8,koningsdag:7,kpi:17,ktp:[13,15],kubernet:1,kwarg:[9,10,11,13,18],label:[7,11,13,19],lag:7,lag_featur:[0,4,5],lag_funct:7,lagtim:7,largest:15,last:[1,7,9,14,17],last_coef:17,lat:[6,7],later:7,latest:8,latitud:7,launch:7,law:7,lazi:12,lc_:19,learn:[1,3,5,9],learning_r:11,least:[7,15,18],left:[10,11,19],legend:8,len:[9,17],length:[12,16,19],level:[13,17],level_label:13,level_nam:13,lgb:[5,9],lgbm:[5,9],lgbmopenstfregressor:[9,11],lgbmregressor:11,lgbregressorobject:9,librari:7,lightgbm:11,like:7,limit:7,line:8,linear:[5,9,17],linearopenstfregressor:[9,11],linearregressor:11,linearregressorobject:9,link:13,list:[6,7,8,10,11,12,18,19],load1:19,load:[1,5,6,7,8,9,14,15,17,19],load_correct:19,load_custom_model:11,load_model:9,loadn:19,loc:17,local:1,locat:[1,7,17],log:[13,18,19],logger:13,logic:[7,19],lon:[6,7],longer:19,longitud:7,look:[1,7],loop:18,loss:8,lot:19,lowest:8,lysenko:8,machin:[1,3,5,9],mae:[8,9,11],mai:[7,19],main:[6,17,21],make:[3,7,9,11,15,17],make_basecase_forecast:9,make_solar_prediction_pj:17,make_wind_forecast_pj:17,manag:17,mani:19,manual:7,map:18,markdown:13,master:11,match:8,matrix:11,max:[12,17,19],max_delta_step:[8,11],max_depth:11,max_length:16,max_n_model:9,maximum:[10,16],mean:[8,9,10,11,17,18,19],meant:17,measur:[8,19],median:[10,11,19],meivakanti:7,messag:[5,13,15,17],meta:10,metaestimatormixin:10,metamodel:[5,9,11],meter:13,method:[7,9,10,11,18],metric:[0,4,5,9,17,18],metric_nam:8,microsoft:13,min:[8,12],min_child_sampl:11,min_child_weight:11,min_split_gain:11,minim:[7,19],minimal_table_length:[6,19],minut:7,minute_list:7,minutes_list:7,miss:[10,11,19],missig:10,missing_valu:[10,11],missing_values_handl:[5,9,11],missingvalueshandl:[10,11],mix:[8,13],mixs:8,mlflow:[1,8,9],mlflow_tracking_uri:9,mlflowseri:9,mlmodeltyp:[5,9,17],mnt:19,model:[0,1,4,5,6,7,8,17,19],model_constructor:9,model_cr:[0,4,5],model_select:[0,4,5],model_spec:9,model_specif:[0,4,5,9],model_train:17,model_typ:[9,11,17],modelcr:9,modelsignatur:8,modelspecificationdataclass:[6,9],modelwithoutstdev:5,modul:[3,4],module_nam:7,moistur:7,moment:19,monitor:[0,4,5],more:[1,7,9,10,11,13,18],moreov:10,most:[9,10,11,13],most_frequ:[10,11],mostli:7,move:7,mozilla:[3,8],mroe:7,msg:13,mtrand:11,much:8,mulitpl:17,multipl:12,multiprocess:10,must:[6,12],mysql:1,n_estim:11,n_features_in_:10,n_fold:12,n_job:[10,11],n_turbin:[6,7],name:[6,7,8,10,11,13,16,18],name_checkpoint:13,namespac:13,nan:[7,10,11,16,17,19],nash:8,nativ:9,ndarrai:[7,8,10,11],ndimag:19,necessari:10,nederland:7,need:[1,3,12,14],neg:[8,15,17],nescarri:7,nescesarri:7,networkx:18,new_coef:17,newli:13,next:9,nieuwjaarsdag:7,nikolai:8,node:18,nodeidtyp:18,nodel:7,non:[7,8,9],non_null_columns_:10,none:[6,7,8,9,10,11,13,15,17,18,19],nopredictedloaderror:[5,17],norealisedloaderror:[5,17],norm:17,normal:[7,9,11,15],normalis:7,normalize_and_convert_weather_data_for_split:15,north:7,note:[7,9,11,12,13,19],notebook:21,notimplementederror:9,nsme:[8,17],nturbin:7,nullabl:[10,11],num_leav:11,number:[7,8,10,12,17],numer:[7,10,11],numpi:[7,8,10,11,12,17,19],object:[0,4,5,6,7,8,10,11,13,17,18,19],objective_cr:[0,4,5],objectivecr:9,observ:19,obtain:7,occur:13,occurr:[10,11],offici:7,offlin:21,old:[5,9],oldmodelhigherscoreerror:5,omit:7,on_end:18,on_end_callback:18,on_except:18,on_exception_callback:18,on_success:18,on_successful_callback:18,onc:7,one:[8,9,10,11,13,17,18],onli:[7,8,9,10,11,17,19],open:1,openstef:[0,1,20,21],openstf:11,openstfregressor:[8,9,11],oper:[10,12],optim:9,optimize_hyperparamet:[0,4,5],option:[6,7,8,9,11,13,17,18,19],optuna:9,orchestr:7,order:[3,5,7,8,12,17,18],org:[3,7,8],other:[1,13,15,18,19],otherwis:[6,7,10,19],our:8,out:[7,12,15,17],outlook:13,output:[7,13,17],over:[17,18],overestim:8,overwrite_delay_hour:9,own:3,packag:[0,1,3,4],page:8,panda:[7,8,9,10,11,12,13,14,15,16,17,19],parallel:[10,18],param1:17,param:[7,9,11,12,13],paramet:[6,7,8,9,10,11,12,13,14,15,16,17,18,19],paramn:17,pars:7,parse_feature_nam:7,parsed_feature_nam:7,parsedfeatur:7,part:1,particular:11,pasen:7,pass:[8,9,13,18],path:8,path_in:8,path_out:8,path_to_school_holidays_csv:7,pdf:7,peak:[7,8,12],peak_pow:17,per:[9,12,17],percent:8,percentil:8,perform:[1,8,10,11,13,17,21],performance_met:[0,4,5],performancemet:13,period:[7,12,17],persist:17,phase:9,photovolta:7,pid:[5,14,17],pinbal:8,pinksteren:7,pip:20,pipelin:[0,1,3,4,5,10,11],pipeline_:10,pj_id:19,pj_kwarg:18,pjs:18,place:7,placehold:[10,11],plane:7,pleas:3,plot:8,plot_data_seri:8,plot_feature_import:8,plotli:8,poa:7,point:[12,15],polynomi:17,pool:10,posit:[7,8,15],posixpath:7,possibl:8,post:[1,13,15],post_process_wind_solar:15,post_team:13,post_teams_on_except:18,postprocess:[0,4,5],potenti:[7,16],power:[7,15],precis:9,pred:8,predetermin:17,predicion:7,prediciton:[7,8,11],predict:[1,5,7,8,9,10,11,14,15,17,18,19],predict_data:8,predicted_load:17,prediction_job:[0,4,5,7,9,14,15,17,18],predictionjobdataclass:[6,7,9,14,15,17,18],predictionjobexcept:18,predictionjobloop:[5,17],predictjob:7,predictor_1:7,predictor_n:7,prefer:18,preprocess:[0,4,5],present:[7,9],pressent:7,pressur:7,prevent:7,previou:[8,13,18],previous:15,price:[1,17],process:[1,3,10,15],produc:15,product:15,profil:[9,17,21],prognos:17,project:3,project_govern:3,proloaf:[5,7,9],proloafregressorobject:9,proper:7,properli:7,properti:[7,9,11],provid:[1,3,7,17],proxi:13,psat:7,pull:3,pv_ref:17,pvlib:7,pydant:6,pymsteam:13,pypi:[1,3],python:[3,6,7,10,11,17,20],qualiti:9,quantifi:8,quantil:[1,6,8,9,11],queri:13,quickstart:3,r_mae:8,r_mae_highest:8,r_mae_lowest:8,r_mne_highest:8,r_mpe_highest:8,radiat:[7,14,15],radiu:17,rais:[1,9,11,17,18],random:[7,11,12,17,18],random_ord:18,random_sampl:12,random_st:11,randomize_fold_split:12,randomize_group:18,randomst:11,rang:[5,8,14,17],range_:8,rate:7,rated_pow:7,ratio:12,read:[1,3,8],readm:21,realis:[5,8,17],reason:8,recent:[7,9],recogn:[7,17],refer:[8,17,21],refrain:13,reg:11,reg_alpha:11,reg_lambda:11,regress:[8,9],regressor:[5,8,9,10],regressor_:10,regressormixin:[9,10,11],regressorobject:9,regular:9,rel:[7,8],relat:[7,15],releas:1,relev:[7,15],remain:7,remov:[7,9,10],remove_non_requested_feature_column:7,remove_old_model:9,repeat:[16,19],repetit:19,replac:[10,11,16,19],replace_repeated_values_with_nan:16,report:[0,4,5,9],report_fold:8,repositori:1,repres:12,request:[3,7,11],requested_featur:7,requier:7,requir:[1,3,7,8,9,11],required_argu:6,required_featur:7,resampl:19,resolution_minut:6,resolv:17,respect:[7,18],rest:1,result:[7,10,17],retriev:[3,11,15,17],reuqest:7,right:7,rmae:17,rmse:[8,17],root:8,row:[10,19],rtype:8,rubric:17,run:[1,10,17,18,20,21],run_traci:[0,4,5],runtim:13,s3_bucket:1,same:7,sampl:12,sample_indices_train_v:12,satisfi:8,satur:7,save:[9,17],save_model:9,save_train_forecast:6,scan:7,schedul:1,schoolvakanti:7,scipi:19,score:[5,8,11],script:[7,17],second:8,secret:13,section:13,section_dict:13,sector:3,secur:[12,13],see:[3,7,8,9,13],select:[1,8,12],self:13,send:[13,15,17],separ:18,sequenc:[6,7,12,16,18],sequenti:16,seri:[7,8,15,17,19],serial:[0,4,5],serv:7,set:[1,7,8,9,10,11,12,14,15,16,17,18,19],set_feature_import:11,set_incomplete_kpi_to_nan:17,setup:21,sever:17,should:[5,7,8,10,11,13,17,18,19],show:21,sid:6,side:15,sign:15,signatur:8,silent:11,similar:[6,8,16],simpl:[13,15],simpleimput:10,sin:[7,17],sinc:[10,11,18],site:7,size:[10,19],skill:8,skill_scor:8,skill_score_positive_peak:8,skip:19,sklearn:[9,10,11],slope_cent:7,small:8,smallest:[10,11],smart_union:6,smooth:17,smooth_entri:17,smoothentri:17,solar:[1,5,7,15,17],some:[9,18],someth:17,sonar:12,sort:[7,12],sourc:[1,17],south:7,specif:[7,9,11,13,17],specifi:[8,10,13,15,17,19],speed:15,split:[1,11,12,14,15,17],split_coef:[14,15],split_data_train_validation_test:12,split_forecast:[0,4,5],split_forecast_in_compon:15,split_forecast_task:17,split_funct:[0,4,5],splitenergi:17,splitfuncdataclass:6,spread:12,sql:[1,13,17],squar:8,squarederror:11,standard:[5,9,17],standard_deviation_gener:[0,4,5],standarddeviationgener:9,start:[3,7,9,12,14,17,19],start_level:13,start_tim:17,statement:[13,19],station:19,stationflatlin:19,stdev:[8,9],steep:7,step:[17,19],still:9,stock:10,stop_on_except:18,storag:[1,3,17],store:1,str:[5,6,7,8,9,10,11,13,15,16,17,18,19],strategi:[9,10,11,12],stratif:12,stratification_min_max:12,string:[6,7,8,10,11,13,16],studi:9,subgroup:18,submit:3,submodul:[0,4],subpackag:[0,4],subsampl:11,subsample_for_bin:11,subsample_freq:11,subsequ:7,substitut:[8,9],success:[5,13],suffici:19,sugar:7,suggest:19,sum:15,support:13,suppress_except:18,sure:[7,8],surfac:7,surface_azimuth:7,surface_tilt:7,sutcliff:8,syntact:7,system:7,t_ahead:17,t_ahead_h:17,tabl:[13,19],take:10,taken:7,target:[8,10,14,19],task:[0,1,4,5,13,20],task_nam:20,taskcontext:[5,17],tdcv:17,team:[0,4,5,17],tekton:1,temperatur:7,tennet:15,term:3,test:[1,12,18],test_data:[8,12],test_fract:[9,11,12],text:13,than:[10,11,19],thank:10,the_name_of_the_holiday_to_be_check:7,them:[6,17],therefor:[13,14],thi:[3,7,8,9,10,11,13,14,15,17,18,19],thise:8,thread:10,threshold:19,through:[7,15],till:17,tilt:7,time:[7,9,11,13,14,17,19],time_delai:19,timedelta:19,timeseri:19,timestamp:19,timestep:19,timezon:7,titl:13,todo:19,top:17,total:[13,19],total_gain:11,trace:7,tracy_todo:13,tracyjobresult:5,trafo:19,train:[1,7,8,9,10,11,12,17,19,21],train_compon:6,train_create_forecast_backtest:[0,4,5],train_data:[8,12],train_horizons_minut:6,train_model:[0,4,5],train_split_func:6,train_val_test_gener:12,transform:10,treat:18,treemap:8,trial:9,trick:8,trivial:7,tupl:[7,8,9,10,11,12,14,17,18],turbin:7,turbine_data:[7,15],turbine_typ:6,turbinedata:7,tutori:3,two:[8,9],type:[6,7,8,9,10,11,12,13,14,15,16,17,18,19],typic:17,under:3,underestim:8,understand:1,uniform:[7,17],union:[6,9,10,11,13,18,19],uniqu:8,unknown:5,unrequest:7,until:7,updat:7,uri:8,url:13,usag:1,use:[1,3,6,8,9,12,15,20],used:[1,3,7,8,9,10,11,15,19],useful:[7,17],user:[1,21],uses:[9,17],using:[3,7,8,9,10,11,12,13,17,18,19],util:[0,4,5,10,17],vaisala:7,val:12,valid:[0,1,4,5,10,12,17],valid_kwarg:11,validation_data:[8,9,12],validation_fract:[9,11,12],valu:[7,8,9,10,11,12,13,14,15,16,17,19],valueerror:[9,11],vapour:7,vapour_pressur:7,vari:15,variabl:7,verbos:[9,11],version:3,via:[1,13],view:8,visual:1,volum:15,voorjaarsvakanti:7,wai:[7,9],want:12,warn:11,water:7,weather:[1,7,14,15,17],weather_data:[14,15],weather_featur:[0,4,5],web:8,webhook:13,week:[1,8,9],weekdai:7,weight:[8,11,19],weight_importance_nam:11,well:17,were:7,west:7,wheather:7,when:[7,8,9,10,11,12,14,17,18],where:[7,8,10,19],whether:[7,13,18],which:[1,3,7,9,11,15,17],why:8,wiki:7,wikipedia:7,wind:[1,5,7,15,17],wind_profile_power_law:7,wind_ref:17,windenergi:15,window:17,windpow:15,windspe:7,windspeed_100m:[14,15],windspeedhub:7,within:[8,11],without:[9,10],worker:10,workspac:7,world:7,wrap:11,write:[1,8,17],write_report_to_disk:8,wrong:[5,17],www:7,xgb:[1,5,9],xgb_quantil:[5,9],xgb_quantile_ev:8,xgb_quantile_obj:8,xgboost:[8,11],xgbopenstfregressor:[9,11],xgbquantil:[9,11],xgbquantileopenstfregressor:[9,11],xgbquantileregressorobject:9,xgbregressor:11,xgbregressorobject:9,y_pred:8,y_true:8,year:7,yesterdai:1,yield:17,you:3,your:[3,7],yourself:1,zero:[8,10,15,19],zero_bound:17,zomervakanti:7,zone:7},titles:["Code modules","Concepts","&lt;no title&gt;","Welcome to the documentation of OpenSTEF!","openstef","openstef package","openstef.data_classes package","openstef.feature_engineering package","openstef.metrics package","openstef.model package","openstef.model.metamodels package","openstef.model.regressors package","openstef.model_selection package","openstef.monitoring package","openstef.pipeline package","openstef.postprocessing package","openstef.preprocessing package","openstef.tasks package","openstef.tasks.utils package","openstef.validation package","Quickstart","Tutorials"],titleterms:{"enum":5,applic:1,apply_featur:7,architectur:1,basecas:9,calculate_kpi:17,code:0,concept:1,confidence_interval_appl:9,content:[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],contribut:3,create_basecase_forecast:[14,17],create_component_forecast:14,create_components_forecast:17,create_forecast:[14,17],create_solar_forecast:17,create_wind_forecast:17,custom_regressor:11,data_class:6,depend:18,document:3,exampl:21,except:5,fallback:9,feature_add:7,feature_appl:7,feature_engin:7,figur:8,gener:7,grouped_regressor:10,historic_featur:7,holiday_featur:7,implement:21,instal:20,lag_featur:7,lgbm:11,licens:3,linear:11,metamodel:10,metric:8,missing_values_handl:10,model:[9,10,11],model_cr:9,model_select:12,model_specif:6,modul:[0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],monitor:13,object:9,objective_cr:9,openstef:[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],optimize_hyperparamet:[14,17],packag:[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],performance_met:13,pipelin:14,postprocess:15,prediction_job:6,predictionjobloop:18,preprocess:16,proloaf:11,quickstart:20,regressor:11,report:8,run_traci:17,serial:9,softwar:1,split_forecast:17,split_funct:6,standard_deviation_gener:9,submodul:[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],subpackag:[5,9,17],task:[17,18],taskcontext:18,team:13,train_create_forecast_backtest:14,train_model:[14,17],tutori:21,usag:20,util:[14,18],valid:19,weather_featur:7,welcom:3,xgb:11,xgb_quantil:11}})