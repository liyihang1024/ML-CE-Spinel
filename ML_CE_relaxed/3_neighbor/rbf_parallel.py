# -*- coding: utf-8 -*-
# @Time    : 2018/11/26 23:53
# @Author  : aries.yu

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.svm import NuSVR
import pandas as pd
import numpy as np
from mpi4py import MPI

def rbf_NuSVR(X_train, X_test, y_train, y_test, loop):

	nusvr = NuSVR()
	parameters = [{'C': [0.1, 1e0, 1e1, 1e2,1e3], 'gamma': [0.001,0.01, 0.1, 0.5], 'kernel': ['rbf']}]
	clf1 = GridSearchCV(nusvr, parameters, scoring='neg_mean_absolute_error', cv=5)
	clf1.fit(X_train, y_train)
	param = clf1.best_params_

	svr1 = NuSVR(C=param.get("C"), gamma=param.get("gamma"), kernel=param.get("kernel"))
	svr1.fit(X_train,y_train)

	y_train_pred = svr1.predict(X_train)
	dftrainname = "%d_rbf_train.csv" % (loop)
	result_train_df = pd.DataFrame()
	result_train_df["y_train"] = y_train
	result_train_df["y_train_pred"] = list(y_train_pred.reshape(1, -1)[0])
	result_train_df.to_csv("./%s" % (dftrainname), index=False)

	y_test_pred = svr1.predict(X_test)
	dftestname = "%d_rbf_test.csv" % (loop)
	result_test_df = pd.DataFrame()
	result_test_df["y_test"] = y_test
	result_test_df["y_test_pred"] = list(y_test_pred.reshape(1, -1)[0])
	result_test_df.to_csv("./%s" % (dftestname), index=False)

	train_r2 = r2_score(y_train, y_train_pred)
	test_r2 = r2_score(y_test, y_test_pred)
	train_score = np.sqrt(mean_squared_error(y_train, y_train_pred))
	test_score = np.sqrt(mean_squared_error(y_test, y_test_pred))
	train_mae = mean_absolute_error(y_train, y_train_pred)
	test_mae = mean_absolute_error(y_test, y_test_pred)

	return train_r2, test_r2, train_score, test_score, train_mae, test_mae


def linear_nuSVR(X_train, X_test, y_train, y_test, loop):

	nusvr = NuSVR()
	parameters = [{'C': [0.1, 1e0, 1e1, 1e2,1e3], 'gamma': [0.001,0.01, 0.1, 0.5], 'kernel': ['linear'],}]
	clf1 = GridSearchCV(nusvr, parameters, scoring='neg_mean_squared_error', cv=5)
	clf1.fit(X_train, y_train)
	param = clf1.best_params_

	svr1 = NuSVR(C=param.get("C"), gamma=param.get("gamma"), kernel=param.get("kernel"),)#nu=param.get("nu")
	svr1.fit(X_train,y_train)

	y_train_pred = svr1.predict(X_train)
	dftrainname = "%d_linear_train.csv" % (loop)
	result_train_df = pd.DataFrame()
	result_train_df["y_train"] = y_train
	result_train_df["y_train_pred"] = list(y_train_pred.reshape(1, -1)[0])
	result_train_df.to_csv("./%s" % (dftrainname), index=False)

	y_test_pred = svr1.predict(X_test)
	dftestname = "%d_linear_test.csv" % (loop)
	result_test_df = pd.DataFrame()
	result_test_df["y_test"] = y_test
	result_test_df["y_test_pred"] = list(y_test_pred.reshape(1, -1)[0])
	result_test_df.to_csv("./%s" % (dftestname), index=False)

	train_r2 = r2_score(y_train, y_train_pred)
	test_r2 = r2_score(y_test, y_test_pred)
	train_score = np.sqrt(mean_squared_error(y_train, y_train_pred))
	test_score = np.sqrt(mean_squared_error(y_test, y_test_pred))
	train_mae = mean_absolute_error(y_train, y_train_pred)
	test_mae = mean_absolute_error(y_test, y_test_pred)

	return train_r2, test_r2, train_score, test_score, train_mae, test_mae


def rf(X_train, X_test, y_train, y_test, fetures, loop):

	rf = RandomForestRegressor()
	parameters = [{'n_estimators': [20,50,70], 'max_depth':[3,4,5,7,10], "min_samples_split":[2,4,6,10]}]
	clf1 = GridSearchCV(rf, parameters, scoring='neg_mean_squared_error', cv=5)
	clf1.fit(X_train, y_train)
	param = clf1.best_params_

	svr1 = RandomForestRegressor(n_estimators=param.get("n_estimators"), max_depth=param.get("max_depth"),
								min_samples_split=param.get("min_samples_split"))
	svr1.fit(X_train,y_train)
   
	features = np.array([fetures]).reshape(-1,1)
	rf_feature_importance = svr1.feature_importances_
	rf_feature_importance = np.array([rf_feature_importance]).reshape(-1,1)
	features_and_importance = np.concatenate((features, rf_feature_importance), axis=1)
	feature_and_importance_sort = features_and_importance[np.lexsort(features_and_importance.T)]  # 按最后一列排序，默认升续
	feature_and_importance_sort = feature_and_importance_sort[::-1]  # 按行反转
	features_importance = pd.DataFrame(feature_and_importance_sort, columns=["feature", "importance"])
	feature_file_name = "%d_rf_feature_importance.csv"%(loop)
	features_importance.to_csv("./%s"%(feature_file_name), index=False)

	y_train_pred = svr1.predict(X_train)
	dftrainname = "%d_rf_train.csv" % (loop)
	result_train_df = pd.DataFrame()
	result_train_df["y_train"] = y_train
	result_train_df["y_train_pred"] = list(y_train_pred.reshape(1, -1)[0])
	result_train_df.to_csv("./%s" % (dftrainname), index=False)

	y_test_pred = svr1.predict(X_test)
	dftestname = "%d_rf_test.csv" % (loop)
	result_test_df = pd.DataFrame()
	result_test_df["y_test"] = y_test
	result_test_df["y_test_pred"] = list(y_test_pred.reshape(1, -1)[0])
	result_test_df.to_csv("./%s" % (dftestname), index=False)

	train_r2 = r2_score(y_train, y_train_pred)
	test_r2 = r2_score(y_test, y_test_pred)
	train_score = np.sqrt(mean_squared_error(y_train, y_train_pred))
	test_score = np.sqrt(mean_squared_error(y_test, y_test_pred))
	train_mae = mean_absolute_error(y_train, y_train_pred)
	test_mae = mean_absolute_error(y_test, y_test_pred)

	return train_r2, test_r2, train_score, test_score, train_mae, test_mae

def loops(num=10):
	
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	if num != size:
		num = size
	alldata   = np.empty([num,3])
	rbf_array = np.empty([num,6])
	linear_array = []
	rf_array = np.empty([num,6])
	loop_index = np.empty(num)


	def dosomething(item):
		data = pd.read_excel("./FE_trainData.xlsx", sheet_name=0, index_col=0, header = 0)
		n_row = data.shape[0]
		n_col = data.shape[1]
		#data[:,:-1]  = np.array(data[:,-1],dtype=float)
		for i in range(1, n_row):
			for j in range(0, n_col-1):
				data.iloc[i, j] = float(data.iloc[i, j])

		data = shuffle(data)

		df_corr = data.corr()
		df_corr.to_csv(r"df_corr"+str(item+1)+".csv")

		X = data.iloc[:, :-1]
		y = data.iloc[:, -1]

		features = list(X.columns)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
		std = MinMaxScaler()
		X_train = std.fit_transform(X_train)
		X_test = std.transform(X_test)
		rbf_list_tmp = np.array(rbf_NuSVR(X_train, X_test, y_train, y_test, item+1))
		rf_list_tmp = np.array(rf(X_train, X_test, y_train, y_test, features, item+1))
		
		return rbf_list_tmp,rf_list_tmp,item+1


	if rank >0:
		a,b,c = dosomething(rank)
	elif rank == 0:
		a,b,c = dosomething(rank)
		
	comm.Gather(a,rbf_array,root=0)
	comm.Gather(b,rf_array,root=0)
	if rank==0 :
		title_rbf = ["rbf_loop", "rbf_train_r2", "rbf_test_r2", "rbf_train_rmse", "rbf_test_rmse",
				 "rbf_train_mae", "rbf_test_mae", "rbf_mae_error", "rbf_mae_error_index"]

		loop_index = range(1,size+1)
		loop_index_array = np.array(loop_index).reshape(-1, 1)
		loop_index_array.dtype = int
 
		rbf_array_result = np.array(rbf_array)

		rbf_array_result_6 = rbf_array_result[:,5]                   # rbf返回结果(loops * 6 的二维数组)的最后一列
		rbf_test_mae_mean = np.mean(rbf_array_result, axis=0)[5]     # rbf返回结果(loops * 6 的二维数组)的最后一列的平均值
		error_rbf = rbf_array_result_6.reshape(-1,1) - rbf_test_mae_mean.reshape(-1,1)
		error_rbf_absolute = np.absolute(error_rbf)

		rbf_data_all = np.concatenate((loop_index_array, rbf_array_result, error_rbf_absolute), axis=1)
		rbf_data_all_sort = np.concatenate((rbf_data_all[np.lexsort(rbf_data_all.T)],loop_index_array), axis=1)

		rbf_array_mean = ["mean_rbf"]
		rbf_array_mean.extend(list(np.mean(rbf_array_result, axis=0)))
		rbf_array_mean.extend(["NAN", "NAN"])
		rbf_array_std = ["std_rbf"]
		rbf_array_std.extend(list(np.std(rbf_array_result, axis=0)))
		rbf_array_std.extend(["NAN", "NAN"])

		rbf_data_mean_std = np.concatenate((np.array([title_rbf]),rbf_data_all_sort, np.array([rbf_array_mean]), np.array([rbf_array_std])), axis=0)


		title_rf = ["rf_loop", "rf_train_r2", "rf_test_r2", "rf_train_rmse", "rf_test_rmse",
					"rf_train_mae", "rf_test_mae", "rf_mae_error", "rf_mae_error_index"]

		loop_index_array = np.array(loop_index).reshape(-1, 1)
		loop_index_array.dtype = int

		rf_array_result = np.array(rf_array)

		rf_array_result_6 = rf_array_result[:, 5]
		rf_test_mae_mean = np.mean(rf_array_result, axis=0)[5]
		error_rf = rf_array_result_6.reshape(-1, 1) - rf_test_mae_mean.reshape(-1, 1)
		error_rf_absolute = np.absolute(error_rf)

		rf_array_mean = ["mean_rf"]
		rf_array_mean.extend(list(np.mean(rf_array_result, axis=0)))
		rf_array_mean.extend(["NAN", "NAN"])
		rf_array_std = ["std_rf"]
		rf_array_std.extend(list(np.std(rf_array_result, axis=0)))
		rf_array_std.extend(["NAN", "NAN"])

		rf_data_all = np.concatenate((loop_index_array,rf_array_result, error_rf_absolute), axis=1)
		rf_data_all_sort = np.concatenate((rf_data_all[np.lexsort(rf_data_all.T)],loop_index_array), axis=1)

		rf_data_mean_std = np.concatenate((np.array([title_rf]) ,rf_data_all_sort, np.array([rf_array_mean]), np.array([rf_array_std])), axis=0)


		rbf_rf = np.concatenate((rbf_data_mean_std, rf_data_mean_std), axis=0)


		columns = ["loop" ,"train_r2", "test_r2", "train_rmse", "test_rmse", "train_mae", "test_mae", "mae_error", "mae_error_index"]
		scores = pd.DataFrame(rbf_rf, columns=columns)
		scores.to_csv(r"./scores.csv", index=False)


if __name__ == '__main__':
	loops(4)