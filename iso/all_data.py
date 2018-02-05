from sklearn.ensemble import IsolationForest as ios
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot

def makearray(csvfile):
	data=pd.read_csv(csvfile,usecols=[1],squeeze=True)
	a=data.values[:]
	a=a[:,None]
	return a 
	

def run():
	xtrain=makearray("../data/alldata.csv")
	clf=ios(contamination=0.005)
	clf.fit(xtrain)
	joblib.dump(clf,"./clfalldata.pkl")
	print("Run model finishd!")


def perdict_plot():
	if len(sys.argv) > 1:
		xtest=makearray(sys.argv[1])
	else:
		print("please input file path!")
		exit()
	clf=joblib.load("./clfalldata.pkl")
	y=clf.predict(xtest)
	for i,pv in enumerate(y):
		if pv == -1:
			pyplot.scatter(i,xtest[i],c="r")
		else:
			pyplot.scatter(i,xtest[i],c="b")
	pyplot.show()

def perdict():
	clf=joblib.load("./clfalldata.pkl")
	if len(sys.argv) > 1:
		test=np.array([sys.argv[1],None])
		y=clf.predict([test])
		print([e for e in zip(test,y)])
	

#run()
perdict_plot()
