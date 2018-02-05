from sklearn.ensemble import IsolationForest as ios
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import sys


def readcsv(csvname):
	f=pd.read_csv(csvname)
	return f

def run():
	data=readcsv("./cpu.csv")
	
	
	a=data.values[:,:]
	
	#a_train=[1,2,3,4,3,5,9,2,4,8,2,23,1,2,3,3,4,2,12,2,2,3,3,1,1,2,2,3,4,2]
	#a=np.reshape(a_train,(-1,2))
	#print(a)
	clf=ios(contamination=0.1)
	clf.fit(a)
	joblib.dump(clf,"./clf.pkl")
	#test=np.array([0.1,0])
	#print(test)
	#y=clf.predict([(test)])
	#print([e for e in zip(a,y) #if e[1] == -1 ])
	#print(y)

def perdict():
	clf=joblib.load("./clf.pkl")
	if len(sys.argv) > 1:
		test=np.array([sys.argv[1],0])
		y=clf.predict([test])
		print([e for e in zip(test,y)])
	

#run()
perdict()
