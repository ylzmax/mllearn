from sklearn.ensemble import IsolationForest as ios
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot


def run():
	#data=pd.read_csv("../data/test.csv")
	data=pd.read_csv("../data/b2.csv",usecols=[1],squeeze=True)
	#data.hist()
	#print((data))
	
	
	tmpa=data.values[:]
	a=tmpa[:,None]
	#print(np.shape(a[:,None]))

	
	clf=ios(contamination=0.01)
	clf.fit(a)
	#joblib.dump(clf,"./clf.pkl")

	#test=np.array([0.1,0])

	#make random testdata
	testdata=np.random.randint(500,10000,100)
	tmpa=np.append(tmpa,testdata)
	testb=tmpa[:,None]
	#testdata=testdata[:,None]
	y=clf.predict(testb)
	#print([e for e in zip(a,y) if e[1] == -1 ])
#	fig,ax=pyplot.subplots()
	for i,pv in enumerate(y):
		if pv == -1:
			pyplot.scatter(i,testb[i],c="r")
		else:
			pyplot.scatter(i,testb[i],c="b")
	pyplot.show()

def perdict():
	clf=joblib.load("./clf.pkl")
	if len(sys.argv) > 1:
		test=np.array([sys.argv[1],0])
		y=clf.predict([test])
		print([e for e in zip(test,y)])
	

run()
#perdict()
