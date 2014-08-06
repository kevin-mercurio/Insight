import pandas as pd
from pandas import DataFrame
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# RF 
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier # random forest
from sklearn.svm import SVC # support vector machine classifier
from sklearn.grid_search import GridSearchCV # hyperparameter grid search to find best model parameters
from sklearn import preprocessing # preprocess string labels into numerics
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_curve, auc



#Setup the file for saving the variable plots
pp = PdfPages('multipage.pdf')
#Read in too much data about students, nrows limits the size of the sample
dfBig = pd.read_csv('/Users/kevinmercurio/Downloads/HSLS_2009_v2_0_CSV_Datasets/hsls_09_student_v2_0.csv',nrows=10000 )


#Make separate datafames for dropouts and completers
df_0 = dfBig.loc[dfBig['s2enrollhs12']==1] 
df_1 = dfBig.loc[dfBig['s2enrollhs12']==2] 
#df_2 = dfBig.loc[dfBig['s2enrollhs12']==2] 
#df_3 = dfBig.loc[dfBig['s2enrollhs12']==3] 
#df_8 = dfBig.loc[dfBig['s2enrollhs12']<0] 

#Remove non-responders (<0)
#Remove homeschoolers for now (==3)
df2 = dfBig.loc[dfBig['s2enrollhs12']>0]
df3 = df2.loc[df2['s2enrollhs12']<3]

df = df3[['stu_id', 's2enrollhs12', 'x1txmquint', 's2absent', 's2birthyr' , 'x1locale','x1paredu','s1nohwdn','x2ses' , 's2frdropout', 's2hsjobnow', 'p1honors']]
#nfeat = len(df3.ncol)
#print nfeat
features = df.columns[2:12]
print features
#print df


#Set a bunch of variables
Drop_Dropout = df_1['s2enrollhs12']
Drop_DropWork = df_1['s2towork']
Drop_DropFamily = df_1['s2supportfam']
Drop_lang = df_1['s2lang1st']

Stay_Dropout = df_0['s2enrollhs12']
Stay_DropWork = df_0['s2towork']
Stay_DropFamily = df_0['s2supportfam']
Stay_lang = df_0['s2lang1st']

#Plot things
f, axarr = plt.subplots(2, sharex=True)
axarr[0].hist(df_0['s2absent'].values,bins=7,range=(-1,6),normed=True,color='b')
axarr[0].set_title('Student absence? bin 1 = never, bin 5 = 10+')
axarr[1].hist(df_1['s2absent'].values,bins=7,range=(-1,6),normed=True,color='r')
f.savefig('absent.eps', format = 'eps')


g, axarr2 = plt.subplots(2, sharex=True)
axarr2[0].hist(df_0['s1pubprv'].values,bins=7,range=(-1,6),normed=True,color='b')
axarr2[0].set_title('likely to go to 1-public 2-private 3 - havent thought')
axarr2[1].hist(df_1['s1pubprv'].values,bins=7,range=(-1,6),normed=True,color='r')
g.savefig('EduLikely.eps', format = 'eps')

h, axarr3 = plt.subplots(2, sharex=True)
axarr3[0].hist(df_0['s2birthyr'].values,bins=13,range=(-1,12),normed=True,color='b')
axarr3[0].set_title('birth year')
axarr3[1].hist(df_1['s2birthyr'].values,bins=13,range=(-1,12),normed=True,color='r')

i, axarr4 = plt.subplots(2, sharex=True)
axarr4[0].hist(df_0['x1locale'].values,bins=7,range=(-1,6),normed=True,color='b')
axarr4[0].set_title('location? urban = 1, rural = 5')
axarr4[1].hist(df_1['x1locale'].values,bins=7,range=(-1,6),normed=True,color='r')

j, axarr5 = plt.subplots(2, sharex=True)
axarr5[0].hist(df_0['c2dropout'].values,bins=7,range=(-1,6),normed=True,color='b')
axarr5[0].set_title('Does schol offer dropout prevention program?')
axarr5[1].hist(df_1['c2dropout'].values,bins=7,range=(-1,6),normed=True,color='r')

k, axarr6 = plt.subplots(2, sharex=True)
axarr6[0].hist(df_0['s2cantsend'].values,bins=7,range=(-1,6),normed=True,color='b')
axarr6[0].set_title('not studying bc college too expensive no =0, yes = 1')
axarr6[1].hist(df_1['s2cantsend'].values,bins=7,range=(-1,6),normed=True,color='r')

l, axarr7 = plt.subplots(2, sharex=True)
axarr7[0].hist(df_0['p1english'].values,bins=7,range=(-1,6),normed=True,color='b')
axarr7[0].set_title('Is English spoken regularly at home?')
axarr7[1].hist(df_1['p1english'].values,bins=7,range=(-1,6),normed=True,color='r')

m, axarr8 = plt.subplots(2, sharex=True)
axarr8[0].hist(df_0['x1paredu'].values,bins=7,range=(-1,6),normed=True,color='b')
axarr8[0].set_title('highest parent education')
axarr8[1].hist(df_1['x1paredu'].values,bins=7,range=(-1,6),normed=True,color='r')
m.savefig('ParentEdu.eps', format = 'eps')

n, axarr8 = plt.subplots(2, sharex=True)
axarr8[0].hist(df_0['x1txmquint'].values,bins=7,range=(-1,6),normed=True,color='b')
axarr8[0].set_title('math quintile')
axarr8[1].hist(df_1['x1txmquint'].values,bins=7,range=(-1,6),normed=True,color='r')

o, axarr9 = plt.subplots(2, sharex=True)
axarr9[0].hist(df_0['x1race'].values,bins=10,range=(-1,9),normed=True,color='b')
axarr9[0].set_title('race')
axarr9[1].hist(df_1['x1race'].values,bins=10,range=(-1,9),normed=True,color='r')
o.savefig('Race.eps', format = 'eps')

p, axarr10 = plt.subplots(2, sharex=True)
axarr10[0].hist(df_0['x2ses'].values,bins=50,range=(-2,2),normed=True,color='b')
axarr10[0].set_title('socio-economic')
axarr10[1].hist(df_1['x2ses'].values,bins=50,range=(-2,2),normed=True,color='r')
p.savefig('SES.eps', format = 'eps')

q, axarr11 = plt.subplots(2, sharex=True)
axarr11[0].hist(df_0['p1honors'].values,bins=5,range=(-1,4),normed=True,color='b')
axarr11[0].set_title('enrolled in honors')
axarr11[1].hist(df_1['p1honors'].values,bins=5,range=(-1,4),normed=True,color='r')
q.savefig('honors.eps', format = 'eps')

#Close the pdf doc with all the figures
pp.close()


#Do the RF Classification here
#This sets the training/testing datasets For now, 50% is training
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .6
df.head()

train, test = df[df['is_train']==True], df[df['is_train']==False]

clf = RandomForestClassifier()
y, _ = pd.factorize(train['s2enrollhs12'])
y_test, _ = pd.factorize(test['s2enrollhs12'])
clf.fit(train[features], y)

preds = clf.predict(test[features])
probs = clf.predict_proba(test[features])

fpr, tpr, thresh = roc_curve(y_test, probs[:,1])
roc_auc = auc(fpr,tpr)
print 'Area under the ROC curve: %f' % roc_auc

fig13 = plt.figure()
ax13 = fig13.add_subplot(111)
ax13.plot(fpr, tpr)
plt.show()


#print the cross validation table
print pd.crosstab(test['s2enrollhs12'], preds, rownames = ['actual'], colnames = ['pred'])
scores = cross_val_score(clf,train[features],y)
print scores.mean()



#count the students who need help
num = 0
numtohelp = 0

probVector = []
for p in probs:
  num+=1
  probVector.append(p)
  if p[1] > 0.3: 
    numtohelp+=1

print 'of total '+str(num) +' students, one should help the '+str(numtohelp)+ r' who  have > 30% chance of dropping out'

#Extract the ID numbers of the students who need help
TheStudents = []
for i, stud in enumerate(test['stu_id']):
  if probVector[i][1] > 0.45:
    print 'student id number = '+str(stud)+ ' has a '+str(round(100*probVector[i][1],2))+ r'% chance of dropping out' 
    TheStudents.append(stud)
'''  
print TheStudents
'''
#for student in TheStudents:
#  dfDrops = test.loc[ test['stu_id'] == student]
#  print dfDrops
print fpr
print tpr
