import sys
from sklearn import cross_validation
import pandas as pd
from pandas import DataFrame
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

# Classifiers 
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier # random forest
from sklearn.grid_search import GridSearchCV # hyperparameter grid search to find best model parameters
from sklearn import preprocessing # preprocess string labels into numerics
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn import learning_curve
from sklearn.pipeline import Pipeline

# setup options here
import getopt

def Predict_Final(behave = [0.71, -0.5], math1 = [0.2554, -1.0], math2 = [0.9774, -1.7], 
                  age = [5, 4], paredu = [4, 2], ses = [1.518, -0.7],
                  frdrop = [0, 1], scieff = [1.64, 0.54], numhs = [1, 1], studentIDs=[10110,10111,10112,10113]):

  #let's open the  model and best features that we save from Predict()
  new_columns = []
  with open('best_features.p','r') as f:
    best_features = pk.load(f)
    new_columns = best_features
    best_features.append('stu_id')
    best_features.append('s2enrollhs12')
  with open('model.p','r') as f:
    clf = pk.load(f)

  #Make a dataframe from the input data, this comes from the web entry form
  data = {'behave' : behave,
          'x1txmth' : math1,
          's2birthyr' : age,
          'x1paredu' : paredu,
          'x2ses' : ses,
          's2frdropout' : frdrop,
          'x2txmth' : math2,
          'x2scieff' : scieff,
          'x2numhs' : numhs}
  #this actually makes the dataframe and sets the column names
  Test = DataFrame(data)
  new_columns = new_columns[:-2]
  Test.columns = new_columns

  cols = [col for col in Test.columns if col not in ['stu_id', 's2enrollhs12']]
  cols2 = [col for col in Test.columns if col not in ['stu_id']]
  Test2 = Test[cols]
  Test3 = Test[cols2]

  #predict!
  probs = clf.predict_proba(Test2)
  preds = clf.predict(Test2)
  #print probs, preds

  #store and sort the probabilities and associated student identfication
  P = []
  for p in probs: 
    P.append(p[1])
  StuVect = []
  for i in sorted(enumerate(P), key=lambda x:x[1], reverse=True):
    StuVect.append(i)

  #Do something to store and return the results for each student / set of students
  Result = []
  for p in (StuVect):
    Result.append('Student #'+str(p[0])+' has a '+str(100*p[1])+'% probability to leave school')

  print Result
  #Return a vector of messages, ranking the students in descending order of dropout risk
  return Result
 


def Predict(SampleSize=10000, studentID = 10111, MakePlots=True, studentIDs=[10110, 10111]):  

  #Setup the file for saving the variable plots
  pp = PdfPages('multipage.pdf')
  #Read in too much data about students, nrows limits the size of the sample
  dfBig = pd.read_csv('/Users/kevinmercurio/Downloads/HSLS_2009_v2_0_CSV_Datasets/hsls_09_student_v2_0.csv',nrows=SampleSize )
  
  #studentIDs
  Livedf = dfBig.loc[dfBig['stu_id']==studentID]  
  Livedf = dfBig.loc[dfBig['stu_id'].isin(studentIDs)]
#  print Livedf['stu_id']
  
  #Remove non-responders (<0)
  #Remove homeschoolers for now (==3)
  df2 = dfBig.loc[dfBig['s2enrollhs12']>0]
  df3 = df2.loc[df2['s2enrollhs12']<3]
  df4 = df3.loc[df3['x2txmth']>-8] 
  df5 = df4.loc[df4['s2absent']>-1]
  df6 = df5.loc[df5['x2ses']>-8]
  df7 = df6.loc[df6['s2birthyr']>-1]  
  df8 = df7.loc[df7['x1txmth']>-8]  
  df9 = df8.loc[df8['x1paredu']>-1]  
  df10 = df9.loc[df9['x2scieff']>-7]  
  #print df10.shape[0]

  #Make separate df for dropouts and not dropouts
  df_0 = df10.loc[df10['s2enrollhs12']==1] 
  df_1 = df10.loc[df10['s2enrollhs12']==2] 

  df = df10[['stu_id', 's2enrollhs12', 'x1stuedexpct','x1sciid','s2latesch','x2behavein','x1txmth', 's2birthyr', 'x1paredu','x2ses','s2frdropout', 'x2txmth','x2scieff', 'x2numhs', 'x2hhnumber', 's2satnum', 'x1locale','s1hrmhomewk', 's1hrshomewk', 's1hrothhomwk', 's1hractivity']]
  features = df.columns[2:21]
  
  
  #Set a bunch of variables
  if(MakePlots):  
  #Plot things
    ah, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['x2numhs'].values,bins=5,range=(0,5),normed=True,color='b')
    axarr[0].set_title('Number of HS')
    axarr[0].set_ylabel('% of Sample')
    axarr[1].hist(df_1['x2numhs'].values,bins=5,range=(0,5),normed=True,color='r')
    ah.savefig('NumHS.eps', format = 'eps')

    ag, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['x1hhnumber'].values,bins=21,range=(-10,11),normed=True,color='b')
    axarr[0].set_title('Family Size')
    axarr[1].hist(df_1['x1hhnumber'].values,bins=21,range=(-10,11),normed=True,color='r')
    ag.savefig('FamSize.eps', format = 'eps')
  
    af, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['x1txmth'].values,bins=14,range=(-7,7),normed=True,color='b')
    axarr[0].set_title('math theta 1')
    axarr[1].hist(df_1['x1txmth'].values,bins=14,range=(-7,7),normed=True,color='r')
    af.savefig('txmath1.eps', format = 'eps')
  
    ae, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['x2txmth'].values,bins=14,range=(-7,7),normed=True,color='b')
    axarr[0].set_title('math theta 2')
    axarr[1].hist(df_1['x2txmth'].values,bins=14,range=(-7,7),normed=True,color='r')
    ae.savefig('txmath2.eps', format = 'eps')
  
    ad, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['x2behavein'].values,bins=30,range=(-7,3),normed=True,color='b')
    axarr[0].set_title('behavior')
    axarr[1].hist(df_1['x2behavein'].values,bins=30,range=(-7,3),normed=True,color='r')
    ad.savefig('Behavior.eps', format = 'eps')
  
    ac, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['s2apother'].values,bins=17,range=(-10,7),normed=True,color='b')
    axarr[0].set_title('taken an ap course not stem?')
    axarr[1].hist(df_1['s2apother'].values,bins=17,range=(-10,7),normed=True,color='r')
    ac.savefig('APnotSTEM.eps', format = 'eps')
  
    ab, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['s2inschsusp'].values,bins=17,range=(-10,7),normed=True,color='b')
    axarr[0].set_title('number of in school suspensions?')
    axarr[1].hist(df_1['s2inschsusp'].values,bins=17,range=(-10,7),normed=True,color='r')
    ab.savefig('nSuspend.eps', format = 'eps')
  
    aa, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['s2satnum'].values,bins=14,range=(0,13),normed=True,color='b')
    axarr[0].set_title('how many times did student take SAT?')
    axarr[1].hist(df_1['s2satnum'].values,bins=14,range=(0,13),normed=True,color='r')
    aa.savefig('nSATattemp.eps', format = 'eps')
  
    a, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['s2clginflu'].values,bins=14,range=(0,13),normed=True,color='b')
    axarr[0].set_title('who had influence over post-HS plan?')
    axarr[1].hist(df_1['s2clginflu'].values,bins=14,range=(0,13),normed=True,color='r')
    a.savefig('CollegeInf.eps', format = 'eps')
  
    b, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['s1stcheasy'].values,bins=12,range=(0,11),normed=True,color='b')
    axarr[0].set_title('did the student\'s parent do the survey?')
    axarr[1].hist(df_1['s1stcheasy'].values,bins=12,range=(0,11),normed=True,color='r')
    b.savefig('SciTchEasy.eps', format = 'eps')
  
    c, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['s1safe'].values,bins=16,range=(-10,5),normed=True,color='b')
    axarr[0].set_title('does the student feel safe?')
    axarr[1].hist(df_1['s1safe'].values,bins=16,range=(-10,5),normed=True,color='r')
    c.savefig('FeelSafe.eps', format = 'eps')
  
    d, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['x1pqstat'].values,bins=12,range=(0,11),normed=True,color='b')
    axarr[0].set_title('did the student\'s parent do the survey?')
    axarr[1].hist(df_1['x1pqstat'].values,bins=12,range=(0,11),normed=True,color='r')
    d.savefig('ParentSurvey.eps', format = 'eps')
   
    e, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['s2jobtime'].values,bins=13,range=(-10,2),normed=True,color='b')
    axarr[0].set_title('can\'t study bc of work?')
    axarr[1].hist(df_1['s2jobtime'].values,bins=13,range=(-10,2),normed=True,color='r')
    e.savefig('NoStudy-JobTime.eps', format = 'eps')
  
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
    k.savefig("CollegePrice.eps", format = 'eps')
  
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
    n.savefig('mathquint.eps', format = 'eps')
  
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
  #End ifMakePlot 

  #Do the RF Classification here
  temp = Livedf.copy()
  Y = df['s2enrollhs12'].get_values()
  X = df[df.columns[df.columns.isin(['x1txmth', 'x2behavein', 'x1stuedexpct', 'x1sciid', 's2birthyr', 'x1paredu', 'x2ses',
    's2frdropout', 'x2txmth', 'x2scieff', 'x2numhs', 'x2hhnumber', 's2satnum', 'x1locale', 's1hrshomewk'])]].get_values()
  Livedf2 = Livedf[Livedf.columns[Livedf.columns.isin(['s2enrollhs12','x1txmth', 'x2behavein', 'x1stuedexpct', 'x1sciid', 's2birthyr', 'x1paredu', 'x2ses',
    's2frdropout', 'x2txmth', 'x2scieff', 'x2numhs', 'x2hhnumber', 's2satnum', 'x1locale', 's1hrshomewk'])]]
  print('test student information: ')
  print(Livedf2)

  #set up RF
  cross_validation_object = cross_validation.StratifiedKFold(Y, n_folds = 5)
  features = df[df.columns[df.columns.isin(['x1txmth', 'x2behavein', 'x1stuedexpct', 'x1sciid', 's2birthyr', 'x1paredu', 'x2ses',
    's2frdropout', 'x2txmth', 'x2scieff', 'x2numhs', 'x2hhnumber', 's2satnum', 'x1locale', 's1hrshomewk'])]].columns.tolist()

  def BestFeat(x,features,top_ten_indices):
    location = np.where(np.array(features) == x)
    if location in top_ten_indices[0:8]:
      return 1
    else:
     return 0

  ranking=[]
  for train,test in cross_validation_object:
    rf_fit = RandomForestClassifier(100)
    sample_weights2 = np.array([20 if i == 2 else 1 for i in Y[train]])
    rf_fit.fit(X[train], Y[train], sample_weights2)
    indices = np.argsort(rf_fit.feature_importances_)[::-1]
    ranking.append(map(lambda(x):BestFeat(x, features,indices), features))


  #Dump These features to pk
  best_features = [features[i] for i in np.where(np.mean(ranking,axis =0) >0.1)[0]]
  print 'best features' , best_features
  with open('best_features.p','w') as f:
    pk.dump(best_features,f)

   #Now use just these features to train
  with open('best_features.p','r') as f:
    best_features = pk.load(f)

  X_new = df[df.columns[df.columns.isin(best_features)]]
  Y_new = Y
  sample_weights = np.array([20 if i == 2 else 1 for i in Y_new])

  cross_validation_object = cross_validation.StratifiedKFold(Y_new, n_folds = 5)
  rf_curve = learning_curve.learning_curve(RandomForestClassifier(100),
    X_new,Y_new, 
    cv = cross_validation_object)
  
 # logistic_curve = learning_curve.learning_curve(LogisticRegression(C=0.1),
  #  X_new,Y_new,
   # cv = cross_validation_object)
  
  if(MakePlots):
    fig, axes = plt.subplots(nrows = 2)
    axes[0].plot(rf_curve[0],np.mean(rf_curve[1],axis = 1))
    axes[0].plot(rf_curve[0],np.mean(rf_curve[2],axis = 1))
    axes[0].set_title('Random Forest')
    axes[0].set_ylabel('Area under ROC curve')
    axes[0].set_ylim([0.85,1.01])

    #axes[1].plot(logistic_curve[0], np.mean(logistic_curve[1], axis = 1))
    #axes[1].plot(logistic_curve[0], np.mean(logistic_curve[2], axis = 1))
    #axes[1].set_title('Logistic Regression')
    #axes[1].set_ylabel('Area under ROC curve')
    #axes[1].set_xlabel('No. of Training Samples')
    #axes[1].set_ylim([0.85,1.01])
    fig.savefig('AUCs.eps', format = 'eps')

  
  #model = LogisticRegression(C=0.1)
  model = RandomForestClassifier(100)
  tuned_parameters = [{'C': [0.05,0.1,0.15,0.2], 'penalty': ['l1','l2']}]

  #grid_search_object = GridSearchCV(model, tuned_parameters, cv = cross_validation_object)
#  grid_search_object.fit(X_new,Y_new)  # use fit if last item in pipeline is fit.
  #Set a final model
  #final_model = LogisticRegression(C=0.1,penalty = 'l1')
  #final_model.fit(X_new,Y_new)
  final_model = RandomForestClassifier(100)
  final_model.fit(X_new,Y_new, sample_weights)
  print 'dump model to pickle'
  #Dump the final model to pk
  with open('model.p','w') as f:
      pk.dump(final_model,f)

  #have a look at the testing sample
  preds_single = final_model.predict(Livedf[best_features])
  probs_single = final_model.predict_proba(Livedf[best_features])
  probs = final_model.predict_proba(Livedf[best_features])
  print probs_single
  #preds = final_model.predict(X_new[best_features])
  #probs = final_model.predict_proba(X_new[best_features])
  #print probs
  #print pd.crosstab(Y_new, preds, rownames = ['actual'], colnames = ['pred'])

  num = 0
  numtohelp = 0
  probVector = []
  for p in probs:
    num+=1
    probVector.append(p)
    if p[1] > 0.3: 
      numtohelp+=1

  PV = []
  Message = []
  for i, p in enumerate(probs):
    PV.append(p[1])
    if p[1] > 0.5:
      Message.append('this student has a ' +str(p[1]*100)+'% probability to leave school')
  
  return 'done  with model'


def main(argv):
#  try:
  opts, args = getopt.getopt(argv,'n:F', ['num=', 'final='])

  for opt, arg in opts:
    print opt, arg
    if opt in ('-n', '--num'):
      Predict(int(arg))
    else: Predict()
  Predict_Final()

if __name__ == '__main__' :
  sys.exit(main(sys.argv[1:]))
