import sys
import pandas as pd
from pandas import DataFrame
import collections
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import pylab as pl

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

def Predict_Final(ID = [12345], absent = [1],  math2 = [0.9774], 
                  age = [4],  ses = [1.518],
                  skip = [0],  susp = [1.1],
                  hhnum = [4],  frdrop = [0], 
                  numhs = [1.3], dummy = [0] ):


  with open('DropModel.p','r') as f:
    clf = pk.load(f)
    print clf.coef_
    #two coeff are zero--why?

  #Make a dataframe from the input data, this comes from the web entry form
  's2absent', 'x2txmquint', 's2birthyr', 's2frdropout', 'x2numhs', 'x2hhnumber', 's2satnum', 's2latesch', 's2skipclass', 's2inschsusp', 'x2ses', 'x1locale'
  data = {
          's2absent' : absent,
          'x2txmquint' : math2,
          's2birthyr' : age,
          's2frdropout' : frdrop,
          'x2numhs' : numhs,
          'x2hhnumber' : hhnum,
          's2satnum' :  dummy,
          's2latesch' : dummy,
          's2skipclass' : skip,
          's2inschsusp' : susp,
          'x2ses' : ses,
          'x1locale' :dummy}
  #this actually makes the dataframe and sets the column names
  #Test = DataFrame(data, columns = ['s2absent', 'x2txmquint', 's2birthyr', 'x2ses', 's2frdropout', 's2skipclass','s2inschsusp','x2hhnumber', 's2satnum', 'x1locale',  's2latesch','x2numhs'])
  Test = DataFrame(data, columns = ['s2absent', 'x2txmquint', 's2birthyr', 's2frdropout', 'x2numhs', 'x2hhnumber', 's2satnum', 's2latesch', 's2skipclass','s2inschsusp','x2ses','x1locale'])
  print Test

  #predict!
  probs = clf.predict_proba(Test)
  preds = clf.predict(Test)
  print probs, preds
  Test['stu_id'] = ID
  StuList = Test['stu_id'].get_values()

  #store and sort the probabilities and associated student identfication
  P = []
  StuID = []
  for i, p in enumerate(probs): 
    P.append(p[1])
    StuID.append(StuList[i])
 
  ResultDict = dict( zip(P,StuID) )
  oResultDict = collections.OrderedDict(sorted(ResultDict.items(), reverse=True))
  Result = []

  def RiskLev(risk):
    if risk < 0.2: return 'LOW'
    elif risk < 0.5: return 'MILD'
    else: return 'HIGH'

  for key, value in oResultDict.iteritems():
    print 'key = %s and value = %s' % (key, value)
    Result.append(('Student %s has a '+RiskLev(key) +' risk of leaving school') % int(value))

  print Result
  #Return a vector of messages, ranking the students in descending order of dropout risk
  return Result


#This function generates the model
def Predict(SampleSize=10000, studentID = 10111, MakePlots=True, studentIDs=[10110, 10111]):  

  #Setup the file for saving the variable plots
  pp = PdfPages('multipage.pdf')
  #Read in too much data about students, nrows limits the size of the sample
  dfBig = pd.read_csv('/Users/kevinmercurio/Downloads/HSLS_2009_v2_0_CSV_Datasets/hsls_09_student_v2_0.csv',nrows=SampleSize )
  
  print dfBig.shape[0], 'Total nSamples'
  #studentIDs
  Livedf = dfBig.loc[dfBig['stu_id'].isin(studentIDs)]

  
  df2 = dfBig.loc[dfBig['s2enrollhs12']>0]
  df3 = df2.loc[df2['s2enrollhs12']<3]
  df4 = df3.loc[df3['x2txmth']>-8] 
  df5 = df4.loc[df4['s2absent']>-1]
  df6 = df5.loc[df5['x2ses']>-8]
  df7 = df6.loc[df6['s2birthyr']>-1]  
  df8 = df7.loc[df7['x1txmth']>-8]  
  df9 = df8.loc[df8['x1paredu']>-1]  
  df12 = df9.loc[df9['x2behavein']>-8]  
  df13 = df12.loc[df12['x1txmquint']>-8]  
  df14 = df13.loc[df13['s2frdropout']>-8]  
  df15 = df14.loc[df14['s2satnum']>-8]  
  df16 = df15.loc[df15['x2hhnumber']>-8]  
  df18 = df16.loc[df16['x2txmquint']>-8]  
  print df18.shape[0]

  #Make separate df for dropouts and not dropouts
  df_0 = df18.loc[df18['s2enrollhs12']==1] 
  df_1 = df18.loc[df18['s2enrollhs12']==2] 

  df = df18[['stu_id', 's2enrollhs12', 's2absent',  'x2txmquint', 's2birthyr','s2frdropout', 'x2numhs', 'x2hhnumber', 's2satnum', 's2latesch','s2skipclass','s2inschsusp',  'x2ses' , 'x1locale'  ]]
  features = df.columns[2:24]
  print features 
  
  #Set a bunch of variables
  if(MakePlots):  
  #Plot things
    ap, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['s2friendsdo'].values,bins=3,range=(0,3),normed=True,color='b')
    axarr[1].set_title('Left because friends did')
    axarr[1].hist(df_1['s2friendsdo'].values,bins=3,range=(0,3),normed=True,color='r')
    ap.savefig('friendsDo.eps', format = 'eps')

    ao, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['s2gedeasier'].values,bins=3,range=(0,3),normed=True,color='b')
    axarr[1].set_title('Left because alternate was easier')
    axarr[1].hist(df_1['s2gedeasier'].values,bins=3,range=(0,3),normed=True,color='r')
    ao.savefig('gedeasier.eps', format = 'eps')

    an, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['s2poorgrade'].values,bins=3,range=(0,3),normed=True,color='b')
    axarr[1].set_title('Leave for grades')
    axarr[1].hist(df_1['s2poorgrade'].values,bins=3,range=(0,3),normed=True,color='r')
    an.savefig('poorgrades.eps', format = 'eps')

    am, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['s2supportfam'].values,bins=3,range=(0,3),normed=True,color='b')
    axarr[1].set_title('Leave for family')
    axarr[1].hist(df_1['s2supportfam'].values,bins=3,range=(0,3),normed=True,color='r')
    am.savefig('toSupport.eps', format = 'eps')

    al, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['s2suspendexp'].values,bins=3,range=(0,3),normed=True,color='b')
    axarr[1].set_title('Leave for behavior')
    axarr[1].hist(df_1['s2suspendexp'].values,bins=3,range=(0,3),normed=True,color='r')
    al.savefig('susp_exp.eps', format = 'eps')

    ak, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['s2dislikesch'].values,bins=3,range=(0,3),normed=True,color='b')
    axarr[1].set_title('Leave for dislike')
    axarr[1].hist(df_1['s2dislikesch'].values,bins=3,range=(0,3),normed=True,color='r')
    ak.savefig('dislike.eps', format = 'eps')

    aj, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['s2towork'].values,bins=3,range=(0,3),normed=True,color='b')
    axarr[1].set_title('Leave to work')
    axarr[1].hist(df_1['s2towork'].values,bins=3,range=(0,3),normed=True,color='r')
    aj.savefig('toWork.eps', format = 'eps')

    ai, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(df_0['s1talkprob'].values,bins=10,range=(0,10),normed=True,color='b')
    axarr[0].set_title('student_teacher_talk')
    axarr[1].hist(df_1['s1talkprob'].values,bins=10,range=(0,10),normed=True,color='r')
    ai.savefig('talk_teacher.eps', format = 'eps')

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
    h.savefig('Age.eps', format = 'eps')
    
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
    axarr11[0].hist(df_0['s2frdropout'].values,bins=6,range=(0,6),normed=True,color='b')
    axarr11[0].set_title('Friends Dropping out')
    axarr11[1].hist(df_1['s2frdropout'].values,bins=6,range=(0,6),normed=True,color='r')
    q.savefig('frdropout.eps', format = 'eps')
    
    #Close the pdf doc with all the figures
    pp.close()
  #End ifMakePlot 

  #Do the RF Classification here
  Y = df['s2enrollhs12'].get_values()
  df = df18[['stu_id', 's2enrollhs12', 's2absent',  'x2txmquint', 's2birthyr','s2frdropout', 'x2numhs', 'x2hhnumber', 's2satnum', 's2latesch','s2skipclass','s2inschsusp', 'x2ses' , 'x1locale'  ]]
  Xcol = df[df.columns[df.columns.isin(['s2absent',  'x2txmquint', 's2birthyr','s2frdropout', 'x2numhs', 'x2hhnumber', 's2satnum', 's2latesch','s2skipclass','s2inschsusp',   'x2ses' , 'x1locale', ])]].columns
  print Xcol, 'Xcol'
  X = df[df.columns[df.columns.isin(['s2absent',  'x2txmquint', 's2birthyr','s2frdropout', 'x2numhs', 'x2hhnumber', 's2satnum', 's2latesch','s2skipclass','s2inschsusp',   'x2ses' , 'x1locale' ])]].get_values()
  #set up RF
  cross_validation_object = cross_validation.StratifiedKFold(Y, n_folds = 4)
  features = df[df.columns[df.columns.isin(['s2absent',  'x2txmquint', 's2birthyr','s2frdropout', 'x2numhs', 'x2hhnumber', 's2satnum', 's2latesch','s2skipclass','s2inschsusp',  'x2ses' , 'x1locale'])]].columns.tolist()

  def BestFeat(x,feat,top_ten_indices):
    location = np.where(np.array(feat) == x)
    if location in top_ten_indices[0:10]:
      return 1
    else:
     return 0

  ranking=[]
  for train,test in cross_validation_object:
    rf_fit = RandomForestClassifier(100)
    sample_weights2 = np.array([45 if i == 2 else .5 for i in Y[train]])
    rf_fit.fit(X[train], Y[train], sample_weights2)
    indices = np.argsort(rf_fit.feature_importances_)[::-1]
    ranking.append(map(lambda(x):BestFeat(x, features,indices), features))
    importances = rf_fit.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_fit.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(10):
      print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    probs = rf_fit.predict_proba(X[train])
    preds1 = rf_fit.predict(X[test] )
    print pd.crosstab(Y[test], preds1, rownames = ['actual'], colnames = ['pred'])

  #Test LogiReg
  print 'now try logistic regression'
  for train,test in cross_validation_object:
    log_fit = LogisticRegression(C=0.15, penalty = 'l1', class_weight = 'auto')
    log_fit.fit(X[train], Y[train])
    
    probs = log_fit.predict_proba(X[test])
    preds1 = log_fit.predict(X[test] )
    print pd.crosstab(Y[test], preds1, rownames = ['actual'], colnames = ['pred'])



  #Dump These features to pk
  best_features = [features[i] for i in np.where(np.mean(ranking,axis =0) > 0.1)[0]]
  #print 'best features' , best_features
  with open('best_features.p','w') as f:
    pk.dump(best_features,f)

  #Now use just these features to train
  with open('best_features.p','r') as f:
    best_features = pk.load(f)

  X_new = X 
  #X_new = df[df.columns[df.columns.isin(best_features)]]
  Y_new = Y


  #Set a final model
  half = int(X_new.shape[0]*.75)
  X_new_train, X_new_test = X_new[:half], X_new[half:]
  Y_new_train, Y_new_test = Y_new[:half], Y_new[half:]

  final_model = LogisticRegression(C=0.07,penalty = 'l1', class_weight = 'auto')
  #final_model = LogisticRegression(C=1,penalty = 'l1', class_weight = 'auto')
  final_model.fit(X_new_train,Y_new_train)
  probs = final_model.predict_proba(X_new_test)
  preds = final_model.predict(X_new_test)
  print df.columns[2:24]
  print final_model.coef_
  print pd.crosstab(Y_new_test, preds, rownames = ['actual'], colnames = ['pred'])
  fpr, tpr, thresh = roc_curve(Y_new_test, probs[:,1], pos_label=2)
  roc_auc = auc(fpr,tpr)
  print 'Area under the ROC curve: %f' % roc_auc

  #Make a plot with the regression coefficients
  Coefs = final_model.coef_
  CoefV = []
  CoefN = []
  for p in Coefs:
    for i,j in enumerate(p):
      CoefV.append(j)
      CoefN.append('')
  #'s2absent', 'x2txmquint', 's2birthyr', 's2frdropout', 'x2numhs', 'x2hhnumber', 's2satnum', 's2latesch', 's2skipclass', 's2inschsusp', 'c2dropout', 'x2ses', 'x1race', 'x1locale', 'x2scieff', 's2wohwdn'
  Vals = df.columns[2:24]
  print Vals.tolist()
  print len(CoefV)
  fig, ax  = plt.subplots()
  index = np.arange(len(Vals))
  plt.bar(index, CoefV, width=.9)
  pl.xticks(index, CoefN)
  #pl.xticks(index, Vals.tolist())
  fig.savefig('test.eps', format = 'eps')
  #plt.show() 

  #print the ROC Curve for the final model with a 75/25 train/test split
  #plt.clf()
  #plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
  #plt.plot([0, 1], [0, 1], 'k--')
  #plt.xlim([0.0, 1.0])
  #plt.ylim([0.0, 1.0])
  #plt.xlabel('False Positive Rate')
  #plt.ylabel('True Positive Rate')
  #plt.title('Receiver operating characteristic')
  #plt.legend(loc="lower right")
  #plt.show()
 
  #Dump the final model to pk
  with open('model.p','w') as f:
    print 'dump model to pickle'
    pk.dump(final_model,f)

  return 'done  with model'

def main(argv):

  opts, args = getopt.getopt(argv,'n:F', ['num=', 'final='])
 
  for opt, arg in opts:
    print opt, arg
    if opt in ('-n', '--num'):
      Predict(int(arg))
    elif opt in ('-F', '--final'):
      Predict_Final()
    else: 
      print ' choose -n for training size or -F for final testing'
      sys.exit() 
   
if __name__ == '__main__' :
  sys.exit(main(sys.argv[1:]))
