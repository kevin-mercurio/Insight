from flask import render_template, request
from app import app
from DropoutPredictor import Predict_Final 

IDl = []
math2l = []
age1l = []
numhsl = []
SESl = []
frdropl = []
behavel = []

@app.route("/")
def landing():
  return render_template("landing.html")

@app.route("/index")
def landing():
  return render_template("landing.html")

@app.route("/AddStudent")
def home():
  return render_template("home2.html")

@app.route("/clear")
def homepage():
  IDl = []
  math2l = []
  age1l = []
  numhsl = []
  SESl = []
  frdropl = []
  behavel = []
  return render_template("home2.html")

@app.route("/result")
def resultpage():
  theActions = []
  ID = float(request.args.get('ID'))
  math2 = float(request.args.get('math2'))
  age1 = int(request.args.get('age1'))
  numhs1 = int(request.args.get('numhs1'))
  SES1 = float(request.args.get('SES1'))
  frdrop1 = int(request.args.get('frdrop1'))
  behave1 = float(request.args.get('behave1'))
  #are they ok?
  print request.args.get('ID')
  print request.args.get('behave1')
  print request.args.get('frdrop1')
  print request.args.get('math2')
  print request.args.get('SES1')
  print request.args.get('age1')
  print request.args.get('numhs1')

  IDl.append(ID)
  math2l.append(math2)
  age1l.append(age1)
  numhsl.append(numhs1)
  SESl.append(SES1)
  frdropl.append(frdrop1)
  behavel.append(behave1)

  theList = Predict_Final(ID = IDl, math2=math2l, age=age1l, behave=behavel, numhs=numhsl, ses=SESl, frdrop=frdropl)  
  if request.args.get('math1') < -1.5:
    theActions.append('Due to low student math scores, intervene with tutoring')
  else:
    theActions.append('This student is awesome, go away')
  return render_template("result.html", theList=theList, theActions=theActions)

