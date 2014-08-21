from flask import render_template, request
from app import app
from DropoutPredictor import Predict_Final 

IDl = []
math2l = []
age1l = []
numhsl = []
SESl = []
frdropl = []
absentl = []
skipl = []
suspl = []
hhnuml = []
dummyl = []

@app.route("/")
def landing():
  return render_template("landing.html")

@app.route("/index")
def index():
  return render_template("landing.html")

@app.route("/AddStudent")
def home():
  return render_template("home2.html")

@app.route("/contact")
def contact():
  return render_template("contact.html")

@app.route("/clear")
def homepage():
  IDl = []
  math2l = []
  age1l = []
  numhsl = []
  SESl = []
  frdropl = []
  absentl = []
  skipl = []
  suspl = []
  hhnuml = []
  dummyl = []
  return render_template("home2.html")

@app.route("/result")
def resultpage():
  theActions = []
  ID = float(request.args.get('ID'))
  math2 = float(request.args.get('math2'))
  agetemp = request.args.get('age')

  #The age is non-intuituve - correct it
  if agetemp == '15 and under':
    age1 = 7
  elif agetemp == '16':
    age1 = 6
  elif agetemp == '17':
    age1 = 5
  elif agetemp == '18':
    age1 = 4
  elif agetemp == '19':
    age1 = 3
  elif agetemp == '20':
    age1 = 2
  elif agetemp == '21':
    age1 = 1
  elif agetemp =='22+':
    age1 = 0
  print age1, 'age'
  SES1 = request.args.get('SES')
  #prepare inputs for SES
  if SES1 == '-2.0 to -1.5':
    SES1 = -1.6
  elif SES1 == '-1.5 to -1.0':
    SES1 = -1.3
  elif SES1 == '-1.0 to -0.5':
    SES1 = -0.7
  elif SES1 == '-0.5 to 0':
    SES1 = -0.3
  elif SES1 == '0 to 0.5':
    SES1 = 0.25
  elif SES1 == '0.5 to 1.0':
    SES1 = -0.72
  elif SES1 == '1.0 to 1.5':
    SES1 = 1.25
  else: 
    SES1 = 1.6

  frdrop1 = int(request.args.get('frdrop'))

  dummy1 = 0 
 
  # prepare inputs for hhnum
  if request.args.get('hhnum') == '8+' :
    hhnum1 = 8
  else: hhnum1 = float(request.args.get('hhnum'))

  # prepare inputs for numhs
  if request.args.get('numhs') == '5+' :
    numhs1 = 5
  else: numhs1 = float(request.args.get('numhs'))
  
  # prepare inputs for skip
  if request.args.get('skip') == '10+' :
    skip1 = 4
  elif request.args.get('skip') == '7-9' :
    skip1 = 3
  elif request.args.get('skip') == '3-6' :
    skip1 = 2
  elif request.args.get('skip') == '1-2' :
    skip1 = 1
  elif request.args.get('skip') == '0' :
    skip1 = 0

  # prepare inputs for absent
  if request.args.get('absent') == '10+' :
    absent1 = 4
  elif request.args.get('absent') == '7-9' :
    absent1 = 3
  elif request.args.get('absent') == '3-6' :
    absent1 = 2
  elif request.args.get('absent') == '1-2' :
    absent1 = 1
  elif request.args.get('absent') == '0' :
    absent1 = 0

  # prepare inputs for susp
  if request.args.get('susp') == '10+' :
    susp1 = 4
  elif request.args.get('susp') == '7-9' :
    susp1 = 3
  elif request.args.get('susp') == '3-6' :
    susp1 = 2
  elif request.args.get('susp') == '1-2' :
    susp1 = 1
  elif request.args.get('susp') == '0' :
    susp1 = 0
 
  #are they ok?
  print ID
  print absent1
  print frdrop1
  print math2
  print SES1
  print age1
  print susp1
  print numhs1
  print hhnum1
  print skip1

  IDl.append(ID)
  math2l.append(math2)
  age1l.append(age1)
  numhsl.append(numhs1)
  hhnuml.append(hhnum1)
  SESl.append(SES1)
  frdropl.append(frdrop1)
  dummyl.append(dummy1)
  absentl.append(absent1)
  suspl.append(susp1)
  skipl.append(skip1)

  # This is the result...
  theList = Predict_Final(ID = IDl, math2=math2l, age=age1l, absent=absentl, skip=skipl, susp = suspl,  numhs=numhsl, ses=SESl, frdrop=frdropl, hhnum=hhnuml, dummy=dummyl)  
  # Do you want to build the actions here or in the other class?
  if absent1 > 4:
    theActions.append('The student is missing many days, investigate why.')
  if request.args.get('math2') < -1.5:
    theActions.append('Due to low student math scores, intervene with tutoring.')
  return render_template("result.html", theList=theList, theActions=theActions)

