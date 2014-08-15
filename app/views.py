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
dropprogl = []
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
  dropprogl = []
  dummyl = []
  return render_template("home2.html")

@app.route("/result")
def resultpage():
  theActions = []
  ID = float(request.args.get('ID'))
  math2 = float(request.args.get('math2'))
  age1 = int(request.args.get('age1'))
  #The age is non-intuituve - correct it
  if age1 == 15:
    age1 = 7
  if age1 == 16:
    age1 = 6
  if age1 == 17:
    age1 = 5
  if age1 == 18:
    age1 = 4
  elif age1 == 19:
    age1 = 3
  elif age1 == 20:
    age1 = 2
  elif age1 == 21:
    age1 = 1
  elif age1 ==22:
    age1 = 0
  else: age1 = 23
  numhs1 = int(request.args.get('numhs'))
  SES1 = float(request.args.get('SES'))
  frdrop1 = int(request.args.get('frdrop'))
  absent1 = float(request.args.get('absent'))
  susp1 = float(request.args.get('susp'))
  dropprog1 = (request.args.get('dropprog'))
  if dropprog1 == 'yes' or dropprog1 == 'Yes' : dropprog1 = 1
  else: dropprog1 = 0
  dummy1 = 0 
  hhnum1 = float(request.args.get('hhnum'))
  skip1 = float(request.args.get('skip'))
 
  #are they ok?
  print request.args.get('ID')
  print request.args.get('absent')
  print request.args.get('frdrop')
  print request.args.get('math2')
  print request.args.get('SES')
  print request.args.get('age1')
  print request.args.get('numhs')
  print request.args.get('hhnum')
  print request.args.get('numhs')

  IDl.append(ID)
  math2l.append(math2)
  age1l.append(age1)
  numhsl.append(numhs1)
  hhnuml.append(hhnum1)
  SESl.append(SES1)
  frdropl.append(frdrop1)
  dropprogl.append(dropprog1)
  dummyl.append(dummy1)
  absentl.append(absent1)
  suspl.append(susp1)
  skipl.append(skip1)

  # This is the result...
  theList = Predict_Final(ID = IDl, math2=math2l, age=age1l, absent=absentl, skip=skipl, susp = suspl,  numhs=numhsl, ses=SESl, frdrop=frdropl, dropprog=dropprogl, hhnum=hhnuml, dummy=dummyl)  
  # Do you want to build the actions here or in the other class?
  if absent1 > 4:
    theActions.append('The student is missing many days, investigate why.')
  if request.args.get('math2') < -1.5:
    theActions.append('Due to low student math scores, intervene with tutoring.')
  return render_template("result.html", theList=theList, theActions=theActions)

