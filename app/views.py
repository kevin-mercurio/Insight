from flask import render_template, request, jsonify
from app import app
import pymysql as mdb
from DropoutPredictor import Predict_Final 

db= mdb.connect(user="root", host="localhost", db="world_innodb", charset='utf8')

@app.route("/index")
def home():
  return render_template("home2.html")

@app.route("/")
def homepage():
  return render_template("home2.html")

@app.route("/result")
def resultpage():
  theActions = []
  math2 = float(request.args.get('math2'))
  age1 = int(request.args.get('age1'))
  numhs1 = int(request.args.get('numhs1'))
  SES1 = float(request.args.get('SES1'))
  frdrop1 = int(request.args.get('frdrop1'))
  behave1 = float(request.args.get('behave1'))
  #are they ok?
  print request.args.get('behave1')
  print request.args.get('frdrop1')
  print request.args.get('math2')
  print request.args.get('SES1')
  print request.args.get('age1')
  print request.args.get('numhs1')

  math2l = []
  math2l.append(math2)
  age1l = []
  age1l.append(age1)
  numhsl = []
  numhsl.append(numhs1)
  SESl = []
  SESl.append(SES1)
  frdropl = [frdrop1]
  frdropl.append(frdrop1)
  behavel = []
  behavel.append(behave1)

  theList = Predict_Final(math2=math2l, age=age1l, behave=behavel, numhs=numhsl, ses=SESl, frdrop=frdropl, math1=math2l, mathquint=math2l)  
  if request.args.get('math1') < -1.5:
    theActions.append('Due to low student math scores, intervene with tutoring')
  else:
    theActions.append('This student is awesome, go away')
  return render_template("result.html", theList=theList, theActions=theActions)


@app.route("/jquery")
def index_jquery():
  return render_template('index_js.html')
@app.route("/db_json")
def cities_json():
  with db:
    cur = db.cursor()
    cur.execute("SELECT Name, CountryCode, Population FROM city ORDER BY Population DESC;")
    query_results = cur.fetchall()
  cities = []
  for result in query_results:
     cities.append(dict(name=result[0], Country=result[1], Population=result[2]))
  return jsonify(dict(cities=cities))
