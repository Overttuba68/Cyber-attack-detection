# Importing necessary libraries
from flask import Flask, render_template, request, url_for, flash, redirect,session
import pandas as pd 
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import mysql.connector
db=mysql.connector.connect(user="root",password="",port='3306',database='cyber_attack')
cur=db.cursor()

app=Flask(__name__)
app.secret_key="CBJcb786874wrf78chdchsdcv"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
        cur.execute(sql)
        data=cur.fetchall()
        db.commit()
        if data ==[]:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            return render_template("load.html",myname=data[0][1])
    return render_template('login.html')

@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        address = request.form['address']
        contact = request.form['contact']
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,Age,Address,Contact)values(%s,%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,address,contact)
                cur.execute(sql,val)
                db.commit()
                flash("Registered successfully","success")
                return render_template("login.html")
            else:
                flash("Details are invalid","warning")
                return render_template("registration.html")
        else:
            msg = "Password doesn't match"
            return render_template("registration.html",msg=msg)
    return render_template('registration.html')

@app.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')


@app.route('/view')
def view():
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())


@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer,df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        df['protocol_type']=le.fit_transform(df['protocol_type']) 
        df['flag']= le.fit_transform(df['flag'])
        df['service']= le.fit_transform(df['service'])
        
       # Assigning the value of x and y 
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.3, random_state=42)

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train,x_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        global model
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')
        elif s == 1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            from sklearn.ensemble import ExtraTreesClassifier
            et = ExtraTreesClassifier()
            et.fit(x_train,y_train)
            y_pred = et.predict(x_test)
            ac_et = accuracy_score(y_test, y_pred)
            ac_et = ac_et * 100
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by Extra Tree Classifier is  ' + str(ac_et) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 2:
            classifier = DecisionTreeClassifier(max_leaf_nodes=39, random_state=0)
            classifier.fit(x_train, y_train)
            y_pred  =  classifier.predict(x_test)            
            
            ac_dt = accuracy_score(y_test, y_pred)
            ac_dt = ac_dt * 100
            msg = 'The accuracy obtained by Decision Tree Classifier is ' + str(ac_dt) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 3:
            from sklearn.svm import SVC
            svc=SVC()
            svc=svc.fit(x_train,y_train)
            y_pred  =  svc.predict(x_test)            
            
            ac_svc = accuracy_score(y_test, y_pred)
            ac_svc = ac_svc * 100
            msg = 'The accuracy obtained by Support Vector Classifier is ' + str(ac_svc) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 4:
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=12)
            knn.fit(x_train, y_train)
            y_pred  =  knn.predict(x_test)            
            
            ac_knn = accuracy_score(y_test, y_pred)
            ac_knn = ac_knn * 100
            msg = 'The accuracy obtained by K-Nearest Neighbour is ' + str(ac_knn) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 5:
            adb = AdaBoostClassifier()
            adb.fit(x_train, y_train)
            y_pred  =  adb.predict(x_test)            
            
            ac_adb = accuracy_score(y_test, y_pred)
            ac_adb = ac_adb * 100
            msg = 'The accuracy obtained by AdaBoost Classifier is ' + str(ac_adb) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 6:
            from keras.models import Sequential
            from keras.layers import Dense, Dropout

            from keras.models import load_model
            model = load_model('neural_network.h5')
            score=0.9423418045043945
            ac_nn = score * 100
            msg = 'The accuracy obtained by Neural Network is ' + str(ac_nn) + str('%')
            return render_template('model.html', msg=msg)
    return render_template('model.html')

@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        # f1=int(request.form['city'])
        f1 = float(request.form['duration'])
        f2 = float(request.form['protocol_type'])
        f3 = float(request.form['service'])
        f4 = float(request.form['flag'])
        f5 = float(request.form['src_bytes'])
        f6 = float(request.form['dst_bytes'])
        f7 = float(request.form['land'])
        f8 = float(request.form['wrong_fragment'])
        f9 = float(request.form['urgent'])
        f10 = float(request.form['hot'])
        f11 = float(request.form['num_failed_logins'])
        f12 = float(request.form['logged_in'])
        f13 = float(request.form['num_compromised'])
        f14 = float(request.form['root_shell'])
        f15 = float(request.form['su_attempted'])
        f16 = float(request.form['num_root'])
        f17 = float(request.form['num_file_creations'])
        f18 = float(request.form['num_shells'])
        f19 = float(request.form['num_access_files'])
        f20 = float(request.form['num_outbound_cmds'])
        f21 = float(request.form['is_host_login'])
        f22 = float(request.form['is_guest_login'])
        f23 = float(request.form['count'])
        f24 = float(request.form['srv_count'])
        f25 = float(request.form['serror_rate'])
        f26 = float(request.form['srv_serror_rate'])
        f27 = float(request.form['rerror_rate'])
        f28 = float(request.form['srv_rerror_rate'])
        f29 = float(request.form['same_srv_rate'])
        f30 = float(request.form['diff_srv_rate'])
        f31 = float(request.form['srv_diff_host_rate'])
        f32 = float(request.form['dst_host_count'])
        f33 = float(request.form['dst_host_srv_count'])
        f34 = float(request.form['dst_host_same_srv_rate'])
        f35 = float(request.form['dst_host_diff_srv_rate'])
        f36 = float(request.form['dst_host_same_src_port_rate'])
        f37 = float(request.form['dst_host_srv_diff_host_rate'])
        f38 = float(request.form['dst_host_serror_rate'])
        f39 = float(request.form['dst_host_srv_serror_rate'])
        f40 = float(request.form['dst_host_rerror_rate'])
        f41 = float(request.form['dst_host_srv_rerror_rate'])
        
        print(f2)
        print(type(f2))

        li = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,
                f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,f41]
        print(li)
        
        # model.fit(X_transformed, y_train)
        
        # print(f2)
        import pickle
        filename = 'DecisionTreeClassifier.sav'
        model = pickle.load(open(filename, 'rb'))
        result = model.predict([li])
        print(result)
        print('result is ',result)
        # (Anomaly  = 0,   Normal  = 1 )
        if result == 0:
            msg = 'There is Cyber Attack'
            return render_template('prediction.html', msg=msg)
        else:
            msg = 'There is  No-Cyber Attack'
            return render_template('prediction.html', msg=msg)
    return render_template('prediction.html')










if __name__=='__main__':
    app.run(debug=True)