from flask import Flask, render_template, request, redirect, url_for, session
import os
# import pickle
import mysql.connector
from catdogclas import predict_img
mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      password="Myabhiroop_174",
      database="mlt"
    )

app = Flask(__name__)
app.secret_key = 'Abhi'
img = os.path.join('static', 'Image')

@app.route("/")
def main():
    return render_template('login.html',msg='')


@app.route('/login', methods=['POST'])
def login():
    msg = ''
    username = request.form['username']
    password = request.form['password']
    cursor = mydb.cursor()
    cursor.execute('SELECT name, password FROM login WHERE name = %s AND password = %s', (username, password,))
    account = cursor.fetchone()
    cursor.close()
    if account:
        session['loggedin'] = True
        session['id'] = account[1]
        session['username'] = account[0]
        return render_template('image.html',data=0)
    else:
        msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['new-username']
    password = request.form['new-password']
    email = request.form['email']
    values = (username, email, password)
    mycursor = mydb.cursor()
    mycursor.execute('SELECT name FROM login WHERE name = %s', (username,))
    r=mycursor.fetchone()
    if r:
        msg="Username exists"
        return render_template('login.html', msg=msg)
    else:
        sql="INSERT INTO login (name,email,password) VALUES (%s,%s,%s)"
        mycursor.execute(sql,values)
        mydb.commit()
        mycursor.close()
        msg="Signup success"
        return redirect(url_for('image'))
    
@app.route('/upload_submit',methods=['POST','GET'])
def upload_submit():
    if request.method == 'POST':
        f = request.files['imgInp']
        path = f.filename
        file = os.path.join(img, path)
        f.save(file)
        result=predict_img(file)
        print(result)
        return render_template('result.html',img=file,res=result)

@app.route("/retry")
def retry():
    return render_template('image.html',data=0)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('main'))

if __name__=='__main__':
    app.run()