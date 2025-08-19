from flask import Flask, render_template, request, redirect, url_for,session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

try:
    from rag_engin import smart_answer 
except ModuleNotFoundError:
    print('check internet connection')
except ImportError as e:
    print("Error importing module:", e)



app = Flask(__name__)
app.secret_key = 'chatapp_ramesh_Ramesh12345@@_securekey'

# SQLite database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)


migrate = Migrate(app, db)

# Create the database and table
with app.app_context():
    db.create_all()
    
    
# Define model (table)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(100))
    user_pass = db.Column(db.String(100))
    user_area = db.Column(db.String(100)) 


# home interface
@app.route('/')
def home():
    if 'user' in session:
        print("Ramesh")
        return redirect(url_for('chat'))
    else:
       
        return redirect(url_for('login'))

# login interface
@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    usernames = []
    
    users_db_data = User.query.all()
    for u in users_db_data:
        usernames.append({
    "user_name": u.user_name,
    "user_pass": u.user_pass
        })

        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        
        matched_user = next((user for user in usernames if user["user_name"] == username and user["user_pass"] == password), None)

        if matched_user:
            session['user'] = username
            return redirect(url_for('chat'))
        else:
            message = 'Invalid username or password'

    return render_template('login.html', message=message)



# signup interface
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    message = ''
    usernames = []
    users_db_data = User.query.all()
    for u in users_db_data:
        usernames.append(u.user_name)
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        area = request.form['area']

        if username in usernames:
            message = 'Username already exists!'
        else:
            user = User(user_name=username, user_pass=password, user_area=area)
            db.session.add(user)
            db.session.commit()
            message = 'Signup successful! Please login.'
            session['user'] = username
            return render_template('chat.html', user=session['user'])

    return render_template('signup.html', message=message)

# user sign out
@app.route('/logout')
def logout():    
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/help')
def help():  
    return render_template('help.html', user=session['user'])


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'user' in session:
        if request.method == 'POST':        
            user_input = request.get_json()['user_input']
            user_output = smart_answer(user_input)
            
            return jsonify({'user_output': user_output})
            
        else:
            return render_template('chat.html', user=session['user'])
    else:
        
        return redirect(url_for('login'))
    
if __name__ == '__main__':
    app.run(debug=True)
