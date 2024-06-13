# from numpy import double
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from flask import Flask, request, render_template
# import pickle
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# app = Flask(__name__)

# # Load the trained model
# model = pickle.load(open("churnprediction.pkl", "rb"))

# @app.route("/")
# def loadPage():
#     return render_template('home.html', query="")

# @app.route("/", methods=['POST'])
# def predict():
#     input_features = {
#         'SeniorCitizen': int(request.form['query1']),
#         'MonthlyCharges': float(request.form['query2']),
#         'TotalCharges': float(request.form['query3']),
#         'gender': request.form['query4'],
#         'Partner': request.form['query5'],
#         'Dependents': request.form['query6'],
#         'PhoneService': request.form['query7'],
#         'MultipleLines': request.form['query8'],
#         'InternetService': request.form['query9'],
#         'OnlineSecurity': request.form['query10'],
#         'OnlineBackup': request.form['query11'],
#         'DeviceProtection': request.form['query12'],
#         'TechSupport': request.form['query13'],
#         'StreamingTV': request.form['query14'],
#         'StreamingMovies': request.form['query15'],
#         'Contract': request.form['query16'],
#         'PaperlessBilling': request.form['query17'],
#         'PaymentMethod': request.form['query18'],
#         'tenure': int(request.form['query19']),
#         'email': request.form['query20']  # Add this line to retrieve the email field
#     }

#     # Convert input features to DataFrame
#     input_df = pd.DataFrame([input_features])

#     # Make predictions
#     prediction = model.predict(input_df)
#     probability = model.predict_proba(input_df)[:, 1]

#     if prediction == 1:
#         output1 = "This customer is likely to be churned!!"
#         send_email(input_features['email'], "Dear Customer, Our prediction indicates that you are likely to churn. Please reach out to us for any assistance.")
#     else:
#         output1 = "This customer is likely to continue!!"
    
#     output2 = "Confidence: {:.2f}%".format(probability[0] * 100)

#     return render_template('home.html', output1=output1, output2=output2, **input_features)

# def send_email(recipient_email, email_content):
#     sender_email = "bsdk7019@gmail.com"
#     password = "bsdk7019@gmail"
#     message = MIMEMultipart()
#     message['From'] = sender_email
#     message['To'] = recipient_email
#     message['Subject'] = "Churn Prediction Result"
#     message.attach(MIMEText(email_content, 'plain'))
#     try:
#         server = smtplib.SMTP('smtp.gmail.com', 587)
#         server.starttls()
#         server.login(sender_email, password)
#         text = message.as_string()
#         server.sendmail(sender_email, recipient_email, text)
#         server.quit()
#         return True
#     except Exception as e:
#         print("Email could not be sent. Error:", str(e))
#         return False

# if __name__ == "__main__":
#     app.run(debug=True)








from numpy import double
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
import pickle
from flask_mail import Mail, Message

app = Flask(__name__)
mail = Mail(app)

# Configure email settings
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_TLS'] =  False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = 'bsdk7019@gmail.com'  # Update with your Gmail address
app.config['MAIL_PASSWORD'] = 'bsdk7019@gmail'         # Update with your Gmail password

# Load the trained model
model = pickle.load(open("churnprediction.pkl", "rb"))

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

#  @app.route("/", methods=['POST'])
# def predict():
#     input_features = {
#         'SeniorCitizen': int(request.form['query1']),
#         'MonthlyCharges': float(request.form['query2']),
#         'TotalCharges': float(request.form['query3']),
#         'gender': request.form['query4'],
#         'Partner': request.form['query5'],
#         'Dependents': request.form['query6'],
#         'PhoneService': request.form['query7'],
#         'MultipleLines': request.form['query8'],
#         'InternetService': request.form['query9'],
#         'OnlineSecurity': request.form['query10'],
#         'OnlineBackup': request.form['query11'],
#         'DeviceProtection': request.form['query12'],
#         'TechSupport': request.form['query13'],
#         'StreamingTV': request.form['query14'],
#         'StreamingMovies': request.form['query15'],
#         'Contract': request.form['query16'],
#         'PaperlessBilling': request.form['query17'],
#         'PaymentMethod': request.form['query18'],
#         'tenure': int(request.form['query19']),
    
#         'email': request.form['query20']  # Add this line to retrieve the email field
#     }

#     # Convert input features to DataFrame
#     input_df = pd.DataFrame([input_features])

#     # Make predictions
#     prediction = model.predict(input_df)
#     probability = model.predict_proba(input_df)[:, 1]

#     if prediction == 1:
#         output1 = "This customer is likely to be churned!!"
#         send_email(input_features['email'], "Dear Customer, Our prediction indicates that you are likely to churn. Please reach out to us for any assistance.")
#     else:
#         output1 = "This customer is likely to continue!!"
    
#     output2 = "Confidence: {:.2f}%".format(probability[0] * 100)

#     return render_template('home.html', output1=output1, output2=output2, **input_features)

# def send_email(recipient_email, email_content):
#     msg = Message('Churn Prediction Result', sender='your-email@gmail.com', recipients=[recipient_email])
#     msg.body = email_content
#     try:
#         mail.send(msg)
#         return True
#     except Exception as e:
#         print("Email could not be sent. Error:", str(e))
#         return False

# if __name__ == "__main__":
#     app.run(debug=True)


@app.route("/", methods=['POST'])
def predict():
    input_features = {
        # Your existing code for input features
              'SeniorCitizen': int(request.form['query1']),
        'MonthlyCharges': float(request.form['query2']),
        'TotalCharges': float(request.form['query3']),
        'gender': request.form['query4'],
        'Partner': request.form['query5'],
        'Dependents': request.form['query6'],
        'PhoneService': request.form['query7'],
        'MultipleLines': request.form['query8'],
        'InternetService': request.form['query9'],
        'OnlineSecurity': request.form['query10'],
        'OnlineBackup': request.form['query11'],
        'DeviceProtection': request.form['query12'],
        'TechSupport': request.form['query13'],
        'StreamingTV': request.form['query14'],
        'StreamingMovies': request.form['query15'],
        'Contract': request.form['query16'],
        'PaperlessBilling': request.form['query17'],
        'PaymentMethod': request.form['query18'],
        'tenure': int(request.form['query19']),
    
        'email': request.form['query20']  # Add this line to retrieve the email field
    }

    # Convert input features to DataFrame
    input_df = pd.DataFrame([input_features])

    # Make predictions
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]

    if prediction == 1:
        output1 = "This customer is likely to be churned!!"
        email_sent = send_email(input_features['email'], "Dear Customer, Our prediction indicates that you are likely to churn. Please reach out to us for any assistance.")
    else:
        output1 = "This customer is likely to continue!!"
        email_sent = send_email(input_features['email'], "Dear Customer, Our prediction indicates that you are likely to continue. Please let us know if you need any assistance.")

    output2 = "Confidence: {:.2f}%".format(probability[0] * 100)

    if email_sent:
        email_status = "Email sent successfully"
    else:
        email_status = "Failed to send email"

    return render_template('home.html', output1=output1, output2=output2, email_status=email_status, **input_features)
def send_email(recipient_email, email_content):
    msg = Message('Churn Prediction Result', sender='your-email@gmail.com', recipients=[recipient_email])
    msg.body = email_content
    try:
        mail.send(msg)
        return True
    except Exception as e:
        print("Email could not be sent. Error:", str(e))
        return False

if __name__ == "__main__":
    app.run(debug=True)