import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(threat_info):
    sender_email = "was12qq3@gmail.com"  # your Gmail
    receiver_email = "sapanadhami1111@gmail.com"
    app_password = "fdpj lmxp erbs ygyg"  # <--- paste your Gmail App Password here (no spaces)

    subject = "🚨 Threat Detected!"
    body = f"A threat has been detected in the network log:\n\n{threat_info}"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
