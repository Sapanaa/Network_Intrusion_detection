import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_alert_email(subject, body, to_email, from_email, smtp_server, smtp_port, smtp_user, smtp_password):
    # Create message container
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    
    # Attach body text
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        # Establish a connection with the email server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Encrypt the connection
        server.login(smtp_user, smtp_password)
        
        # Send the email
        server.sendmail(from_email, to_email, msg.as_string())
        print(f"Alert sent to {to_email}")
    except Exception as e:
        print(f"Failed to send alert: {e}")
    finally:
        server.quit()

# Example usage
send_alert_email(
    subject="Intrusion Detection Alert",
    body="An attack was detected in your network!",
    to_email="admin@example.com",
    from_email="alertsystem@example.com",
    smtp_server="smtp.example.com",
    smtp_port=587,
    smtp_user="your_smtp_username",
    smtp_password="your_smtp_password"
)
