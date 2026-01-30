import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class AlertSystem:
    def __init__(self, sender_email, sender_password, receiver_email):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.receiver_email = receiver_email

    def send_alert(self, subject, message):
        try:
            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = self.receiver_email
            msg["Subject"] = subject

            msg.attach(MIMEText(message, "plain"))

            server = smtplib.SMTP("smtp.gmail.com", 587, timeout=30)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()

            print("🚨 Alert email sent successfully!")

        except Exception as e:
            print("❌ Failed to send alert:", e)
