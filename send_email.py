import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load email template
def load_template():
    with open("email_template.txt", "r") as file:
        template = file.read()
    return template

# Send email
def send_email(to_email, name, message_body):
    # Your email credentials
    sender_email = "balabhaskarnaraharisetti@gmail.com"
    sender_password = "baip iyji qszc olnf"  # Use App Password for Gmail

    # Prepare message content
    template = load_template()
    body = template.format(name=name, message=message_body)

    # Set up MIME
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = "Hello from Python"
    msg.attach(MIMEText(body, 'plain'))

    # Send the email
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
send_email("naraharisettibalabhaskar@gmail.com", "Ramesh", "This is a test email from the ML Dashboard.")
