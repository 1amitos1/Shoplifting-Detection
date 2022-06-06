import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

class Email_Alert:
    def __init__(self,users_Email_list=["amitos684@gmail.com","barloupo@gmail.com","eyal@gat.org.il"]):
        self.user_email_address1 =users_Email_list[0]
        self.user_email_address2 =users_Email_list[1]
        self.user_email_address3 = users_Email_list[2]

    def send_email_alert(self, toaddr, filename, absulutefilepath):
            """
            :param toaddr: user email address
            :param filename:name of the video clips
            :param absulutefilepath: full path to video file
            :return: None
            """

            print(toaddr)
            print(filename)
            print(absulutefilepath)
            fromaddr = "absabusedetection@gmail.com"

            # instance of MIMEMultipart
            msg = MIMEMultipart()

            # storing the senders email address
            msg['From'] = fromaddr

            # storing the receivers email address
            msg['To'] = toaddr

            # storing the subject
            msg['Subject'] = "ADS Alert"

            # string to store the body of the mail
            # body = "Body_of_the_mail"
            body = f"Hello ,\nADS Alert: Warning, we found the following video to contain abuse\ntime:{datetime.now()}"

            # attach the body with the msg instance
            msg.attach(MIMEText(body, 'plain'))

            # open the file to be sent
            # filename = "v_13_.avi"
            attachment = open(absulutefilepath, "rb")

            # instance of MIMEBase and named as p
            p = MIMEBase('application', 'octet-stream')

            # To change the payload into encoded form
            p.set_payload((attachment).read())

            # encode into base64
            encoders.encode_base64(p)

            p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

            # attach the instance 'p' to instance 'msg'
            msg.attach(p)

            # creates SMTP session
            s = smtplib.SMTP('smtp.gmail.com', 587)

            # start TLS for security
            s.starttls()

            # Authentication
            s.login(fromaddr, "ABS1234amitBAR")

            # Converts the Multipart msg into a string
            text = msg.as_string()

            # sending the mail
            s.sendmail(fromaddr, toaddr, text)

            # terminating the session
            s.quit()
            print(f"[+][+]Done sending Email\n")



