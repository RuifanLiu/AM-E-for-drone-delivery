# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:50:58 2022

@author: s313488
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.header import Header

sender = 'liuruifan1218@gmail.com'  # Suggest to use mail.sina.cn
pwd = 'Happygirl187'  # Password of sina mail
receiver = 'ruifan.liu@cranfield.ac.uk'
to = ['destination mail address']  # Destination mail address
msg = MIMEMultipart()
msg['Subject'] = Header('Go booking the appointment', 'utf-8')
msg['From'] = Header(sender)

text = 666666
content1 = MIMEText(('25' + '\n' + 'ruifan'), 'plain', 'utf-8')
msg.attach(content1)

port = 587  # For SSL    

s = smtplib.SMTP('smtp.gmail.com', port)  # SMTP server of sina mail
s.starttls()
s.login(sender, pwd)
s.sendmail(sender, receiver, msg.as_string())
s.close()