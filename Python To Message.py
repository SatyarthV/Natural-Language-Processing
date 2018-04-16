#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:42:27 2017

@author: satyarthvaidya
"""

from twilio.rest import Client

# Your Account SID from twilio.com/console
account_sid = "XXXXX"
# Your Auth Token from twilio.com/console
auth_token  = "YYYYY"

client = Client(account_sid, auth_token)

message = client.messages.create(
    to="ENTER DEST NUMBER", 
    from_="13072985798",
    body="Hello from Python!")

print(message.sid)
