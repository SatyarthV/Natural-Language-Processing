#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:42:27 2017

@author: fractaluser
"""

from twilio.rest import Client

# Your Account SID from twilio.com/console
account_sid = "ACdf33746298a74e7df8c73331762b6a32"
# Your Auth Token from twilio.com/console
auth_token  = "3ba848a1f74dd2682909dd5cfd0e6453"

client = Client(account_sid, auth_token)

message = client.messages.create(
    to="+918369857961", 
    from_="13072985798",
    body="Hello from Python!")

print(message.sid)