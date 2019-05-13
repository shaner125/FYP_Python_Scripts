import pyrebase, time, json, datetime
from gpiozero import Servo

configdb = {
    "apiKey": "AIzaSyBNZNvGzSo0BWLiy0ykfzpjKaRNKr7hTSs",
    "authDomain": "smart-feeder-1d272.firebaseapp.com",
    "databaseURL": "https://smart-feeder-1d272.firebaseio.com",
    "storageBucket": "smart-feeder-1d272.appspot.com",
    "serviceAccount": "smart-feeder-1d272-firebase-adminsdk-z648w-281cedb936.json"
}

firebase = pyrebase.initialize_app(configdb)

db = firebase.database()
auth = firebase.auth()
detectStatus = db.child("data").child("user")
storage = firebase.storage()
mylist = []
lastfed = ''

servo = Servo(17)
servo.detach()
pet_type = "dog"
feed_time = 0.5
feed_size = 2
user = auth.sign_in_with_email_and_password("shane@test.com","password123")

def stream_handler(message):
    global mylist
    global pet_type
    global feed_size
    global feed_time
    print(message["event"])
    print(message["path"])
    print(message["data"])
    if message["path"] == "/":
        payload_dict = message["data"]
        if 'feeding_schedule' in payload_dict:
            test = payload_dict["feeding_schedule"]
            for val in test:
                if str(val) != 'None':
                    mylist.append(val)
        if "pet_type" in payload_dict:
                pet_type = payload_dict["pet_type"]
                if pet_type == "dog":
                    feed_time = 0.5
                elif pet_type == "cat":
                    feed_time = 0.25
        if "feeding_size" in payload_dict:
            feed_size = int(payload_dict["feeding_size"])
    if "feeding_schedule" in message["path"]:
        if not message["data"]:
            del mylist[-1]
        else:
            mylist.append(message["data"])
    elif "pet_type" in message["path"]:
            pet_type = message["data"]
            if pet_type == "dog":
                    feed_time = 0.5
            elif pet_type == "cat":
                feed_time = 0.25
    elif "feeding_size" in message["path"]:
        feed_size = int(message["data"])

my_stream = detectStatus.stream(stream_handler)

while True:
    dateSTR = datetime.datetime.now().strftime("%H:%M")
    for item in mylist:
        if item == dateSTR and lastfed != item:
            print("dispensing food! = "+str(feed_size)+"*"+str(pet_type))
            servo.max()
            feed_size_time = (feed_time*feed_size)
            print(feed_size_time)
            time.sleep(feed_size_time)
            servo.detach()
            lastfed = item 
    time.sleep(10)
