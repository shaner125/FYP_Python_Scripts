import pyrebase, time

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
detectStatus = db.child("data")
storage = firebase.storage()


detectFlag = 0
livestreamflag = 0
user = auth.sign_in_with_email_and_password("shane@test.com","password123")

def stream_handler(message):
    print(message["event"])
    print(message["path"])
    print(message["data"])
    global detectFlag
    global livestreamflag
    if (message["path"] == "/detect"):
        print("changing!")
        detectFlag = message["data"]
    elif (message["path"] == "/livestream"):
        print("changing!")
        livestreamflag = message["data"]

my_stream = detectStatus.stream(stream_handler)

f1 = "20190512-223101"
f2 = f1+".mp4"
data = {f1:f2}
##storage.child("20190512-170853.avi").put("output/20190512-170853.avi", user['idToken'])
detectStatus.child("medialinks").update(data,user['idToken'])


while True:
    time.sleep(5)
