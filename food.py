from gpiozero import Servo
from time import sleep

servo = Servo(17)

servo.max()
sleep(0.75)
servo.detach()
