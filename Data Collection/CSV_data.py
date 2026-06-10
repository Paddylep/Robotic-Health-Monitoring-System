import serial
from datetime import datetime
import csv
i = 0
#Open a csv file and set it up to receive comma delimited input
logging = open('logging.csv',mode='a')
writer = csv.writer(logging, delimiter=",", escapechar=' ', quoting=csv.QUOTE_NONE)

#Open a serial port that is connected to an Arduino
ser = serial.Serial('COM4', baudrate=115200, timeout=1)
ser.flushInput()

while True:
    #Read in data from Serial until \n n received
    ser_bytes = ser.readline()
    print(ser_bytes)

    #Convert received bytes to text format
    decoded_bytes = (ser_bytes[0:len(ser_bytes)-2].decode("utf-8"))
    print(decoded_bytes)

    #Retreive current time
    c = datetime.now()
    current_time = c.strftime('%H:%M:%S')
    print(current_time)

    #Write received data to CSV file
    writer.writerow([current_time,decoded_bytes])

# Close port and CSV file to exit
ser.close()
logging.close()
print("logging finished")
