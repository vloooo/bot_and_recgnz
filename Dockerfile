FROM python:3.5
RUN apt-get update -y
RUN apt-get install swig -y
RUN apt-get install libpulse-dev -y
RUN apt-get install libasound2-dev -y
RUN apt-get install -y libsm6 libxext6
RUN apt-get install -y libfontconfig1 libxrender1
ADD record_webhook.py /
ADD ents.py /
ADD phrases.py /
ADD history.csv /
ADD DetectChars.py /
ADD DetectPlates.py /
ADD config.py /
ADD Preprocess.py /
ADD PossibleChar.py /
ADD PossiblePlate.py /
ADD Main.py /
ADD modelL2.h5 /
ADD two_cars_client.csv /
ADD haarcascade_russian_plate_number.xml /
RUN pip install --upgrade pocketsphinx
RUN pip install twilio pandas flask numpy tensorflow keras openpyxl xlrd eventlet selenium
RUN pip install Pillow opencv-python matplotlib SpeechRecognition lxml beautifulsoup4
CMD [ "python", "/record_webhook.py"]




