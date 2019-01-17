FROM python:3.5
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
ADD haarcascade_russian_plate_number.xml /
RUN pip install twilio pandas flask numpy tensorflow keras openpyxl xlrd 
RUN pip install --upgrade pocketsphinx
RUN pip install Pillow opencv-python matplotlib SpeechRecognition lxml beautifulsoup4
CMD [ "python", "/record_webhook.py"]


