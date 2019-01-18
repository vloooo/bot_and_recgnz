import re
from bs4 import BeautifulSoup
import string
from random import shuffle
from flask import Flask, request
from ents import ents, subjects_of_stages
import pandas as pd
import warnings
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
from urllib.parse import urlencode
import phrases
import ents
import numpy as np
import time
import requests
import config
from datetime import datetime
import tensorflow as tf
import Main
from keras.models import load_model
from keras.optimizers import RMSprop
import io
import speech_recognition as sr
from urllib.request import urlopen
import cv2
from collections import Counter


# to number plate rcgn
model = load_model('modelL2.h5')
model.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.005), loss='categorical_crossentropy',
              metrics=['accuracy'])
graph = tf.get_default_graph()

# for comfortable debug
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Account Sid and Auth Token from twilio.com/console
client = Client(config.account_sid, config.auth_token)

# start url for twiML
echo_url = config.echo_url

# main url for all requests
ngrok_url = config.qwestion_url
app = Flask(__name__)

# for exel parsing, (exel_updated is to switch we have/haven't new exel for parsing)
# (exel_doc_counter for knowing how much doc we have to parse)
exel_updated = False
exel_doc_counter = 0

#
cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# inner DB
"""
this Db use phone like client unique identificator (if phones different clients will coincide programm will behave
 unexpected)
"""
out = pd.DataFrame(
    {"reg_num": ['AS 123 SD', None, None], "mileage": ['1500', None, None], "city": ['London', None, None],
     "phone": ["+380999051660", "+380959293096", "+3333333333333"], "phone_for_offer": [None, None, None],
     "serv_hist": [None, None, None], 'again_key': [False, False, False], 'stage': [7, 1, 1],
     'pst': [False, False, False], 'ngt': [False, False, False], 'cnvrs_key': [0, 1, 2], 'num_calls': [0, 0, 3],
     'first_ques': [False, False, False], 'call_day': [0, 0, 0], 'accept': [None, None, None], 'img_url':
     ['https://www.classicdriver.com/sites/default/files/styles/full_width_slider/public/article_images/v12_laf_laferrari_01.jpg?itok=oS1qvHEQ', None, None], 'repeat_qwe': [True, True, True]})

# tokens for outer ALPR
tokens = ['7eb82f1ed5e6ceeca6b26f8316b31717fde0bb25', 'f9e106e2c0a0c6a2493181fd724cdb7b89600af9',
          '9118e9c1b2a3f65c39b8d90453db99165fb201f0', '0e8c566ad072d543aae409d576012ee4e98a766e']

quiq_recall_numbers = []  # list for numbers that need quick recall


#                                            find special values
#######################################################################################################################
def find_service_hist(client_speech, phone):
    """
    find service history in client speech, and update inner Db if serv.hi. was found
    :return found --> True/False (find special pattern or no)
    """
    global out

    found = False
    words = client_speech.lower().split()
    for word in words:
        if word in ents.serv_hist:
            out['serv_hist'][out.phone == phone] = ents.serv_hist_kind[word]
            found = True
            break
    return found


def find_city(client_speech, phone):
    """
    find client's city in client speech, and update inner Db if city was found
    :return found --> True/False (finded special pattern or no)
    """
    # delete punct
    translation = {ord(x): None for x in string.punctuation}
    client_speech = client_speech.translate(translation).lower()

    # find full city name
    for i in ents.cities:
        if i.lower() in client_speech:
            out['city'][out['phone'] == phone] = i
            return True

    # find part of city name
    client_speech = client_speech.split(' ')
    for key, value in ents.part_of_cities.items():
        if value.lower() in client_speech:
            out['city'][out['phone'] == phone] = key
            return True

    return False


def find_plate_out_serv(phone):
    """
    find plate by outer service
    """
    try:
        # read image by url
        resp = urlopen(str(out['img_url'][out['phone'] == phone].values[0]))
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        cv2.imwrite('tst.png', img)

        # post request to service
        with open('tst.png', 'rb') as fp:
            response = requests.post(
                'https://platerecognizer.com/plate-reader/',
                files=dict(upload=fp),
                headers={'Authorization': 'Token ' + tokens[np.random.randint(4)]})
        # filling right value
        out['reg_num'][out['phone'] == phone] = response.json()['results'][0]['plate'].upper()
    except:
        # if some error occur we add standard false value
        out['reg_num'][out['phone'] == phone] = 'A A 0 0 A A A'


def find_plate_from_speech(client_speech, phone):
    """
    find client's LNP in his speech, and update inner right pattern if city was found
    :return found --> True/False (finded special pattern or no)
    """
    found = False
    # delete punct and " "
    translation = {ord(x): None for x in string.punctuation}
    client_speech = client_speech.translate(translation)
    clued_str = client_speech.replace(" ", "")

    # find necessary patterns
    plate_format_1 = re.findall(r'[a-z][0-9]{3}[a-z]{3}', clued_str)
    plate_format_2 = re.findall(r'[a-z][0-9]{2}[a-z]{3}', clued_str)

    # if we have 3 digits in client speech we have one pattern and add it
    if plate_format_1:
        out['reg_num'][out['phone'] == phone] = plate_format_1[0].upper()
        found = True
    # if we have 2 digits we have two patterns and find out which one to add
    elif plate_format_2:
        plate_format = re.findall(r'[a-z][0-9]{2}[a-z]{3}', clued_str)  # find let_dig_dig
        tmp_str = plate_format[0][0] + ' ' + plate_format[0][1:3]
        pos_plate = client_speech.find(tmp_str)
        # choose between let_dig_dig_let_let_let and let_let_dig_dig_let_let_let
        if pos_plate == -1:
            tmp_str = plate_format[0][:3]
            pos_plate = client_speech.find(tmp_str)
        if pos_plate == 0 or (
                pos_plate > 2 and client_speech[pos_plate - 3] != ' ' and client_speech[pos_plate - 1] == ' '):
            plate_format_2 = re.findall(r'[a-z][0-9]{2}[a-z]{3}', clued_str)
            out['reg_num'][out['phone'] == phone] = plate_format_2[0].upper()
            found = True
        elif client_speech[pos_plate - 1] != ' ' or pos_plate - 3 == -1 or client_speech[pos_plate - 3] == ' ':
            plate_format_2 = re.findall(r'[a-z]{2}[0-9]{2}[a-z]{3}', clued_str)
            out['reg_num'][out['phone'] == phone] = plate_format_2[0].upper()
            found = True

    return found


def find_plate_by_network(url, req, indx):
    """
    find LP on all images from main url
    """
    counter_for_break = 0
    # BeautifulSoup can failed 4 times from 10, and we can receive unvalid url,
    # if some error occur return standard false plate
    while True:
        # if unvalid url was receivde, we can't parse site and return standard false plate
        if counter_for_break == 10:
            break
        try:
            # parse site
            soup = BeautifulSoup(urlopen(url), "lxml")
            div = soup.find("div", {"class": "fpaImages__mainImage"})
            imgs = div.find_all("img", {"class": "tracking-standard-link"})
            psb_plates = []

            # extract all image urls
            urls = []
            for i in range(len(imgs)):
                try:
                    urls.append(imgs[i]['data-src'])
                except KeyError:
                    urls.append(imgs[i]['src'])

            # to save the most likely image_url
            plate_with_url = {}
            for i in urls:
                # read image from url
                resp = urlopen(i)
                image_url = np.asarray(bytearray(resp.read()), dtype="uint8")
                im_org = cv2.imdecode(image_url, cv2.IMREAD_COLOR)

                # find plates on image by Hoar's casscad, and filter posbl plates remain only plates centrally located
                h, w = im_org.shape[:2]
                im_org = im_org[int(h / 100 * 35): h - 20, 40: w - 40]
                gray = cv2.cvtColor(im_org, cv2.COLOR_BGR2GRAY)
                lower = 0.4
                uppper = 0.6
                plate_area = [gray.shape[1] * lower, gray.shape[1] * uppper]
                plates = cascade.detectMultiScale(gray, scaleFactor=1.1)
                plates = [plates for x, y, w, h in plates if plate_area[0] < (x + x + w) / 2 < plate_area[1]]

                # if we find valid plates, then find plate's value
                if len(plates):
                    with graph.as_default():
                        try:
                            plt = ' '.join(list(Main.main(i)))
                            psb_plates.append(plt)
                            plate_with_url[plt] = i
                        except:
                            psb_plates.append('A A 0 0 A A A')
                            plate_with_url['A A 0 0 A A A'] = i

            # if all plate's values is the same, then return this value
            if len(set(psb_plates)) == 1:
                req['img_url'][indx] = plate_with_url[psb_plates[0]]
                return psb_plates[0]
            # else check whether we have value
            elif len(set(psb_plates)):
                psb_plates = [x for x in psb_plates if x != 'A A 0 0 A A A' and len(x) > 10]
                # delete all standard errors  and check if all another plate's values is the same
                if len(set(psb_plates)) == 1:
                    req['img_url'][indx] = plate_with_url[list(set(psb_plates))[0]]
                    return list(set(psb_plates))[0]
                counted_enters = Counter(psb_plates)
                counted_enters = counted_enters.most_common(2)
                # return most common or longest value
                if counted_enters[0][1] > counted_enters[1][1] or \
                        len(counted_enters[0][0]) >= len(counted_enters[1][0]):

                    req['img_url'][indx] = plate_with_url[counted_enters[0][0]]
                    return counted_enters[0][0]

                else:
                    req['img_url'][indx] = plate_with_url[counted_enters[1][0]]
                    return counted_enters[1][0]
            else:
                return 'A A 0 0 A A A'

        except AttributeError:
            counter_for_break += 1
            continue

    return 'A A 0 0 A A A'
#######################################################################################################################


#                                           collecting and standard operation part
#######################################################################################################################
def collect_twml_record(text, phone_number, rsp, sufix=''):
    """

    :param text: text to say
    :param phone_number: client identificator
    :param rsp: twml respons which need to add record
    :param sufix: '' or '_1' which of function have to recive request
    :return:
    """
    stg = str(out[out.phone == phone_number]['stage'].values[0])  # stage of conversation with certain client
    rsp.say(text)
    rsp.record(finish_on_key='*', play_beep=False, timeout=str(config.timeout), action=ngrok_url + stg + sufix,
               max_length=6)

    return rsp


def collect_redirect(phone):
    """
    create response(main TwiML element)
    """
    global out
    response = VoiceResponse()

    # tell which theme was the last in previous conversation and pass necessary stage
    response.say(phrases.phone_again + subjects_of_stages[out['stage'][out.phone == phone].values[0] - 2])
    out['stage'][out.phone == phone] = out['stage'][out.phone == phone].values[0] - 1
    response.redirect(ngrok_url + str(out['stage'][out.phone == phone].values[0]))
    return response.to_xml()


def collect_dgt_gather(text, phone, sufix=''):
    """
    collect TwiML digit gather that will say TEXT
    """
    stg = str(out[out.phone == phone]['stage'].values[0])
    gather = Gather(input='dtmf', numDigits=str(config.digits_per_phone), timeout=6, action=ngrok_url + stg + sufix,
                    finish_on_key='*')
    gather.say(text)
    return gather


def collect_2gathers_response(text, phone, sufix='', add_step=True, timeout='auto'):
    """
    функция собирает TwiML с gather для получения ответа на вопрос заданый в параметре text.
    stage увеличивается если ответ удовлетворительный и разговор движется к следующему вопросу.
    Params:
        text ==> (str) вопрос на который мы хотим получить ответ.
        sufix ==> (str) '' или '_1' для перехода к побочной ветке диалога
        add_step ==> (bool) если True диалог движется к следующему вопросу
    """

    global out

    if add_step:
        out['stage'][out['phone'] == phone] = 1 + out['stage'][out['phone'] == phone].values[0]

    twiml_response = VoiceResponse()
    twiml_response = collect_twml_record(text, phone, twiml_response, sufix)
    twiml_response = collect_twml_record(text, phone, twiml_response, sufix)
    twiml_response.say(phrases.ddnt_rcv_inp)
    return twiml_response.to_xml()


def collect_keybrd_response(text, phone, add_step=False):
    """
    collect TwiML response with digit gather
    if necessary go to next conversation stage ADD_STEP = TRUE
    """
    if add_step:
        out['stage'][out['phone'] == phone] = 1 + out['stage'][out['phone'] == phone].values[0]

    response = VoiceResponse()
    response.append(collect_dgt_gather(text=text, phone=phone, sufix='_1'))
    response.say(phrases.ddnt_rcv_inp)
    return response.to_xml()


def collect_end_conversation(text):
    """
    функция собирает TwiML в котором бот проговаривает text и ложет трубку.
    Params:
        text ==> (str) текст-прощание с абонентом.
    """

    twiml_response = VoiceResponse()
    twiml_response.say(text)
    twiml_response.pause()
    twiml_response.hangup()
    return twiml_response.to_xml()


def check_for_pos_neg(text, phone):
    """
    проверка, поступил негативный или позитивный ответ
    Params:
        text ==> (str) строка для проверки.
    """

    global out
    out['pst'][out.phone == phone], out['ngt'][out.phone == phone] = (False, False)
    text = text.lower()
    print(text)

    # проверка присутствует ли в фразе одобрение или отрицание
    find_entyties(text=text, case='pst', phone=phone)
    find_entyties(text=text, case='ngt', phone=phone)


def find_entyties(text, case, phone):
    """
    find positive or negative entities in client speech
    case ==> 'pos' or 'neg'
    text ==> client speech
    """
    translation = {ord(x): None for x in string.punctuation}
    text = text.translate(translation).lower()

    # find phrases
    for i in ents.ents[case + '_1']:
        if i in text:
            out[case][out.phone == phone] = True
            break

    # find separate words
    text = text.split(' ')
    for i in ents.ents[case]:
        if i in text:
            out[case][out.phone == phone] = True
            break


def get_pos_neg(client_speech, phone):
    """
    find positive and negative entities in client speech (is it useful function?)
    """
    check_for_pos_neg(client_speech, phone=phone)
    return out['pst'][out.phone == phone].values[0], out['ngt'][out.phone == phone].values[0]


def choose_rigth_answer(positive_text, negative_text, client_speech, phone,
                        end_call=False, timeout='auto', sfx_key=False):
    """
    функция решает как продолжить диалог в зависимости от того, положительно ли клиент ответил на вопрос или нет.
    если положительно, то задаётся следующий вопрос, в противном случае будет задан уточняющий вопрос.
    Params:
        positive_text ==> (str) вопрос на который мы хотим получить ответ.
        negative_text ==> (str) текст который будет озвучен перед окончанием звонка
        positive_hints ==> (str) подсказки для лучшего распознавания ответа на вопрос positive_text
        negative_hints ==> (str) подсказки для лучшего распознавания ответа на вопрос negative_text
        client_speech ==> (str) speech2text ответ клиента
    """

    pos, neg = get_pos_neg(client_speech=client_speech, phone=phone)

    # if we find positive entity in client answer go to next stage
    if pos:
        if sfx_key:
            sufix = '_1'
        else:
            sufix = ''
        twiml_xml = collect_2gathers_response(text=positive_text, phone=phone, sufix=sufix)

    # if find negative  sometimes we have to end call sometimes ask specifying question
    elif neg and end_call:
        set_convrs_key(phone=phone, key=2)
        twiml_xml = collect_end_conversation(negative_text)

    elif neg:
        twiml_xml = collect_2gathers_response(text=negative_text, sufix='_1', add_step=False,
                                              phone=phone, timeout=timeout)
    # if find nothing reask current question
    else:
        # second time repeat current question
        twiml_xml = choose_repeat_qwe(phone=phone, else_text='could you repeat ' +
                                 ents.repeat_subj[out['stage'][out['phone'] == phone].values[0] - 1])

    return str(twiml_xml)


def recognize_audio(phone, form):
    """
    recognize audio by google_recognizer (if some errors occur return error)
    """
    # reading audio by url
    url = form.get('RecordingUrl')
    data = io.BytesIO(urlopen(url).read())

    # prepare data to convenient format
    r = sr.Recognizer()
    with sr.WavFile(data) as source:
        audio_ = r.record(source)

    # recognize audio
    try:
        client_speech = r.recognize_google(audio_)
    except sr.UnknownValueError:
        client_speech = 'UnknownValueError'
    except sr.RequestError:
        try:
            client_speech = r.recognize_sphinx(audio_)
        except sr.UnknownValueError:
            client_speech = 'UnknownValueError'
        except sr.RequestError:
            client_speech = 'RequestError'

    # write client speech to personal file
    write_user_answer(text=client_speech, phone=phone)
    return client_speech


def undrstnd_newcall_recall(phone, form):
    """
    if we call to client not first time, we don't need his answer for first question,
    because we only repeat it for client
    form ==> web form that we reciev from twilio
    """
    global out
    if out['first_ques'][out.phone == phone].values[0]:
        out['first_ques'][out.phone == phone] = False
        return 'YES'
    else:
        client_speech = recognize_audio(phone=phone, form=form)
        return client_speech


def write_user_answer(text, phone):
    """
    write user answer to personal file
    """
    with open(phone, "a") as f:
        f.write("\n" + phrases.stage_content[out[out.phone == phone]['stage'].values[0] - 1] + '\n' + text + "\n\n")


def get_stage_values(form):
    """
    function return client's phone number, speech and if necessary
    form ==> web form that we receive from twilio
    """
    phone = form.get('To')
    return phone, undrstnd_newcall_recall(phone=phone, form=form)


def choose_repeat_qwe(phone, else_text, sufix=''):
    """
    choose what repeat phrase to say (standard cld_u_rpt || cld_u_rpt with current qwe)
    one of two time repeat current question
    """
    global out
    if out['repeat_qwe'][out['phone'] == phone].values[0]:
        text = phrases.cld_u_rpt
        out['repeat_qwe'][out['phone'] == phone] = False
    else:
        text = else_text
        out['repeat_qwe'][out['phone'] == phone] = True
    twiml_xml = collect_2gathers_response(text=text, phone=phone, add_step=False, sufix=sufix)

    return twiml_xml


#                                               seters
###############################################################################################################
def set_call_day(phone):
    """
    set day when we called to client
    """
    global out
    out['call_day'][out.phone == phone] = datetime.utcnow().day


def set_first_qwe(phone, key):
    """
    set it's first question in conversation or no
    """
    global out
    out['first_ques'][out.phone == phone] = key


def set_convrs_key(phone, key):
    """
    set client state 0-need call, 1-need recall, 2-don't call for it number, 3-completed conversation
    """
    global out
    out['cnvrs_key'][out.phone == phone] = key


def set_num_calls(phone):
    """
    increase number of calls to this number
    """
    global out
    out['num_calls'][out.phone == phone] += 1


@app.route('/set_urls/<url>')  # c798f811.ngrok.io --> url example
def set_urls(url):
    """
    set new url path
    """
    global ngrok_url

    config.main_url = 'https://' + url + '/'
    config.recive_url = config.main_url + 'receiver'

    config.qwestion_url = config.main_url + 'question'
    ngrok_url = config.qwestion_url
    return 'success', 200


@app.route('/set_phone_4_test/<phone>')  # +44..
def set_phone_4_test(phone):
    """
    set new phone_for_test
    """
    config.phone_for_test = phone
    return 'success', 200


@app.route('/set_timeout/<timeout>')
def set_timeout(timeout):
    """
    set new timeout (time of silence that twilio wait before stop record)
    """
    config.timeout = timeout
    return 'success', 200


@app.route('/set_digits_per_phone/<digits>')
def set_digits_per_phone(digits):
    """
    set new digits_per_phone (how much digits have to contain phone number in concrete country)
    """
    config.digits_per_phone = digits
    return 'success', 200


@app.route('/set_sleep_min/<sleep_min>')
def set_sleep_min(sleep_min):
    """
    set new sleep_min (time beetwen queues of calls)
    """
    config.sleep_min = sleep_min
    return 'success', 200


@app.route('/set_port/<port>')
def set_port(port):
    """
    set new port
    """
    config.port = port
    return 'success', 200


@app.route('/set_host/<host>')
def set_host(host):
    """
    set new host
    """
    config.host = host
    return 'success', 200


@app.route('/set_twilio_params', methods=['GET'])
def set_twilio_params():
    req = request.args.to_dict()
    config.account_sid = req['sid']
    config.auth_token = req['token']
    return 'success', 200


#                                                   question part
######################################################################################################################
@app.route('/question1', methods=['POST'])
def question1():
    """
    обрабатывается вопрос: вы всё ещё продаёте машину?
    задаётся вопрос: интересно ли клиенту предложение.
    """
    phone, client_speech = get_stage_values(request.form)
    set_convrs_key(phone, 1)

    # choose what twilio will say
    return choose_rigth_answer(end_call=True, positive_text=phrases.first_stage,
                               phone=phone, client_speech=client_speech, negative_text=phrases.sorry_4_bothr)


@app.route('/question2', methods=['POST'])
def question2():
    """
    обрабатывается вопрос: интересно ли клиенту предложение.
    задаётся вопрос:
        Да --> это ваш регестрационный номер.
        Нет --> кладётся трубка.
    """

    phone, client_speech = get_stage_values(request.form)

    # if we have standard error plate value, we ask to dictate client right number
    # if we have psbl value we specify it asking
    if 'A A 0 0 A A A' == str(out[out.phone == phone]['reg_num'].values[0]):
        sfx = True
        txt = 'Please dictate registration number of your car'
    else:
        sfx = False
        txt = 'Can I confirm your vehicle registration number is '+str(out[out.phone == phone]['reg_num'].values[0])+'?'

    return choose_rigth_answer(client_speech=client_speech, negative_text=phrases.sorry_4_bothr, positive_text=txt,
                               phone=phone, end_call=True, sfx_key=sfx)


@app.route('/question3', methods=['POST'])
def question3():
    """
    обрабатывается вопрос: это ваш регестрационный номер. / правильно ли я понял ваш номер?
    задаётся вопрос:
        Да --> это ваш пробег
        Нет --> могу я уточнить ваш регестрационный номер
    """
    global out

    phone, client_speech = get_stage_values(request.form)

    pos, neg = get_pos_neg(client_speech=client_speech, phone=phone)

    nex_qwe = 'Can I confirm the mileage as ' + str(out['mileage'][out.phone == phone].values[0]) + '?'

    if pos:
        out['reg_num'][out['phone'] == phone] = ''.join(str(out['reg_num'][out['phone'] == phone].values[0]).split(' '))
        twiml_xml = collect_2gathers_response(text=nex_qwe, phone=phone)

    elif neg:
        find_plate_out_serv(phone=phone)
        twiml_xml = collect_2gathers_response(phone=phone, text=nex_qwe)

    else:
        twiml_xml = choose_repeat_qwe(phone=phone, else_text='could you repeat ' +
                                 str(out['reg_num'][out['phone'] == phone].values[0]) + ' is your number?')
        return str(twiml_xml)

    if out['mileage'][out['phone'] == phone].values[0] == 0:
        twiml_xml = collect_keybrd_response(text=phrases.keybrd_inp_ml, phone=phone, add_step=True)

    return str(twiml_xml)


@app.route('/question3_1', methods=['GET', 'POST'])
def question3_1():
    """
    обрабатывается вопрос:  могу я уточнить ваш регестрационный номер
    задаётся вопрос:
        Нашел --> правильно ли я понял ваш номер?
        неНашел --> могли бы вы повторить ответ
        Нет --> конец разговора.
    """

    global out

    phone = request.form.get('To')

    client_speech = recognize_audio(phone=phone, form=request.form)
    found = find_plate_from_speech(client_speech=client_speech, phone=phone)
    next_qwe = 'Can I confirm the mileage as ' + str(out['mileage'][out.phone == phone].values[0]) + '?'

    if found:
        out[out.phone == phone]["again_key"] = True
        twiml_xml = collect_2gathers_response(phone=phone, text=next_qwe)
    else:
        twiml_xml = choose_repeat_qwe(phone=phone, sufix='_1', else_text='could you repeat ' +
                                 ents.repeat_subj[out['stage'][out['phone'] == phone].values[0] - 1])

    return str(twiml_xml)


@app.route('/question4', methods=['POST'])
def question4():
    """
    обрабатывается вопрос: это ваш пробег. / правильно ли я понял ваш пробег?
    задаётся вопрос:
        Да --> вы живёте в этом городе?
        Нет --> могу я уточнить ваш пробег
    """

    phone, client_speech = get_stage_values(form=request.form)
    pos, neg = get_pos_neg(client_speech=client_speech, phone=phone)

    next_qwe = 'Can I just confirm you live in ' + str(out['city'][out.phone == phone].values[0]) + '?',

    # при одобрении задаётся следующий вопрос    'Please dictate registration number of your car'
    if pos:
        twiml_xml = check_city_is_nan(phone, next_qwe)

    # при отрицании прошу ввести номер телефона
    elif neg:
        twiml_xml = collect_keybrd_response(text=phrases.keybrd_inp_ml, phone=phone)

    # если не нашел никакой реакции переспроси
    else:
        twiml_xml = choose_repeat_qwe(phone=phone, else_text='could you repeat ' +
                                 str(out['mileage'][out['phone'] == phone].values[0]) + ' is your mileage?')

    return str(twiml_xml)


@app.route('/question4_1', methods=['POST'])
def question4_1():
    """
    обрабатывается вопрос:  могу я уточнить ваш пробег
    задаётся вопрос:
        Нашел --> правильно ли я понял ваш пробег?
        неНашел --> могли бы вы повторить ответ
        Нет --> конец разговора.
    """

    global out

    client_inp = request.form.get('Digits')  # получаю введённый с клавиатуры мобильный
    phone = request.form.get('To')
    write_user_answer(text='dictate: ' + client_inp, phone=phone)

    out['mileage'][out.phone == phone] = client_inp

    nex_qwe = 'Can I just confirm you live in ' + str(out['city'][out.phone == phone].values[0]) + '?',
    twiml_xml = check_city_is_nan(phone, nex_qwe)

    return str(twiml_xml)


def check_city_is_nan(phone, txt):
    """
    if city is NaN, will ask another question
    """
    if pd.isna(out['city'][out['phone'] == phone].values)[0]:
        twiml_xml = collect_2gathers_response(text='Please dictate the nearest city to you.', sufix='_1', phone=phone)
    else:
        twiml_xml = collect_2gathers_response(text=txt, phone=phone)
    return twiml_xml


@app.route('/question5', methods=['POST'])
def question5():
    """
    обрабатывается вопрос: вы живёте в этом городе? / правильно ли я понял вы живёте в?
    задаётся вопрос:
        Да --> это ваш мобильный?
        Нет --> могу ли я уточнить ближайший город?
    """

    phone, client_speech = get_stage_values(request.form)

    # проверка не задаём ли мы данный вопрос повторно (если да то вопросительная фраза другая)
    if out['again_key'][out.phone == phone].values[0]:
        ngt_txt = 'could you talk ' + subjects_of_stages[out['stage'][out.phone == phone].values[0] - 2] + ' again?'
    else:
        ngt_txt = 'Please dictate ' + subjects_of_stages[out['stage'][out.phone == phone].values[0] - 2]

    pos, neg = get_pos_neg(client_speech=client_speech, phone=phone)

    if pos:
        twiml_xml = collect_2gathers_response(text=phrases.fivth_stage, phone=phone)

    elif neg:
        twiml_xml = collect_2gathers_response(text=ngt_txt, sufix='_1', add_step=False,
                                              phone=phone, timeout='3')

    else:
        twiml_xml = choose_repeat_qwe(phone=phone, else_text='could you repeat, do you live in ' +
                                      str(out['city'][out.phone == phone].values[0]) + '?')

    return str(twiml_xml)


@app.route('/question5_1', methods=['POST'])
def question5_1():
    """
    обрабатывается вопрос:  могу я уточнить ближайший город?
    задаётся вопрос:
        Нашел --> правильно ли я понял ваш ГОРОД?
        неНашел --> могли бы вы повторить ответ
        Нет --> конец разговора.
    """

    global out
    phone = request.form.get('To')
    client_speech = recognize_audio(phone=phone, form=request.form)

    found = find_city(client_speech=client_speech, phone=phone)
    next_qwe = 'Can I validate, you live in ' + str(out['city'][out.phone == phone].values[0]) + '?'

    if found:
        out[out.phone == phone]["again_key"] = True
        twiml_xml = collect_2gathers_response(add_step=False, phone=phone, text=next_qwe)

    else:
        twiml_xml = choose_repeat_qwe(phone=phone, sufix='_1', else_text='could you repeat ' +
                                      ents.repeat_subj[out['stage'][out['phone'] == phone].values[0] - 1])

    return str(twiml_xml)


@app.route('/question6', methods=['POST'])
def question6():
    """
    обрабатывается вопрос: это ваш мобильный?
    задаётся вопрос:
        Да --> какая у вас сервисная история?
        Нет --> введите мобильный с помощью клавиатуры.
    """

    phone, client_speech = get_stage_values(form=request.form)
    pos, neg = get_pos_neg(client_speech=client_speech, phone=phone)

    # при одобрении задаётся следующий вопрос
    if pos:
        out['phone_for_offer'][out.phone == phone] = phone
        twiml_xml = collect_2gathers_response(text=phrases.sixth_stage, phone=phone)

    # при отрицании прошу ввести номер телефона
    elif neg:
        twiml_xml = collect_keybrd_response(text=phrases.keybrd_inp_ph, phone=phone)
    # если не нашел никакой реакции переспроси
    else:
        twiml_xml = choose_repeat_qwe(phone=phone, else_text='could you repeat ' +
                                      ents.repeat_subj[out['stage'][out['phone'] == phone].values[0] - 1])

    return str(twiml_xml)


@app.route('/question6_1', methods=['POST'])
def question6_1():
    """
    обрабатывается вопрос: введите мобильный с помощью клавиатуры.
    задаётся вопрос:
        Нашел --> какая у вас сервисная история?
        неНашел --> могли бы вы повторить ответ
    """

    global out

    client_inp = request.form.get('Digits')  # получаю введённый с клавиатуры мобильный
    phone = request.form.get('To')
    write_user_answer(text='dictate: ' + client_inp, phone=phone)

    if len(client_inp) == config.digits_per_phone:
        out['phone_for_offer'][out.phone == phone] = "+" + client_inp
        twiml_xml = collect_2gathers_response(text=phrases.sixth_stage, phone=phone)

    else:
        twiml_xml = collect_keybrd_response(text=phrases.nt_vld_input, phone=phone)

    return str(twiml_xml)


@app.route('/question7', methods=['POST'])
def question7():
    """
    обрабатывается вопрос: какая у вас сервисная история?
    задаётся вопрос:
        удовлетворительный ответ --> вы примите оффер?
    """

    global out

    phone = request.form.get('To')
    if out['cnvrs_key'][out.phone == phone].values[0] == 1 and out['first_ques'][out.phone == phone].values[0]:
        found = True
    else:
        phone = request.form.get('To')
        client_speech = recognize_audio(phone=phone, form=request.form)
        found = find_service_hist(client_speech, phone)

    # если найдено одно из допустимых значений сервисной истории --> следующий вопрос
    if found:
        twiml_xml = collect_2gathers_response(text=phrases.seventh_stage, phone=phone)
        set_convrs_key(phone, 3)

    # иначе вопрос задаётся повторно
    else:
        twiml_xml = choose_repeat_qwe(phone=phone, else_text='could you repeat ' +
                                      ents.repeat_subj[out['stage'][out['phone'] == phone].values[0] - 1])

    return str(twiml_xml)


@app.route('/question8', methods=['POST'])
def question8():
    """
     обрабатывается вопрос: вы примите оффер?
     бот произносит прощание, разговор кончается
    """
    global out

    phone = request.form.get('To')
    client_speech = recognize_audio(phone=phone, form=request.form)
    pos, neg = get_pos_neg(client_speech=client_speech, phone=phone)

    if pos:
        out['accept'][out.phone == phone] = True
    elif neg:
        out['accept'][out.phone == phone] = False

    twiml_xml = collect_end_conversation(phrases.u_can_vld_it)
    send_inf(phone=phone)

    return str(twiml_xml)


########################################################################################################


#                                                   call part
#######################################################################################################################
def del_person():
    """
    deleting records that have 2-3 convers key, adding them to history
    """
    global out
    hist_data = pd.read_csv('history.csv', converters={"phone": str, 'phone for offer': str})

    unsend_inf = config.unwanted_inf
    not_lack_call = out[out['cnvrs_key'] > 1].drop(unsend_inf, axis=1)
    out = out.drop(out[out['cnvrs_key'] > 1].index)
    too_many_calls = out[out['num_calls'] >= config.last_call].drop(unsend_inf, axis=1)
    out = out.drop(out[out['num_calls'] >= config.last_call].index)
    hist_data = pd.concat([hist_data, not_lack_call, too_many_calls], ignore_index=True)
    hist_data.to_csv('history.csv', index=False)


@app.route('/send', methods=['POST'])
def send_inf(phone):
    """
    sending information about persons that end call
    """
    # drop unuseful inf
    unsend_inf = config.unwanted_for_webform_inf
    pers = out[out.phone == phone].drop(unsend_inf, axis=1)
    # form json and send it
    pers = pers.to_dict('records')[0]
    requests.post(config.recive_url, json=pers)


@app.route('/receiver', methods=['POST'])
def receiver():
    """
    test func
    """
    print(request.get_json())


def initiate_call(twiml, phone):
    """
    initiate call by twilio
    """
    try:
        # decide which twilio number choose to call
        outbound_num = config.twilio_numbers['Eng']

        if phone.find('+44028') != -1:
            outbound_num = config.twilio_numbers['Irl']

        desire_city = out['city'][out.phone == phone].values[0]
        for i in ents.scottish_city:
            if i in desire_city:
                outbound_num = config.twilio_numbers['Sco']
                break

        for i in ents.welsh_city:
            if i in desire_city:
                outbound_num = config.twilio_numbers['Wel']
                break

        client.calls.create(url=echo_url + urlencode({'Twiml': twiml}), to=phone, from_=outbound_num)
    except:
        print('')


@app.route('/test_call', methods=['POST', 'GET'])
def test_call():
    """
    test func
    """
    twiml = collect_2gathers_response(text=phrases.greeting, phone=config.phone_for_test, add_step=False)
    client.calls.create(url=echo_url + urlencode({'Twiml': twiml}), to=config.phone_for_test,
                        from_=config.twilio_numbers['Eng'])


@app.route('/call_auto')
def call_auto():
    """
    main function to automatic calls
    if it is unwanted time to call we update inner Db, else make calls
    """
    global exel_updated

    while True:
        if config.lwr_time_lim <= datetime.utcnow().hour < config.upr_time_lim:
            make_calls()
            time.sleep(config.sleep_min * 60)
        else:
            del_person()
            if exel_updated:
                exel_updated = False
                parse_exel()
            time.sleep(config.sleep_min * 2 * 60)


@app.route('/make_calls', methods=['POST'])
def make_calls():
    """
    initiate calls if have suitable records
    """
    global quiq_recall_numbers, exel_updated
    # recall clients that didn't finish conversation
    num_calls = np.max([config.num_calls_per_time - quiq_recalls(), 2])

    # find clients for new call and recall, filtered client with many calls and if we call client today
    today = datetime.utcnow().day
    fltrd_for_recall = out[out.cnvrs_key == 1][abs(out.call_day - today) >= config.recall_day_step] \
        [out.num_calls < config.last_call]
    fltrd_for_call = out[out.cnvrs_key == 0][abs(out.call_day - today) >= config.recall_day_step] \
        [out.num_calls < config.last_call]
    candidates_to_call = pd.concat([fltrd_for_call, fltrd_for_recall])

    if len(candidates_to_call):

        # find number of recalls for this time
        if len(fltrd_for_call) >= int(round(num_calls / 100 * config.percent_of_new)):
            number_of_client = int(round(num_calls / 100 * (100 - config.percent_of_new)))
        else:
            number_of_client = int(round(num_calls / 100 * (100 - len(fltrd_for_call) * 100 / num_calls)))

        for i in range(1, -1, -1):
            # calls to one of categories (new call/recall)
            phone_num_list = candidates_to_call[out.cnvrs_key == i]['phone'].values
            shuffle(phone_num_list)
            phone_num_list = phone_num_list[:number_of_client]
            number_of_client = len(phone_num_list)

            for phone in phone_num_list:
                # set constants
                quiq_recall_numbers.append(phone)
                set_num_calls(phone)
                set_call_day(phone)

                if i == 1:
                    set_first_qwe(phone=phone, key=True)
                    twiml_xml = collect_redirect(phone=phone)
                else:
                    set_first_qwe(phone=phone, key=False)
                    twiml_xml = collect_2gathers_response(text=phrases.greeting, phone=phone, add_step=False)

                initiate_call(twiml=twiml_xml, phone=phone)

            number_of_client = num_calls - number_of_client  # calculate number of new calls for this time
    else:
        # if we haven't candidates to call we start to parse exel
        if exel_updated:
            exel_updated = False
            parse_exel()


def quiq_recalls():
    """
    recall to clients that didn't pick up the phone or stoped call before conversation ends.
    """
    global quiq_recall_numbers
    num_calls = 0

    for phone in quiq_recall_numbers:

        # choose only clients that didn't finish call
        if out['cnvrs_key'][out.phone == phone].values[0] == 1 or out['cnvrs_key'][out.phone == phone].values[0] == 0:
            set_num_calls(phone)
            set_call_day(phone)

            # there is diffrnt phrases to clients that didn't pick up the phone and stoped call be4 conversation ends
            twiml_xml = ''
            if out['cnvrs_key'][out.phone == phone].values[0] == 1:
                set_first_qwe(phone=phone, key=True)
                twiml_xml = collect_redirect(phone=phone)
            elif out['cnvrs_key'][out.phone == phone].values[0] == 0:
                set_first_qwe(phone=phone, key=False)
                twiml_xml = collect_2gathers_response(text=phrases.greeting, phone=phone, add_step=False)
            initiate_call(twiml=twiml_xml, phone=phone)
            num_calls += 1

    quiq_recall_numbers.clear()
    return num_calls


######################################################################################################################
@app.route('/snd', methods=['POST'])
def snd():
    """
    test func
    """
    pers = {'img_url': 'http://www.letchworthmini.co.uk/s/cc_images/cache_71011477.JPG', 'phone': '+9379992'}
    requests.post(config.main_url + 'get_data', json=pers)
    return 'success', 200


@app.route('/get_data', methods=['POST'])
def get_data():
    global out, exel_updated, exel_doc_counter

    # save exel
    try:
        # download it from html Form
        xls = pd.ExcelFile(request.files['ex'])
        df = xls.parse(xls.sheet_names[0], converters={"phone": str})

        # save it named like number of downloaded exels
        writer = pd.ExcelWriter(str(exel_doc_counter) + '.xlsx')
        df.to_excel(writer, 'Sheet1')
        writer.save()
        exel_doc_counter += 1

        exel_updated = True
        return "SUCCESS"
    # if we receive json with one record --> parse json
    except:
        photo_url = request.get_json()['img_url']

        with graph.as_default():
            c = Main.main(photo_url)
        req = request.get_json()

        # adding fields necessary to innerDb to json
        for i in range(len(config.adding_filds)):
            if i < 3:
                req[config.adding_filds[i]] = 0
            elif i < 8:
                req[config.adding_filds[i]] = False
            elif i == 8:
                req[config.adding_filds[i]] = 1
            else:
                req[config.adding_filds[i]] = None
        req['reg_num'] = c

        values = [[req[k]] for k in config.flds]
        values = np.array(values).T
        # add it to Db
        dt = pd.DataFrame(data=values, columns=config.flds)
        out = pd.concat([out, dt], ignore_index=True)

        return "SUCCESS"


def parse_exel():
    global out, exel_doc_counter
    # handle all available exels
    for i in range(exel_doc_counter):
        # loading exel and rename some columns
        xls = pd.ExcelFile(str(i) + '.xlsx')
        df = xls.parse(xls.sheet_names[0], converters={"Phone": str, "Mileage": int, 'Location': str})

        df = df.dropna(subset=['Phone'])

        df = df[['Phone', 'Featured Image', 'Mileage', 'Location']].rename(columns={'Phone': 'phone',
                                                                                    'Mileage': 'mileage',
                                                                                    'Location': 'city',
                                                                                    'Featured Image': 'img_url'})
        # add + to phones
        df['phone'] = '+44' + df['phone']

        # delete United Kingdom from city names
        for city_name in df['city']:
            indx = city_name.find('United Kingdom')
            if indx != -1:
                df['city'][df.city == city_name] = city_name[:indx]

        req = df.to_dict()

        # adding reg num to all cars recognized it from image
        req['reg_num'] = {}
        for j in range(len(req['phone'])):
            try:
                req['reg_num'][j] = find_plate_by_network(req['img_url'][j], req, j)
            except:
                req['reg_num'][j] = 'A A 0 0 A A A'

        # adding fields necessary to innerDb to json
        req = filling_dict(req=req)

        df = pd.DataFrame.from_dict(req)
        # fill nan to 0 in mileage and convert it to int from float
        df['mileage'] = df['mileage'].fillna(0)
        df['mileage'] = df['mileage'].astype('int64', errors='ignore')

        df = find_clients_with_2_cars(df=df)
        # add current exel data to innerDb
        out = pd.concat([out, df], ignore_index=True)

    # set number of available exels to handle to 0
    exel_doc_counter = 0


def filling_dict(req):
    """
    adding all necessary columns
    param ==> req dict to filling
    """
    # adding fields necessary to innerDb to json
    for j in range(len(config.adding_filds)):
        if j < 3:
            req = add_field_to_dict(0, j, req)

        elif j < 8:
            req = add_field_to_dict(False, j, req)

        elif j == 8:
            req = add_field_to_dict(1, j, req)

        else:
            req = add_field_to_dict(None, j, req)

    return req


def find_clients_with_2_cars(df):
    """
    if client sell more than one car we add another all his records except one to archive,
    and get another one when handel first his record
    """
    two_cars_clients = pd.read_csv('two_cars_clienet.csv', converters={"phone": str, 'phone for offer': str,
                                                                      'mileage': int})

    df = pd.concat([two_cars_clients, df], ignore_index=True)

    # find clients with 2 cars in innerDb
    new_clients = df.phone.values
    filtred_client = out[:0]
    for some_client in new_clients:
        if some_client in out.phone.values:
            filtred_client = pd.concat([filtred_client, df[df.phone == some_client]], ignore_index=True)
            df = df.drop(df[df.phone == some_client].index)

    # find clients with 2 cars in current exel
    new_clients = Counter(df.phone.values).most_common()
    for some_client in new_clients:
        if some_client[1] < 2:
            break
        else:
            filtred_client = pd.concat([filtred_client, df[df.phone == some_client[0]][1:]], ignore_index=True)
            df = df.drop(df[df.phone == some_client[0]].index[1:])
    filtred_client.to_csv('two_cars_client.csv', index=False)
    return df


def add_field_to_dict(value, pos_field, req):
    """
    add field with concrete value to dict
    value ==> what to add to column
    pos_field ==> which column to add to Df
    req ==> dict to adding
    :return utdated dict
    """
    req[config.adding_filds[pos_field]] = {}
    # adding value to all cllients
    for j in range(len(req['phone'])):
        req[config.adding_filds[pos_field]][j] = value
    return req


@app.route('/i')
def index():
    """
    web-form to load exel with data
    :return:
    """
    return '<form action="' + config.main_url + """get_data" method="post" enctype="multipart/form-data">
                <input type="file" name="ex">
                <input type="submit" name="submit">
                </form>"""


@app.route('/ir')
def ir():
    """
    test func to print innerDb
    """
    print(out)
    return 'success', 200


if __name__ == '__main__':
    app.run(host=config.host, port=config.port, threaded=True)
