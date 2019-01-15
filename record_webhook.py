import re
from bs4 import BeautifulSoup
import string
from random import shuffle
from flask import Flask, request
from ents import ents, subjects
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

# for comfortble debug
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Account Sid and Auth Token from twilio.com/console
client = Client(config.account_sid, config.auth_token)

# start url for twiML
echo_url = config.echo_url
start_url = config.start_url

# main url for all requests
ngrok_url = config.ngrok_url
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
    {"reg_num": [''], "mileage": [10000], "city": [''], "phone": [''], "phone_for_offer": [''], 'num_calls': [9],
     "serv_hist": [''], 'again_key': [True], 'stage': [1], 'pst': [True], 'ngt': [True], 'cnvrs_key': [3],
     'first_ques': [False], 'call_day': [0], 'accept': [None], 'img_url': [''], 'repeat_qwe': [True]})

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
        out['reg_num'][out['phone'] == phone] = 'AA00AAA'


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


def find_plate_by_network(url):
    counter_for_break = 0
    while True:
        if counter_for_break == 100:
            break
        try:
            soup = BeautifulSoup(urlopen(url), "lxml")
            div = soup.find("div", {"class": "fpaImages__mainImage"})
            imgs = div.find_all("img", {"class": "tracking-standard-link"})
            psb_plates = []

            urls = []  # imgs[0]['src']
            for i in range(len(imgs)):
                try:
                    urls.append(imgs[i]['data-src'])
                except KeyError:
                    urls.append(imgs[i]['src'])

            for i in urls:
                resp = urlopen(i)
                image_url = np.asarray(bytearray(resp.read()), dtype="uint8")
                im_org = cv2.imdecode(image_url, cv2.IMREAD_COLOR)

                h, w = im_org.shape[:2]
                im_org = im_org[int(h / 100 * 35): h - 20, 40: w - 40]
                gray = cv2.cvtColor(im_org, cv2.COLOR_BGR2GRAY)
                lower = 0.4
                uppper = 0.6
                plate_area = [gray.shape[1] * lower, gray.shape[1] * uppper]
                plates = cascade.detectMultiScale(gray, scaleFactor=1.1)
                plates = [plates for x, y, w, h in plates if plate_area[0] < (x + x + w) / 2 < plate_area[1]]
                if len(plates):
                    # cv2.imshow('ll', im_org)
                    # cv2.waitKey(0)
                    with graph.as_default():
                        # try:
                        psb_plates.append(' '.join(list(Main.main(i))))
                        # except:
                        #     psb_plates.append('A A 0 0 A A A')

            if len(set(psb_plates)) == 1:
                return psb_plates[0]
            elif len(set(psb_plates)):
                psb_plates = [x for x in psb_plates if x != 'A A 0 0 A A A' and len(x) > 10]
                if len(set(psb_plates)) == 1:
                    return list(set(psb_plates))[0]
                counted_enters = Counter(psb_plates)
                counted_enters = counted_enters.most_common(2)

                if counted_enters[0][1] > counted_enters[1][1] or \
                        len(counted_enters[0][0]) >= len(counted_enters[1][0]):
                    return counted_enters[0][0]
                else:
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
    rsp.record(finish_on_key='*', play_beep=False, timeout=config.timeout, action=ngrok_url + stg + sufix, max_length=6)

    return rsp


def collect_redirect(phone):
    """

    """
    global out
    response = VoiceResponse()

    response.say(phrases.phone_again + subjects[out['stage'][out.phone == phone].values[0] - 2])

    response.redirect(ngrok_url + str(out['stage'][out.phone == phone].values[0] - 1))
    out['stage'][out.phone == phone] = out['stage'][out.phone == phone].values[0] - 1

    return response.to_xml()


def collect_dgt_gather(text, phone, sufix=''):
    """

    """
    stg = str(out[out.phone == phone]['stage'].values[0])
    gather = Gather(input='dtmf', numDigits="10", timeout=6, action=ngrok_url + stg + sufix, finish_on_key='*')
    gather.say(text)
    return gather


def collect_2gathers_response(text, phone, sufix='', add_step=True, timeout='auto'):
    """
    функция собирает TwiML с gather для получения ответа на вопрос заданый в параметре text.
    stage увеличивается если ответ удовлетворительный и разговор движется к следующему вопросу.
    Params:
        text ==> (str) вопрос на который мы хотим получить ответ.
        hints ==> (str) подсказки для лучшего распознавания ответа на вопрос text
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

    """
    translation = {ord(x): None for x in string.punctuation}
    text = text.translate(translation).lower()

    for i in ents.ents[case + '_1']:
        if i in text:
            out[case][out.phone == phone] = True
            break

    text = text.split(' ')
    for i in ents.ents[case]:
        if i in text:
            out[case][out.phone == phone] = True
            break


def get_pos_neg(client_speech, phone):
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

    if pos:
        if sfx_key:
            sufix = '_1'
        else:
            sufix = ''
        twiml_xml = collect_2gathers_response(text=positive_text, phone=phone, sufix=sufix)

    elif neg and end_call:
        set_convrs_key(phone=phone, key=2)
        twiml_xml = collect_end_conversation(negative_text)

    elif neg:
        twiml_xml = collect_2gathers_response(text=negative_text, sufix='_1', add_step=False,
                                              phone=phone, timeout=timeout)

    else:
        if out['repeat_qwe'][out['phone'] == phone].values[0]:
            text = phrases.cld_u_rpt
            set_repeat_qwe(phone=phone, key=False)

        else:
            text = 'could you repeat ' + ents.repeat_subj[out['stage'][out['phone'] == phone].values[0] - 1]
            set_repeat_qwe(phone=phone, key=True)

        twiml_xml = collect_2gathers_response(text=text, phone=phone, add_step=False)

    return str(twiml_xml)


def undrstnd_newcall_recall(phone, form):
    """

    """
    global out
    if out['first_ques'][out.phone == phone].values[0]:
        out['first_ques'][out.phone == phone] = False
        return 'YES'
    else:
        url = form.get('RecordingUrl')
        data = io.BytesIO(urlopen(url).read())
        r = sr.Recognizer()

        with sr.AudioFile(data) as source:
            audio_ = r.record(source)
        try:
            client_speech = r.recognize_google(audio_)
        except sr.UnknownValueError:
            client_speech = 'repeat'
        except sr.RequestError:
            client_speech = 'repeat'
        write_user_answer(text=client_speech, phone=phone)
        return client_speech


def undrstnd_newques_reask(phone):
    """

    """
    global out
    # проверка не задаём ли мы данный вопрос повторно (если да то вопросительная фраза другая)
    if out['again_key'][out.phone == phone].values[0]:
        out['again_key'][out.phone == phone] = False
        return 'could you talk ' + subjects[out['stage'][out.phone == phone].values[0] - 2] + ' again?'
    else:
        return 'Please dictate ' + subjects[out['stage'][out.phone == phone].values[0] - 2]


def write_user_answer(text, phone):
    with open(phone, "a") as f:
        f.write("\n" + phrases.stage_content[out[out.phone == phone]['stage'].values[0] - 1] + '\n' + text + "\n\n")


def get_stage_values(form, lack_ngt_text=True):
    phone = form.get('To')
    if lack_ngt_text:
        return phone, undrstnd_newcall_recall(phone=phone, form=form), undrstnd_newques_reask(phone=phone)
    else:
        return phone, undrstnd_newcall_recall(phone=phone, form=form)


###############################################################################################################


def set_call_day(phone):
    """

    """
    global out
    out['call_day'][out.phone == phone] = datetime.utcnow().day


def set_first_qwe(phone, key):
    """

    """
    global out
    out['first_ques'][out.phone == phone] = key


def set_convrs_key(phone, key):
    """

    """
    global out
    out['cnvrs_key'][out.phone == phone] = key


def set_num_calls(phone):
    global out
    out['num_calls'][out.phone == phone] += 1


def set_repeat_qwe(phone, key):
    global out
    out['repeat_qwe'][out['phone'] == phone] = key


@app.route('/set_urls/<urlo>')  # работа с вебом
def set_urls(urlo):
    global start_url, ngrok_url

    config.main_url = 'https://' + urlo + '/'
    config.recive_url = config.main_url + 'receiver'

    config.ngrok_url = config.main_url + 'question'
    config.start_url = config.ngrok_url + '1'

    start_url = config.start_url
    ngrok_url = config.ngrok_url


#                                                   question part
######################################################################################################################
@app.route('/question1', methods=['POST'])  # работа с вебом
def question1():
    """
    обрабатывается вопрос: вы всё ещё продаёте машину?
    задаётся вопрос: интересно ли клиенту предложение.
    """
    phone, client_speech = get_stage_values(request.form, lack_ngt_text=False)
    set_convrs_key(phone, 1)

    # twilio выбор ответа
    return choose_rigth_answer(end_call=True, positive_text=phrases.first_stage,
                               phone=phone, client_speech=client_speech, negative_text=phrases.sorry_4_bothr)


@app.route('/question2', methods=['POST'])  # работа с вебом
def question2():
    """
    обрабатывается вопрос: интересно ли клиенту предложение.
    задаётся вопрос:
        Да --> это ваш регестрационный номер.
        Нет --> кладётся трубка.
    """

    phone, client_speech = get_stage_values(request.form, lack_ngt_text=False)

    if 'A A 0 0 A A A' == str(out[out.phone == phone]['reg_num'].values[0]):
        return choose_rigth_answer(client_speech=client_speech, negative_text=phrases.sorry_4_bothr,
                                   positive_text='Please dictate registration number of your car',
                                   phone=phone, end_call=True, sfx_key=True)

    else:
        # twilio выбор ответа
        return choose_rigth_answer(client_speech=client_speech, negative_text=phrases.sorry_4_bothr,
                                   positive_text='Can I confirm your vehicle registration number is ' + str(
                                       out[out.phone == phone]['reg_num'].values[0]) + '?',
                                   phone=phone, end_call=True)


@app.route('/question3', methods=['POST'])  # работа с вебом
def question3():
    """
    обрабатывается вопрос: это ваш регестрационный номер. / правильно ли я понял ваш номер?
    задаётся вопрос:
        Да --> это ваш пробег
        Нет --> могу я уточнить ваш регестрационный номер
    """
    global out

    phone, client_speech, ngt_txt = get_stage_values(request.form)

    pos, neg = get_pos_neg(client_speech=client_speech, phone=phone)

    pst_txt = 'Can I confirm the mileage as ' + str(out['mileage'][out.phone == phone].values[0]) + '?'

    if pos:
        out['reg_num'][out['phone'] == phone] = ''.join(str(out['reg_num'][out['phone'] == phone].values[0]).split(' '))
        twiml_xml = collect_2gathers_response(text=pst_txt, phone=phone)

    elif neg:
        find_plate_out_serv(phone=phone)
        twiml_xml = collect_2gathers_response(phone=phone, text=pst_txt)

    else:
        if out['repeat_qwe'][out['phone'] == phone].values[0]:
            text = phrases.cld_u_rpt
            set_repeat_qwe(phone=phone, key=False)

        else:
            text = 'could you repeat ' + str(out['reg_num'][out['phone'] == phone].values[0]) + ' is your number?'
            set_repeat_qwe(phone=phone, key=True)
        twiml_xml = collect_2gathers_response(text=text, phone=phone, add_step=False)
        return str(twiml_xml)

    if out['mileage'][out['phone'] == phone].values[0] == 0:
        twiml_xml = collect_keybrd_response(text=phrases.keybrd_inp_ml, phone=phone, add_step=True)

    return str(twiml_xml)


@app.route('/question3_1', methods=['GET', 'POST'])  # работа с вебом
def question3_1():
    """
    обрабатывается вопрос:  могу я уточнить ваш регестрационный номер
    задаётся вопрос:
        Нашел --> правильно ли я понял ваш номер?
        неНашел --> могли бы вы повторить ответ
        Нет --> конец разговора.
    """

    global out

    url = request.form.get('RecordingUrl')
    data = io.BytesIO(urlopen(url).read())

    r = sr.Recognizer()
    with sr.AudioFile(data) as source:
        audio_ = r.record(source)

    client_speech = r.recognize_google(audio_)
    phone = request.form.get('To')
    write_user_answer(text='dictate: ' + client_speech, phone=phone)

    _, neg = get_pos_neg(client_speech=client_speech, phone=phone)
    found = find_plate_from_speech(client_speech=client_speech, phone=phone)

    if found:
        out[out.phone == phone]["again_key"] = True

        twiml_xml = collect_2gathers_response(phone=phone,
                                              text='Can I confirm the mileage as ' + str(
                                                  out['mileage'][out.phone == phone].values[0]) + '?')

    else:
        if out['repeat_qwe'][out['phone'] == phone].values[0]:
            text = phrases.cld_u_rpt
            set_repeat_qwe(phone=phone, key=False)

        else:
            text = 'could you repeat ' + ents.repeat_subj[out['stage'][out['phone'] == phone].values[0] - 1]
            set_repeat_qwe(phone=phone, key=True)
        twiml_xml = collect_2gathers_response(text=text, phone=phone, add_step=False, sufix='_1')

    return str(twiml_xml)


@app.route('/question4', methods=['POST'])  # работа с вебом
def question4():
    """
    обрабатывается вопрос: это ваш пробег. / правильно ли я понял ваш пробег?
    задаётся вопрос:
        Да --> вы живёте в этом городе?
        Нет --> могу я уточнить ваш пробег
    """

    phone, client_speech = get_stage_values(form=request.form, lack_ngt_text=False)
    pos, neg = get_pos_neg(client_speech=client_speech, phone=phone)

    pst_txt = 'Can I just confirm you live in ' + str(out['city'][out.phone == phone].values[0]) + '?',
    # при одобрении задаётся следующий вопрос    'Please dictate registration number of your car'
    if pos:
        if pd.isna(out['city'][out['phone'] == phone].values)[0]:
            twiml_xml = collect_2gathers_response(text='Please dictate the nearest city to you.', sufix='_1',
                                                  phone=phone)
        else:
            twiml_xml = collect_2gathers_response(text=pst_txt, phone=phone)

    # при отрицании прошу ввести номер телефона
    elif neg:
        twiml_xml = collect_keybrd_response(text=phrases.keybrd_inp_ml, phone=phone)
    # если не нашел никакой реакции переспроси
    else:
        if out['repeat_qwe'][out['phone'] == phone].values[0]:
            text = phrases.cld_u_rpt
            set_repeat_qwe(phone=phone, key=False)

        else:
            text = 'could you repeat ' + str(out['mileage'][out['phone'] == phone].values[0]) + ' is your mileage?'
            set_repeat_qwe(phone=phone, key=True)
        twiml_xml = collect_2gathers_response(text=text, phone=phone, add_step=False)

    return str(twiml_xml)


@app.route('/question4_1', methods=['POST'])  # работа с вебом
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

    pst_txt = 'Can I just confirm you live in ' + str(out['city'][out.phone == phone].values[0]) + '?',
    if pd.isna(out['city'][out['phone'] == phone].values)[0]:
        twiml_xml = collect_2gathers_response(text='Please dictate the nearest city to you.', sufix='_1', phone=phone)
    else:
        twiml_xml = collect_2gathers_response(text=pst_txt, phone=phone)

    return str(twiml_xml)


@app.route('/question5', methods=['POST'])  # работа с вебом
def question5():
    """
    обрабатывается вопрос: вы живёте в этом городе? / правильно ли я понял вы живёте в?
    задаётся вопрос:
        Да --> это ваш мобильный?
        Нет --> могу ли я уточнить ближайший город?
    """

    phone, client_speech, ngt_txt = get_stage_values(request.form)
    pos, neg = get_pos_neg(client_speech=client_speech, phone=phone)

    if pos:
        twiml_xml = collect_2gathers_response(text=phrases.fivth_stage, phone=phone)

    elif neg:
        twiml_xml = collect_2gathers_response(text=ngt_txt, sufix='_1', add_step=False,
                                              phone=phone, timeout='3')

    else:
        if out['repeat_qwe'][out['phone'] == phone].values[0]:
            text = phrases.cld_u_rpt
            set_repeat_qwe(phone=phone, key=False)

        else:
            text = 'could you repeat, do you live in ' + str(out['city'][out.phone == phone].values[0]) + '?'
            set_repeat_qwe(phone=phone, key=True)

        twiml_xml = collect_2gathers_response(text=text, phone=phone, add_step=False)

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
    url = request.form.get('RecordingUrl')
    data = io.BytesIO(urlopen(url).read())
    r = sr.Recognizer()

    with sr.AudioFile(data) as source:
        audio_ = r.record(source)

    try:
        client_speech = r.recognize_google(audio_)
    except sr.UnknownValueError:
        client_speech = 'repeat'
    except sr.RequestError:
        client_speech = 'repeat'
    phone = request.form.get('To')
    write_user_answer(text='dictate: ' + client_speech, phone=phone)

    found = find_city(client_speech=client_speech, phone=phone)

    if found:
        out[out.phone == phone]["again_key"] = True

        twiml_xml = collect_2gathers_response(add_step=False, phone=phone,
                                              text='Can I validate, you live in ' + str(
                                                  out['city'][out.phone == phone].values[0]) + '?')

    else:
        if out['repeat_qwe'][out['phone'] == phone].values[0]:
            text = phrases.cld_u_rpt
            set_repeat_qwe(phone=phone, key=False)

        else:
            text = 'could you repeat ' + ents.repeat_subj[out['stage'][out['phone'] == phone].values[0] - 1]
            set_repeat_qwe(phone=phone, key=True)

        twiml_xml = collect_2gathers_response(text=text, phone=phone, add_step=False, sufix='_1')

    return str(twiml_xml)


@app.route('/question6', methods=['POST'])
def question6():
    """
    обрабатывается вопрос: это ваш мобильный?
    задаётся вопрос:
        Да --> какая у вас сервисная история?
        Нет --> введите мобильный с помощью клавиатуры.
    """

    phone, client_speech = get_stage_values(form=request.form, lack_ngt_text=False)
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
        if out['repeat_qwe'][out['phone'] == phone].values[0]:
            text = phrases.cld_u_rpt
            set_repeat_qwe(phone=phone, key=False)

        else:
            text = 'could you repeat ' + ents.repeat_subj[out['stage'][out['phone'] == phone].values[0] - 1]
            set_repeat_qwe(phone=phone, key=True)
        twiml_xml = collect_2gathers_response(text=text, phone=phone, add_step=False)

    return str(twiml_xml)


@app.route('/question6_1', methods=['POST'])  # работа с вебом
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
        url = request.form.get('RecordingUrl')
        data = io.BytesIO(urlopen(url).read())
        r = sr.Recognizer()

        with sr.AudioFile(data) as source:
            audio_ = r.record(source)

        try:
            client_speech = r.recognize_google(audio_)
        except sr.UnknownValueError:
            client_speech = 'repeat'
        except sr.RequestError:
            client_speech = 'repeat'
        found = find_service_hist(client_speech, phone)
        write_user_answer(text='dictate: ' + client_speech, phone=phone)

    # если найдено одно из допустимых значений сервисной истории --> следующий вопрос
    if found:
        twiml_xml = collect_2gathers_response(text=phrases.seventh_stage, phone=phone)
        set_convrs_key(phone, 3)

    # иначе вопрос задаётся повторно
    else:
        if out['repeat_qwe'][out['phone'] == phone].values[0]:
            text = phrases.cld_u_rpt
            set_repeat_qwe(phone=phone, key=False)

        else:
            text = 'could you repeat ' + ents.repeat_subj[out['stage'][out['phone'] == phone].values[0] - 1]
            set_repeat_qwe(phone=phone, key=True)
        twiml_xml = collect_2gathers_response(text=text, phone=phone, add_step=False)

    return str(twiml_xml)


@app.route('/question8', methods=['POST'])
def question8():
    """
     обрабатывается вопрос: вы примите оффер?
     бот произносит прощание, разговор кончается
    """
    url = request.form.get('RecordingUrl')
    data = io.BytesIO(urlopen(url).read())
    r = sr.Recognizer()

    with sr.AudioFile(data) as source:
        audio_ = r.record(source)

    try:
        client_speech = r.recognize_google(audio_)
    except sr.UnknownValueError:
        client_speech = 'repeat'
    except sr.RequestError:
        client_speech = 'repeat'
    phone = request.form.get('To')
    pos, neg = get_pos_neg(client_speech=client_speech, phone=phone)
    write_user_answer(text='dictate: ' + client_speech, phone=phone)
    global out
    if pos:
        out['accept'][out.phone == phone] = True
        twiml_xml = collect_end_conversation(phrases.u_can_vld_it)

    elif neg:
        out['accept'][out.phone == phone] = False
        twiml_xml = collect_end_conversation(phrases.u_can_vld_it)

    else:
        twiml_xml = collect_end_conversation(phrases.u_can_vld_it)
    send_inf(phone=phone)

    return str(twiml_xml)  # twiML в конце перевожу в строку, потому что так было в примерчике


########################################################################################################


#                                                   call part
#######################################################################################################################
def del_person():
    """

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

    """
    # phone = config.phone_for_test
    unsend_inf = config.unwanted_for_webform_inf
    pers = out[out.phone == phone].drop(unsend_inf, axis=1)
    pers = pers.to_dict('records')[0]
    requests.post(config.recive_url, json=pers)


@app.route('/receiver', methods=['POST'])
def receiver():
    print(request.get_json())


def initiate_call(twiml, phone):
    """

    """
    client.calls.create(url=echo_url + urlencode({'Twiml': twiml}), to=phone, from_=config.twilio_number, record=True)


@app.route('/test_call', methods=['POST'])
def test_call():
    """

    """
    twiml = collect_2gathers_response(text=phrases.greeting, phone=config.phone_for_test, add_step=False)
    client.calls.create(url=echo_url + urlencode({'Twiml': twiml}), to=config.phone_for_test,
                        from_=config.twilio_number, record=True)


@app.route('/call_auto')
def call_auto():
    """

    """
    global exel_updated

    while True:
        if config.lower_limit <= datetime.utcnow().hour < config.upper_limit:
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
    совершает звонок, если имеются подходящие записи
    """
    global quiq_recall_numbers, exel_updated
    num_recalls = quiq_recalls()
    num_calls = np.max([config.num_calls_per_time - num_recalls, 2])
    today = datetime.utcnow().day

    fltrd_for_recall = out[out.cnvrs_key == 1][abs(out.call_day - today) >= config.recall_step] \
        [out.num_calls < config.last_call]

    fltrd_for_call = out[out.cnvrs_key == 0][abs(out.call_day - today) >= config.recall_step] \
        [out.num_calls < config.last_call]

    candidates_to_call = pd.concat([fltrd_for_call, fltrd_for_recall])

    if len(candidates_to_call):
        # print('ggg')

        if len(fltrd_for_call) >= int(round(num_calls / 100 * config.percent_of_new)):
            number_of_client = int(round(num_calls / 100 * (100 - config.percent_of_new)))
        else:
            number_of_client = int(round(num_calls / 100 * (100 - len(fltrd_for_call) * 100 / num_calls)))
        for i in range(1, -1, -1):
            phone_num_list = candidates_to_call[out.cnvrs_key == i]['phone'].values
            shuffle(phone_num_list)
            # print(phone_num_list, 'ppp', number_of_client, i)
            phone_num_list = phone_num_list[:number_of_client]
            number_of_client = len(phone_num_list)
            # print(phone_num_list, 'lll', number_of_client, i)

            for phone in phone_num_list:

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
            # print(config.num_calls_per_time, number_of_client, num_calls, number_of_client, 'eee')
            number_of_client = num_calls - number_of_client
            # print(config.num_calls_per_time, number_of_client, num_calls, number_of_client, 'kkk')

            # else:
            #     number_of_client = config.num_calls_per_time - num_calls
    else:
        if exel_updated:
            exel_updated = False
            parse_exel()


def quiq_recalls():
    global quiq_recall_numbers
    num_calls = 0

    for phone in quiq_recall_numbers:
        set_num_calls(phone)
        set_call_day(phone)
        if out['cnvrs_key'][out.phone == phone].values[0] == 1 or out['cnvrs_key'][out.phone == phone].values[0] == 0:
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
    pers = {'img_url': 'http://www.letchworthmini.co.uk/s/cc_images/cache_71011477.JPG', 'phone': '+9379992'}
    requests.post(config.main_url + 'extract_num', json=pers)


@app.route('/extract_num', methods=['POST'])
def extract_num():
    global out, exel_updated, exel_doc_counter

    try:
        xls = pd.ExcelFile(request.files['ex'])
        df = xls.parse(xls.sheet_names[0], converters={"phone": str})

        writer = pd.ExcelWriter(str(exel_doc_counter) + '.xlsx')
        df.to_excel(writer, 'Sheet1')
        writer.save()
        exel_doc_counter += 1

        exel_updated = True
        return "SUCCESS"

    except:
        photo_url = request.get_json()['img_url']

        with graph.as_default():
            c = Main.main(photo_url)
        req = request.get_json()

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
        dt = pd.DataFrame(data=values, columns=config.flds)
        out = pd.concat([out, dt], ignore_index=True)

        return "SUCCESS"


def parse_exel():
    global out, exel_doc_counter
    for i in range(exel_doc_counter):
        xls = pd.ExcelFile(str(i) + '.xlsx')
        df = xls.parse(xls.sheet_names[0], converters={"Phone": str, "Mileage": int, 'Location': str})

        df = df.dropna(subset=['Phone'])

        df = df[['Phone', 'Featured Image', 'Mileage', 'Location']].rename(columns={'Phone': 'phone',
                                                                                    'Mileage': 'mileage',
                                                                                    'Location': 'city',
                                                                                    'Featured Image': 'img_url'})
        df['phone'] = '+' + df['phone']

        for city_name in df['city']:
            indx = city_name.find('United Kingdom')
            if indx != -1:
                df['city'][df.city == city_name] = city_name[:indx]

        req = df.to_dict()
        req['reg_num'] = {}

        for j in range(len(req['phone'])):
            # try:
            req['reg_num'][j] = find_plate_by_network(req['img_url'][j])
            # except:
            #     req['reg_num'][j] = 'A A 0 0 A A A'

        for j in range(len(config.adding_filds)):
            if j < 3:
                req = add_field_to_dict(0, j, req)

            elif j < 8:
                req = add_field_to_dict(False, j, req)

            elif j == 8:
                req = add_field_to_dict(1, j, req)

            else:
                req = add_field_to_dict(None, j, req)

        df = pd.DataFrame.from_dict(req)
        df['mileage'] = df['mileage'].fillna(0)
        df['mileage'] = df['mileage'].astype('int64', errors='ignore')

        #####################################################################################################
        two_cars_clients = pd.read_csv('two_cars_client.csv', converters={"phone": str, 'phone for offer': str,
                                                                          'mileage': int})

        df = pd.concat([two_cars_clients, df], ignore_index=True)

        new_clients = df.phone.values
        filtred_client = out[:0]
        for client in new_clients:
            if client in out.phone.values:
                filtred_client = pd.concat([filtred_client, df[df.phone == client]], ignore_index=True)
                df = df.drop(df[df.phone == client].index)

        new_clients = Counter(df.phone.values).most_common()
        for client in new_clients:
            if client[1] < 2:
                break
            else:
                filtred_client = pd.concat([filtred_client, df[df.phone == client[0]][1:]], ignore_index=True)
                df = df.drop(df[df.phone == client[0]].index[1:])
        filtred_client.to_csv('two_cars_client.csv', index=False)

        print(df, 'LLLLLLLLLLLLLLLL', '\n\n\n\n')
        ######################################################################################################
        print(out, '\n\n')

        out = pd.concat([out, df], ignore_index=True)
        print(out)
    exel_doc_counter = 0
    # requests.post(config.add_pers_url, json=req)


def add_field_to_dict(value, pos_field, req):
    req[config.adding_filds[pos_field]] = {}
    for j in range(len(req['phone'])):
        req[config.adding_filds[pos_field]][j] = value
    return req


@app.route('/i')
def index():
    return '<form action="' + config.main_url + """extract_num" method="post" enctype="multipart/form-data">
                <input type="file" name="ex">
                <input type="submit" name="submit">
                </form>"""


@app.route('/ir')
def ir():
    print(out)
    return ''


if __name__ == '__main__':
    app.run(host=config.host, port=config.port, threaded=True)
