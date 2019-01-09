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

model = load_model('modelL2.h5')
model.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.005), loss='categorical_crossentropy',
              metrics=['accuracy'])
graph = tf.get_default_graph()

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Your Account Sid and Auth Token from twilio.com/console
client = Client(config.account_sid, config.auth_token)

# начало url для twiML
echo_url = config.echo_url
start_url = config.start_url

ngrok_url = config.ngrok_url
app = Flask(__name__)  # работа с вебом

out = pd.DataFrame(
    {"reg_num": [''], "mileage": [10000], "city": [''], "phone": [''], "phone_for_offer": [''], 'number_of_calls': [9],
     "serv_hist": [''], 'again_key': [True], 'stage': [1], 'pst': [True], 'ngt': [True], 'cnvrs_key': [3],
     'first_ques': [False], 'call_day': [0], 'accept': [None], 'img_url': [''], 'repeat_qwe': [True]})

tokens = ['7eb82f1ed5e6ceeca6b26f8316b31717fde0bb25', 'f9e106e2c0a0c6a2493181fd724cdb7b89600af9',
          '9118e9c1b2a3f65c39b8d90453db99165fb201f0', '0e8c566ad072d543aae409d576012ee4e98a766e']
quiq_recall_phones = []


#                                            find special values
#######################################################################################################################

def find_service_hist(client_speech, phone):
    """

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
    translation = {ord(x): None for x in string.punctuation}
    client_speech = client_speech.translate(translation).lower()

    for i in ents.cities:
        if i.lower() in client_speech:
            out['city'][out['phone'] == phone] = i
            return True

    client_speech = client_speech.split(' ')
    for i in ents.part_of_cities:
        if i.lower() in client_speech:
            out['city'][out['phone'] == phone] = i
            return True

    return False


def find_plate(phone):
    try:
        resp = urlopen(str(out['img_url'][out['phone'] == phone].values[0]))
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        imgOriginalScene = cv2.imdecode(image, cv2.IMREAD_COLOR)
        cv2.imwrite('tst.png', imgOriginalScene)

        with open('tst.png', 'rb') as fp:
            response = requests.post(
                'https://platerecognizer.com/plate-reader/',
                files=dict(upload=fp),
                headers={'Authorization': 'Token ' + tokens[np.random.randint(4)]})

        out['reg_num'][out['phone'] == phone] = response.json()['results'][0]['plate']
    except:
        out['reg_num'][out['phone'] == phone] = 'AA11AAA'


#######################################################################################################################


#                                           collecting and standart operation part
#######################################################################################################################
def collect_speech_gather(text, hints, phone_number, rsp, sufix=''):
    """

    """
    stg = str(out[out.phone == phone_number]['stage'].values[0])
    rsp.say(text)
    rsp.record(finish_on_key='*', play_beep=False, timeout=config.timeout, action=ngrok_url + stg + sufix, max_length=6)

    return rsp


def collect_redirect_speech(phone):
    """

    """
    global out
    response = VoiceResponse()

    response.say('Hi, I am phoning you again to talk about your car. Last time we stoped on ' +
                 subjects[out['stage'][out.phone == phone].values[0] - 2])

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


def collect_2gathers_response(text, hints, phone, sufix='', add_step=True, timeout='auto'):
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
    twiml_response = collect_speech_gather(text, hints, phone, twiml_response, sufix)
    twiml_response = collect_speech_gather(text, hints, phone, twiml_response, sufix)
    twiml_response.say(phrases.ddnt_rcv_inp)
    return twiml_response.to_xml()


def collect_keybrd_response(text, phone):
    """

    """
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
    for i in ents.ents[case]:
        if out[case][out.phone == phone].values[0]:
            break
        if i in text:
            out[case][out.phone == phone] = True


def get_pos_neg(client_speech, phone):
    check_for_pos_neg(client_speech, phone=phone)
    return out['pst'][out.phone == phone].values[0], out['ngt'][out.phone == phone].values[0]


def choose_rigth_answer(positive_text, negative_text, client_speech, phone, positive_hint='',
                        negative_hint='', end_call=False, timeout='auto', repeat_hint=''):
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
        twiml_xml = collect_2gathers_response(text=positive_text, hints=positive_hint, phone=phone)

    elif neg and end_call:
        set_convrs_key(phone=phone, key=2)
        twiml_xml = collect_end_conversation(negative_text)

    elif neg:
        twiml_xml = collect_2gathers_response(text=negative_text, hints=negative_hint, sufix='_1', add_step=False,
                                              phone=phone, timeout=timeout)

    else:
        if out['repeat_qwe'][out['phone'] == phone].values[0]:
            text = phrases.cld_u_rpt
            set_repeat_qwe(phone=phone, key=False)

        else:
            text = 'could you repeat ' + ents.repeat_subj[out['stage'][out['phone'] == phone].values[0] - 1]
            set_repeat_qwe(phone=phone, key=True)

        twiml_xml = collect_2gathers_response(text=text, hints=repeat_hint, phone=phone, add_step=False)

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


def set_number_of_calls(phone):
    global out
    out['number_of_calls'][out.phone == phone] += 1


def set_repeat_qwe(phone, key):
    global out
    out['repeat_qwe'][out['phone'] == phone] = key


#                                                   question part
######################################################################################################################
@app.route('/question1', methods=['GET', 'POST'])  # работа с вебом
def question1():
    """
    обрабатывается вопрос: вы всё ещё продаёте машину?
    задаётся вопрос: интересно ли клиенту предложение.
    """
    phone, client_speech = get_stage_values(request.form, lack_ngt_text=False)
    set_convrs_key(phone, 1)

    # twilio выбор ответа
    return choose_rigth_answer(end_call=True, positive_text=phrases.first_stage, positive_hint=phrases.pst_hint,
                               phone=phone, client_speech=client_speech, negative_text=phrases.sorry_4_bothr,
                               repeat_hint=phrases.pst_hint)


@app.route('/question2', methods=['GET', 'POST'])  # работа с вебом
def question2():
    """
    обрабатывается вопрос: интересно ли клиенту предложение.
    задаётся вопрос:
        Да --> это ваш регестрационный номер.
        Нет --> кладётся трубка.
    """

    phone, client_speech = get_stage_values(request.form, lack_ngt_text=False)

    # twilio выбор ответа
    return choose_rigth_answer(client_speech=client_speech, negative_text=phrases.sorry_4_bothr,
                               positive_text='Can I confirm your vehicle registration number is ' + str(
                                   out[out.phone == phone]['reg_num'].values[0]) + '?',
                               positive_hint=phrases.pst_hint, phone=phone, end_call=True, repeat_hint=phrases.pst_hint)


@app.route('/question3', methods=['GET', 'POST'])  # работа с вебом
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

    if pos:
        twiml_xml = collect_2gathers_response(text='Can I confirm the mileage as ' + str(
            out['mileage'][out.phone == phone].values[0]) + '?', hints='', phone=phone)

    elif neg:
        find_plate(phone=phone)
        twiml_xml = collect_2gathers_response(hints='', phone=phone, timeout='5',
                                              text='Can I confirm the mileage as ' + str(
                                                  out['mileage'][out.phone == phone].values[0]) + '?')

    else:
        if out['repeat_qwe'][out['phone'] == phone].values[0]:
            text = phrases.cld_u_rpt
            set_repeat_qwe(phone=phone, key=False)

        else:
            text = 'could you repeat ' + out['reg_num'][out['phone'] == phone] + ' is your number?'
            set_repeat_qwe(phone=phone, key=True)
        twiml_xml = collect_2gathers_response(text=text, hints='', phone=phone, add_step=False)

    return str(twiml_xml)


@app.route('/question4', methods=['GET', 'POST'])  # работа с вебом
def question4():
    """
    обрабатывается вопрос: это ваш пробег. / правильно ли я понял ваш пробег?
    задаётся вопрос:
        Да --> вы живёте в этом городе?
        Нет --> могу я уточнить ваш пробег
    """

    phone, client_speech = get_stage_values(form=request.form, lack_ngt_text=False)
    pos, neg = get_pos_neg(client_speech=client_speech, phone=phone)

    # при одобрении задаётся следующий вопрос
    if pos:
        out['mileage'][out.phone == phone] = phone
        twiml_xml = collect_2gathers_response(text='Can I just confirm you live in ' +
                                                   str(out['city'][out.phone == phone].values[0]) + '?',
                                              phone=phone, hints=phrases.serv_hist)

    # при отрицании прошу ввести номер телефона
    elif neg:
        twiml_xml = collect_keybrd_response(text=phrases.keybrd_inp_ml, phone=phone)
    # если не нашел никакой реакции переспроси
    else:
        if out['repeat_qwe'][out['phone'] == phone].values[0]:
            text = phrases.cld_u_rpt
            set_repeat_qwe(phone=phone, key=False)

        else:
            text = 'could you repeat ' + out['mileage'][out['phone'] == phone] + ' is your mileage?'
            set_repeat_qwe(phone=phone, key=True)
        twiml_xml = collect_2gathers_response(text=text, hints='', phone=phone, add_step=False)

    return str(twiml_xml)


@app.route('/question4_1', methods=['GET', 'POST'])  # работа с вебом
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
    twiml_xml = collect_2gathers_response(text='Can I just confirm you live in ' +
                                               str(out['city'][out.phone == phone].values[0]) + '?',
                                          phone=phone, hints=phrases.serv_hist)

    return str(twiml_xml)


@app.route('/question5', methods=['GET', 'POST'])  # работа с вебом
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
        twiml_xml = collect_2gathers_response(text=phrases.fivth_stage, hints='', phone=phone)

    elif neg:
        twiml_xml = collect_2gathers_response(text=ngt_txt, hints='', sufix='_1', add_step=False,
                                              phone=phone, timeout='3')

    else:
        if out['repeat_qwe'][out['phone'] == phone].values[0]:
            text = phrases.cld_u_rpt
            set_repeat_qwe(phone=phone, key=False)

        else:
            text = 'could you repeat, do you live in ' + str(out['city'][out.phone == phone].values[0]) + '?'
            set_repeat_qwe(phone=phone, key=True)

        twiml_xml = collect_2gathers_response(text=text, hints='', phone=phone, add_step=False)

    return str(twiml_xml)


@app.route('/question5_1', methods=['GET', 'POST'])
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

        twiml_xml = collect_2gathers_response(add_step=False, phone=phone, hints=phrases.pst_hint,
                                              text='Can I validate, you live in ' + str(
                                                  out['city'][out.phone == phone].values[0]) + '?')

    else:
        if out['repeat_qwe'][out['phone'] == phone].values[0]:
            text = phrases.cld_u_rpt
            set_repeat_qwe(phone=phone, key=False)

        else:
            text = 'could you repeat ' + ents.repeat_subj[out['stage'][out['phone'] == phone].values[0] - 1]
            set_repeat_qwe(phone=phone, key=True)

        twiml_xml = collect_2gathers_response(text=text, hints='', phone=phone, add_step=False, sufix='_1')

    return str(twiml_xml)


@app.route('/question6', methods=['GET', 'POST'])
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
        twiml_xml = collect_2gathers_response(text=phrases.sixth_stage, phone=phone, hints=phrases.serv_hist)

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
        twiml_xml = collect_2gathers_response(text=text, hints='', phone=phone, add_step=False)

    return str(twiml_xml)


@app.route('/question6_1', methods=['GET', 'POST'])  # работа с вебом
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
        twiml_xml = collect_2gathers_response(text=phrases.sixth_stage, phone=phone, hints=phrases.serv_hist)

    else:
        twiml_xml = collect_keybrd_response(text=phrases.nt_vld_input, phone=phone)

    return str(twiml_xml)


@app.route('/question7')
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
        twiml_xml = collect_2gathers_response(text=phrases.seventh_stage, hints=phrases.pst_hint, phone=phone)
        set_convrs_key(phone, 3)
        send_inf(phone=phone)

    # иначе вопрос задаётся повторно
    else:
        if out['repeat_qwe'][out['phone'] == phone].values[0]:
            text = phrases.cld_u_rpt
            set_repeat_qwe(phone=phone, key=False)

        else:
            text = 'could you repeat ' + ents.repeat_subj[out['stage'][out['phone'] == phone].values[0] - 1]
            set_repeat_qwe(phone=phone, key=True)
        twiml_xml = collect_2gathers_response(text=text, hints='', phone=phone, add_step=False)

    return str(twiml_xml)


@app.route('/question8', methods=['GET', 'POST'])
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
    too_many_calls = out[out['number_of_calls'] >= config.finaly_number_of_calls].drop(unsend_inf, axis=1)
    out = out.drop(out[out['number_of_calls'] >= config.finaly_number_of_calls].index)
    hist_data = pd.concat([hist_data, not_lack_call, too_many_calls], ignore_index=True)
    hist_data.to_csv('history.csv', index=False)


@app.route('/send', methods=['GET', 'POST'])
def send_inf(phone):
    """

    """
    # phone = config.phone_for_test
    unsend_inf = config.unwanted_for_webform_inf
    pers = out[out.phone == phone].drop(unsend_inf, axis=1)
    pers = pers.to_dict('records')[0]
    requests.post(config.recive_url, json=pers)


@app.route('/receiver', methods=['GET', 'POST'])
def receiver():
    print(request.get_json())


def initiate_call(twiml, phone):
    """

    """
    client.calls.create(url=echo_url + urlencode({'Twiml': twiml}), to=phone, from_=config.twilio_number, record=True)


@app.route('/test_call', methods=['GET', 'POST'])
def test_call():
    """

    """
    twiml = collect_2gathers_response(text=phrases.greeting,
                                      hints=phrases.pst_hint, phone=config.phone_for_test, add_step=False)
    client.calls.create(url=echo_url + urlencode({'Twiml': twiml}), to=config.phone_for_test,
                        from_=config.twilio_number,
                        record=True)


@app.route('/call_auto')
def call_auto():
    """

    """
    while True:
        if config.lower_limit <= datetime.utcnow().hour < config.upper_limit:
            print('ddd')
            make_calls()
            time.sleep(config.sleep_min * 60)
        else:
            del_person()
            time.sleep(12 * 60 * 60)


@app.route('/make_calls', methods=['GET', 'POST'])
def make_calls():
    """
    совершает звонок, если имеются подходящие записи
    """

    global quiq_recall_phones
    number_of_calls = np.max([config.number_of_calls_per_time - quiq_recalls(), 2])

    filtered_for_recall = out[out.cnvrs_key == 1][abs(out.call_day - datetime.utcnow().day) >= config.recall_step] \
        [out.number_of_calls < config.finaly_number_of_calls]
    filtered_for_call = out[out.cnvrs_key == 0][abs(out.call_day - datetime.utcnow().day) >= config.recall_step] \
        [out.number_of_calls < config.finaly_number_of_calls]

    candidates_to_call = pd.concat([filtered_for_call, filtered_for_recall])
    if len(candidates_to_call):
        number_of_client = int(round(number_of_calls / 100 * (100 - config.percent_of_new)))

        for i in range(1, -1, -1):

            phone_num_list = candidates_to_call[out.cnvrs_key == i]['phone'].values
            shuffle(phone_num_list)
            if len(phone_num_list):
                phone_num_list = phone_num_list[:np.min([number_of_client, len(phone_num_list)])]
                number_of_client = len(phone_num_list)
                for phone in phone_num_list:

                    quiq_recall_phones.append(phone)
                    set_number_of_calls(phone)
                    set_call_day(phone)
                    if i == 1:
                        set_first_qwe(phone=phone, key=True)
                        twiml_xml = collect_redirect_speech(phone=phone)
                    else:
                        set_first_qwe(phone=phone, key=False)

                        twiml_xml = collect_2gathers_response(text=phrases.greeting,
                                                              hints=phrases.pst_hint, phone=phone, add_step=False)
                    initiate_call(twiml=twiml_xml, phone=phone)
                number_of_client = config.number_of_calls_per_time - number_of_client
            else:
                number_of_client = config.number_of_calls_per_time


def quiq_recalls():
    global quiq_recall_phones
    number_of_calls = 0

    for phone in quiq_recall_phones:
        set_number_of_calls(phone)
        set_call_day(phone)
        if out['cnvrs_key'][out.phone == phone].values[0] == 1 or out['cnvrs_key'][out.phone == phone].values[0] == 0:
            if out['cnvrs_key'][out.phone == phone].values[0] == 1:
                set_first_qwe(phone=phone, key=True)
                twiml_xml = collect_redirect_speech(phone=phone)
            elif out['cnvrs_key'][out.phone == phone].values[0] == 0:
                set_first_qwe(phone=phone, key=False)
                twiml_xml = collect_2gathers_response(text=phrases.greeting,
                                                      hints=phrases.pst_hint, phone=phone, add_step=False)
            initiate_call(twiml=twiml_xml, phone=phone)
            number_of_calls += 1
    quiq_recall_phones.clear()
    return number_of_calls


######################################################################################################################


@app.route('/add_person', methods=['GET', 'POST'])
def add_person():
    """

    """
    global out
    if request.method == "POST":
        json = request.get_json()
        dt = pd.DataFrame.from_dict(json)
        out = pd.concat([out, dt], ignore_index=True)

    return " "


@app.route('/snd', methods=['POST', 'GET'])
def snd():
    pers = {'img_url': 'http://www.letchworthmini.co.uk/s/cc_images/cache_71011477.JPG', 'phone': '+9379992'}
    requests.post(config.main_url + 'extract_num', json=pers)


@app.route('/extract_num')
def extract_num():
    global out

    try:
        xls = pd.ExcelFile(request.files['pic'])
        df = xls.parse(xls.sheet_names[0], converters={"phone": str})

        req = df.to_dict()
        req['reg_num'] = {}

        with graph.as_default():
            for j in range(len(req['phone'])):
                req['reg_num'][j] = Main.main(req['img_url'][j])

        for i in range(len(config.adding_filds)):
            if i < 3:
                req = add_field_to_dict(0, i, req)

            elif i < 8:
                req = add_field_to_dict(False, i, req)

            elif i == 8:
                req = add_field_to_dict(1, i, req)

            else:
                req = add_field_to_dict(None, i, req)

        requests.post(config.add_pers_url, json=req)
        writer = pd.ExcelWriter('tmp.xlsx')
        df.to_excel(writer, 'Sheet1')
        writer.save()
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


def add_field_to_dict(value, pos_field, req):
    req[config.adding_filds[pos_field]] = {}
    for j in range(len(req['phone'])):
        req[config.adding_filds[pos_field]][j] = value
    return req


@app.route('/i')
def index():
    return """<form action="/extract_num" method="post" enctype="multipart/form-data">
                <input type="file" name="pic">
                <input type="submit" name="submit">
                </form>"""


@app.route('/ir')
def ir():
    print(out)


if __name__ == '__main__':
    app.run(host=config.host, port=config.port, threaded=True)
