import string
from flask import Flask, request
import spacy
from ents import ents, subjects
import pandas as pd
from nltk.stem import WordNetLemmatizer
import re
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
from word2number import w2n
import tensorflow as tf
import Main
from keras.models import load_model
from keras.optimizers import RMSprop

model = load_model('char-reg.h5')
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
nlp = spacy.load('en_core_web_md')
app = Flask(__name__)  # работа с вебом

out = pd.DataFrame(
    {"reg_num": ['AS 123 SD', None, None], "mileage": ['1500', None, None], "city": ['London', None, None],
     "phone": ["+380938482501", "+55555555555", "+3333333333333"], "phone_for_offer": [None, None, None],
     "serv_hist": [None, None, None], 'again_key': [False, False, False], 'stage': [2, 1, 1],
     'pst': [False, False, False], 'ngt': [False, False, False], 'cnvrs_key': [1, 2, 2], 'number_of_calls': [2, 1, 3],
     'first_ques': [True, True, True], 'call_hour': [0, 0, 0]})


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
            out['service history'][out.phone == phone] = ents.serv_hist_kind[word]
            found = True
            break
    return found


def find_city(client_speech, phone):
    doc = nlp(client_speech)
    for ent in doc.ents:
        # print(ent.label_, ent.text)
        if ent.label_ == 'GPE':
            out['city'][out['phone'] == phone] = ent.text

            out.to_csv('out.csv')
            return True
    return False


def find_mileage(client_speech, phone):
    clear = client_speech.replace(".", "")  # lower case

    clear_lower = clear.lower()  # lemmatisation

    lemmer = WordNetLemmatizer()
    tokens = clear_lower.split()
    lemm = [lemmer.lemmatize(token) for token in tokens]
    lemm_str = ' '.join(lemm)

    doc = nlp(w2n.word_to_num(lemm_str))

    # Перезаписывание ответа клиента в файл
    for ent in doc.ents:
        # print(ent.label_, ent.text)
        if ent.label_ == 'QUANTITY' or ent.label_ == 'CARDINAL':
            out['mileage'][out.phone == phone] = ent.text
            return True
    return False


def find_plate(client_speech, phone):
    found = False
    translation = {ord(x): None for x in string.punctuation}

    client_speech = client_speech.translate(translation)

    clue_str = client_speech.replace(" ", "")
    plate_format_1 = re.findall(r'[a-z][0-9]{3}[a-z]{3}', clue_str)
    plate_format_2 = re.findall(r'[a-z][0-9]{2}[a-z]{3}', clue_str)

    if plate_format_1:
        out['registration number'][out['phone'] == phone] = plate_format_1[0].upper()
        found = True
    elif plate_format_2:
        plate_format = re.findall(r'[a-z][0-9]{2}[a-z]{3}', clue_str)
        tmp_str = plate_format[0][0] + ' ' + plate_format[0][1:3]
        pos_plate = client_speech.index(tmp_str)

        if pos_plate == 0 or (
                pos_plate > 2 and client_speech[pos_plate - 3] != ' ' and client_speech[pos_plate - 1] == ' '):
            plate_format_2 = re.findall(r'[a-z][0-9]{2}[a-z]{3}', clue_str)
            out['registration number'][out['phone'] == phone] = plate_format_2[0].upper()
            found = True

        elif client_speech[pos_plate - 1] != ' ' or pos_plate - 3 == -1 or client_speech[pos_plate - 3] == ' ':
            plate_format_2 = re.findall(r'[a-z]{2}[0-9]{2}[a-z]{3}', clue_str)
            out['registration number'][out['phone'] == phone] = plate_format_2[0].upper()
            found = True

    return found


#######################################################################################################################


#                                           collecting and standart operation part
#######################################################################################################################
def collect_speech_gather(text, hints, phone_number, sufix='', timeout='auto'):
    """

    """
    if timeout != 'auto':
        timeout = int(timeout)
    stg = str(out[out.phone == phone_number]['stage'][0])
    gather = Gather(speechTimeout=timeout, hints=hints, action=ngrok_url + stg + sufix, input='speech')
    gather.say(text)
    return gather


def collect_redirect_speech(phone):
    """

    """
    response = VoiceResponse()

    response.say('Hi, I am phoning you again to talk about your car. Last time we stoped on ' +
                 subjects[out['stage'][out.phone == phone][0] - 1])

    response.redirect(ngrok_url + str(out['stage'][out.phone == phone][0]))
    return response.to_xml()


def collect_dgt_gather(text, phone, sufix=''):
    """

    """
    stg = str(out[out.phone == phone]['stage'][0])
    gather = Gather(input='dtmf', numDigits="10", timeout=7, action=ngrok_url + stg + sufix)
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
        out['stage'][out['phone'] == phone] = 1 + out['stage'][out['phone'] == phone][0]

    twiml_response = VoiceResponse()
    twiml_response.append(collect_speech_gather(text, hints, phone, sufix, timeout))
    twiml_response.append(collect_speech_gather(phrases.cld_u_rpt, hints, phone, sufix, timeout))
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
        if out[case][out.phone == phone][0]:
            break
        if i in text:
            out[case][out.phone == phone] = True


def get_pos_neg(client_speech, phone):
    check_for_pos_neg(client_speech, phone=phone)
    return out['pst'][out.phone == phone][0], out['ngt'][out.phone == phone][0]


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
        twiml_xml = collect_2gathers_response(text=phrases.cld_u_rpt, hints=repeat_hint, phone=phone, add_step=False)
    return str(twiml_xml)


def undrstnd_newcall_recall(phone, form):
    """

    """
    global out
    if out['first_ques'][out.phone == phone][0]:
        out['first_ques'][out.phone == phone] = False
        return 'yes lll'
    else:
        client_speech = form.get('SpeechResult')
        write_user_answer(text=client_speech, phone=phone)
        return client_speech


def undrstnd_newques_reask(phone):
    """

    """
    global out
    # проверка не задаём ли мы данный вопрос повторно (если да то вопросительная фраза другая)
    if out['again_key'][out.phone == phone][0]:
        out['again_key'][out.phone == phone] = False
        return 'could you talk ' + subjects[out['stage'][out.phone == phone][0] - 2] + ' again?'
    else:
        return 'Please dictate ' + subjects[out['stage'][out.phone == phone][0] - 2]


def write_user_answer(text, phone):
    with open(phone, "a") as f:
        f.write("\n" + text + "\n")


def get_stage_values(form, lack_ngt_text=True):
    phone = form.get('To')
    if lack_ngt_text:
        return phone, undrstnd_newcall_recall(phone=phone, form=form), undrstnd_newques_reask(phone=phone)
    else:
        return phone, undrstnd_newcall_recall(phone=phone, form=form)


###############################################################################################################


def set_call_hour(phone):
    """

    """
    out['call_hour'][out.phone == phone] = datetime.utcnow().hour
    if datetime.utcnow().minute > 25:
        out['call_hour'][out.phone == phone] += 1


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
    out['number_of_calls'][out.phone == phone] += 1


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
                                   out[out.phone == phone]['registration number'][0]) + '?',
                               positive_hint=phrases.pst_hint, phone=phone, end_call=True, repeat_hint=phrases.pst_hint)


@app.route('/question3', methods=['GET', 'POST'])  # работа с вебом
def question3():
    """
    обрабатывается вопрос: это ваш регестрационный номер. / правильно ли я понял ваш номер?
    задаётся вопрос:
        Да --> это ваш пробег
        Нет --> могу я уточнить ваш регестрационный номер
    """

    phone, client_speech, ngt_txt = get_stage_values(request.form)

    return choose_rigth_answer(positive_hint=phrases.pst_hint, negative_hint=phrases.reg_num,
                               positive_text='Can I confirm the mileage as ' + str(
                                   out['mileage'][out.phone == phone][0]) + '?',
                               repeat_hint=phrases.pst_hint + ', ' + phrases.reg_num,
                               client_speech=client_speech, negative_text=ngt_txt, phone=phone, timeout='5')


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

    client_speech = request.form.get('SpeechResult')
    phone = request.form.get('To')
    write_user_answer(text=client_speech, phone=phone)

    _, neg = get_pos_neg(client_speech=client_speech, phone=phone)
    found = find_plate(client_speech=client_speech, phone=phone)

    if found:
        out[out.phone == phone]["again_key"] = True

        twiml_xml = collect_2gathers_response(add_step=False, phone=phone, hints=phrases.pst_hint,
                                              text='Can I validate, your vehicle registration number is ' +
                                                   str(out['registration number'][out.phone == phone][0]) + '?')

    elif neg:
        set_convrs_key(phone=phone, key=2)
        twiml_xml = collect_end_conversation(phrases.sorry_4_bothr)

    else:
        twiml_xml = collect_2gathers_response(text=phrases.cld_u_rpt, hints=phrases.reg_num, phone=phone,
                                              add_step=False, sufix='_1', timeout='5')

    return str(twiml_xml)


@app.route('/question4', methods=['GET', 'POST'])  # работа с вебом
def question4():
    """
    обрабатывается вопрос: это ваш пробег. / правильно ли я понял ваш пробег?
    задаётся вопрос:
        Да --> вы живёте в этом городе?
        Нет --> могу я уточнить ваш пробег
    """

    phone, client_speech, ngt_txt = get_stage_values(request.form)

    return choose_rigth_answer(positive_hint=phrases.pst_hint, client_speech=client_speech,
                               phone=phone, negative_hint=phrases.mileage, negative_text=ngt_txt,
                               positive_text='Can I just confirm you live in ' +
                                             str(out['city'][out.phone == phone][0]) + '?',
                               repeat_hint=phrases.pst_hint + ', ' + phrases.mileage)


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

    client_speech = request.form.get('SpeechResult')
    phone = request.form.get('To')
    write_user_answer(text=client_speech, phone=phone)

    _, neg = get_pos_neg(client_speech=client_speech, phone=phone)
    found = find_mileage(client_speech=client_speech, phone=phone)

    if found:
        out[out.phone == phone]["again_key"] = True

        twiml_xml = collect_2gathers_response(add_step=False, phone=phone, hints=phrases.pst_hint,
                                              text='Can I validate, your mileage is ' +
                                                   str(out['mileage'][out.phone == phone][0]) + '?')

    elif neg:
        set_convrs_key(phone=phone, key=2)
        twiml_xml = collect_end_conversation(phrases.sorry_4_bothr)

    else:
        twiml_xml = collect_2gathers_response(text=phrases.cld_u_rpt, hints=phrases.mileage, phone=phone,
                                              add_step=False, sufix='_1')

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

    return choose_rigth_answer(positive_text=phrases.fivth_stage, positive_hint=phrases.pst_hint,
                               client_speech=client_speech, negative_text=ngt_txt, phone=phone,
                               negative_hint=phrases.city, repeat_hint=phrases.pst_hint + ', ' + phrases.city)


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
    client_speech = request.form.get('SpeechResult')
    phone = request.form.get('To')
    write_user_answer(text=client_speech, phone=phone)

    _, neg = get_pos_neg(client_speech=client_speech, phone=phone)
    found = find_city(client_speech=client_speech, phone=phone)

    if found:
        out[out.phone == phone]["again_key"] = True

        twiml_xml = collect_2gathers_response(add_step=False, phone=phone, hints=phrases.pst_hint,
                                              text='Can I validate, you live in ' + str(
                                                  out['city'][out.phone == phone][0]) + '?')

    elif neg:
        set_convrs_key(phone=phone, key=2)
        twiml_xml = collect_end_conversation(phrases.sorry_4_bothr)

    else:
        twiml_xml = collect_2gathers_response(text=phrases.cld_u_rpt, hints=phrases.city, phone=phone, add_step=False,
                                              sufix='_1')

    return str(twiml_xml)


@app.route('/question6', methods=['GET', 'POST'])  # работа с вебом
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
        out['phone for offer'][out.phone == phone] = phone
        twiml_xml = collect_2gathers_response(text=phrases.sixth_stage, phone=phone, hints=phrases.serv_hist)

    # при отрицании прошу ввести номер телефона
    elif neg:
        twiml_xml = collect_keybrd_response(text=phrases.keybrd_inp, phone=phone)
    # если не нашел никакой реакции переспроси
    else:
        twiml_xml = collect_2gathers_response(text=phrases.cld_u_rpt, hints=phrases.pst_hint, phone=phone,
                                              add_step=False)

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
    write_user_answer(text=client_inp, phone=phone)

    if len(client_inp) == 10:
        out['phone for offer'][out.phone == phone] = "+" + client_inp
        twiml_xml = collect_2gathers_response(text=phrases.sixth_stage, phone=phone, hints=phrases.serv_hist)

    else:
        twiml_xml = collect_keybrd_response(text=phrases.nt_vld_input, phone=phone)

    return str(twiml_xml)


@app.route('/question7', methods=['GET', 'POST'])
def question7():
    """
    обрабатывается вопрос: какая у вас сервисная история?
    задаётся вопрос:
        удовлетворительный ответ --> вы примите оффер?
    """

    global out

    phone = request.form.get('To')
    if out['cnvrs_key'][out.phone == phone][0] == 1 and out['first_ques'][out.phone == phone][0]:
        found = True
        out['first_ques'][out.phone == phone] = False
    else:
        client_speech = request.form.get('SpeechResult')
        found = find_service_hist(client_speech, phone)
        write_user_answer(text=client_speech, phone=phone)

    # если найдено одно из допустимых значений сервисной истории --> следующий вопрос
    if found:
        twiml_xml = collect_2gathers_response(text=phrases.seventh_stage, hints=phrases.pst_hint, phone=phone)
        out['cnvrs_key'][out.phone == phone] = 3
        send_inf(phone=phone)

    # иначе вопрос задаётся повторно
    else:
        twiml_xml = collect_2gathers_response(text=phrases.cld_u_rpt, hints=phrases.serv_hist, phone=phone,
                                              add_step=False)

    return str(twiml_xml)


@app.route('/question8', methods=['GET', 'POST'])
def question8():
    """
     обрабатывается вопрос: вы примите оффер?
     бот произносит прощание, разговор кончается
    """
    pos, _ = get_pos_neg(client_speech=request.form.get('SpeechResult'), phone=request.form.get('To'))

    if pos:
        twiml_xml = collect_end_conversation(phrases.bye)

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
    too_many_calls = out[out['number_of_calls'] >= 6].drop(unsend_inf, axis=1)
    out = out.drop(out[out['number_of_calls'] >= 6].index)
    hist_data = pd.concat([hist_data, not_lack_call, too_many_calls], ignore_index=True)
    hist_data.to_csv('history.csv', index=False)


def send_inf(phone):
    """

    """
    unsend_inf = config.unwanted_for_webform_inf
    pers = out[out.phone == phone].drop(unsend_inf, axis=1)
    pers = pers.to_dict('records')[0]
    requests.post(config.recive_url, json=pers)


@app.route('/receiver', methods=['GET', 'POST'])
def receiver():
    print(request.form)


def initiate_call(twiml, phone):
    """

    """
    client.calls.create(url=echo_url + urlencode({'Twiml': twiml}), to=phone, from_=config.twilio_number)


@app.route('/call_auto', methods=['GET', 'POST'])
def call_auto():
    """

    """
    while True:
        if config.lower_limit <= datetime.utcnow().hour < config.upper_limit:
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

    filtered_for_recall = out[out.cnvrs_key == 1][abs(out.call_hour - datetime.utcnow().hour) > config.recall_step] \
        [out.number_of_calls < config.finaly_number_of_calls]
    candidates_to_call = pd.concat([out[out.cnvrs_key == 0], filtered_for_recall])

    if len(candidates_to_call):
        number_of_client = int(config.number_of_calls_per_time / 100 * (100 - config.percent_of_new))
        for i in range(1, -1, -1):
            phone_num_list = candidates_to_call[out.cnvrs_key == i]['phone'][:number_of_client].values
            for phone in phone_num_list:
                set_number_of_calls(phone)
                set_call_hour(phone)
                if i == 1:
                    set_first_qwe(phone=phone, key=True)
                    twiml_xml = collect_redirect_speech(phone=phone)
                else:
                    set_first_qwe(phone=phone, key=False)

                    twiml_xml = collect_2gathers_response(text=phrases.greeting,
                                                          hints=phrases.pst_hint, phone=phone, add_step=False)
                initiate_call(twiml=twiml_xml, phone=phone)
            number_of_client = int(config.number_of_calls_per_time / 100 * config.percent_of_new)


######################################################################################################################


@app.route('/add_person', methods=['GET', 'POST'])
def add_person():
    """

    """
    global out
    if request.method == "POST":
        json = request.get_json()
        print(json)
        # values = [[json[k]] for k in config.flds]
        # values = np.array(values).T
        # dt = pd.DataFrame(data=values, columns=config.flds)
        dt = pd.DataFrame.from_dict(json)
        out = pd.concat([out, dt])
        print('\n\n\n\n\n\n\n', out)


@app.route('/snd', methods=['POST', 'GET'])
def snd():
    pers = {'photo_url': 'http://www.letchworthmini.co.uk/s/cc_images/cache_71011477.JPG', 'phone': '+9379992'}
    requests.post('https://0d5cbef5.ngrok.io/extract_num', json=pers)


@app.route('/extract_num', methods=['POST', 'GET'])
def extract_num():
    # photo_url = request.get_json()['photo_url']

    xls = pd.ExcelFile(request.files['pic'])

    df = xls.parse(xls.sheet_names[0],  converters={"phone": str})
    print(df)
    req = df.to_dict()
    print(req)
    req['reg_num'] = {}

    global graph
    with graph.as_default():
        for j in range(len(req['phone'])):
            req['reg_num'][j], _ = Main.main(req['photo_url'][j])

    for i in range(len(config.adding_filds)):
        if i < 3:
            req = add_field_to_dict(0, i, req)

        elif i < 7:
            req = add_field_to_dict(False, i, req)

        elif i == 7:
            req = add_field_to_dict(1, i, req)

        else:
            req = add_field_to_dict(None, i, req)

    del req['photo_url']
    print(req)
    requests.post(config.add_pers_url, json=req)


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


if __name__ == '__main__':
    app.run(host=config.host, port=config.port)
