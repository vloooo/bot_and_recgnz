lwr_time_lim = 8      # time when we don't want to call
upr_time_lim = 20     #
recall_day_step = 1  # how many days we wait to next call to client
sleep_min = 3        # step between initiates call queue
timeout = 3        # time of silence that twilio

account_sid = 'ACf619e2ec98b61829ae5bf61ee745c29a'
auth_token = '225f34810027794296c7bdfc5003fba5'
twilio_numbers = {'Eng': '+16267689921', 'Sco': "+44", 'Wel': '+44', 'Irl': '+44'}
phone_for_test = '+380999051660'

main_url = 'https://c798f811.ngrok.io/'
recive_url = main_url + 'receiver'  # url to send collected clients inf
qwestion_url = main_url + 'question'
echo_url = 'http://twimlets.com/echo?'  # url for twimlets

last_call = 4   # bigest number of calls to one client
percent_of_new = 50  # percent new clients in calling queue
num_calls_per_time = 20  # maximum calls in queue
digits_per_phone = 10  # how much digits have to contain phone number in concrete country

host = '0.0.0.0'
port = 5000

# fields that necessary to inner Db
adding_filds = ['num_calls', 'call_day', 'cnvrs_key', 'pst', 'ngt', 'again_key', 'first_ques', 'repeat_qwe',
                'stage', 'phone_for_offer', 'serv_hist', 'accept']

# all fields from innerDb
flds = ['num_calls', 'call_day', 'cnvrs_key', 'pst', 'ngt', 'again_key', 'first_ques', 'repeat_qwe',
        'stage', 'phone_for_offer', 'mileage', 'serv_hist', 'city', 'reg_num', 'phone', 'accept', 'img_url']

# fields that don't saves for anltc.
unwanted_inf = ['first_ques', 'pst', 'ngt', 'again_key', 'img_url', 'repeat_qwe']

# fields that don't sends to web-form
unwanted_for_webform_inf = ['first_ques', 'pst', 'ngt', 'cnvrs_key', 'num_calls', 'again_key', 'stage',
                            'call_day', 'img_url', 'repeat_qwe']


'''
lnxinstance.westeurope.cloudapp.azure.com
'''