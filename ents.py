import phrases

ents = {
    'pst': ["yes", "yeah", "exactly", "ok", "yeh", "yep", "correct", "affirmative", "okey",
            "perfect", "great", "cool", "accept", "microwavable", "precisely", "available", "acceptable"],

    'pst_1': ["you are right", "just so", "still available", "yes, it is", "ok thats great", "perfect, thanks",
            "whats offer", "thank you for contact me", "cool", "why not", "if its free", "accept", "whats the offer",
            "thanks for the offer", "you may well say so", "go ahead"],

    'ngt': ["sold", "no", "didnt" "not", "nope", "dont", "salt"],

    'ngt_1': ["its sold", "no way", "sorry, sold", "its gone", "it sold"
            "sold it", "i dont have a car for sale", "you have missed it",
            "no thanks", "not for me", "i dont think so", "fuck off", "piss off", "thats not my",
            "i dont own", "thats not correct", "thats not", "is wrong", "i live in", "where did you get",
            "why do you need to know that?", "my mobile is", "dont know", "what is service history",
            "i have lost", "not interested", "changed my mind"]
}

subjects = ['my greeting', 'registration number of your car', ' mileage of your car', 'the nearest city to you',
            'phone number to offer', 'your car service history', 'making you offer']

repeat_subj = [' is your car available?', 'are you interested?', '', '', 'the nearest city to you.',
               phrases.fivth_stage, phrases.sixth_stage]

serv_hist = ['1', '2', '3', 'full', 'part', 'parts', 'missing', 'lost', 'miss', 'no']

serv_hist_kind = {'1': 'Full service history', 'full': 'Full service history',
                  '2': 'Part service history', 'part': 'Part service history', 'parts': 'Part service history',
                  '3': 'haven\'t service history', 'no': 'haven\'t service history',
                  'miss': 'haven\'t service history', 'lost': 'haven\'t service history',
                  'missing': 'haven\'t service history'}

cities = ['London', 'Birmingham', 'Leeds', 'Glasgow', 'Sheffield', 'Bradford', 'Liverpool', 'Edinburgh', 'Manchester',
          'Bristol', 'Kirklees', 'Fife', 'Wirral', 'North Lanarkshire', 'Wakefield', 'Cardiff', 'Dudley', 'Wigan',
          'East Riding', 'South Lanarkshire', 'Coventry', 'Belfast', 'Leicester', 'Sunderland', 'Sandwell', 'Doncaster',
          'Stockport', 'Sefton', 'Nottingham', 'Newcastle upon Tyne', 'Kingston upon Hull', 'Bolton', 'Walsall', 'Arun',
          'Plymouth', 'Rotherham', 'Stoke on Trent', 'Wolverhampton', 'Rhondda', 'Cynon', 'Taff', 'Oldham', 'Derby',
          'South Gloucestershire', 'Swansea', 'Salford', 'Aberdeenshire', 'Barnsley', 'Tameside', 'Trafford', 'York',
          'Aberdeen', 'Southampton', 'Highland', 'Rochdale', 'Solihull', 'Gateshead', 'Milton Keynes', 'North Tyneside',
          'Calderdale', 'Northampton', 'Portsmouth', 'Warrington', 'North Somerset', 'Bury', 'Luton', 'Suffolk Coastal',
          'Stockton on Tees', 'Renfrewshire', 'Thamesdown', 'Southend on Sea', 'New Forest', 'Caerphilly', 'Wycombe',
          'Carmarthenshire', 'Bath and North East Somerset', 'Basildon', 'Bournemouth', 'Peterborough', 'Colchester',
          'North East Lincolnshire', 'Chelmsford', 'Brighton', 'South Tyneside', 'Charnwood', 'Aylesbury Vale', 'Swale',
          'Knowsley', 'North Lincolnshire', 'Huntingdonshire', 'Macclesfield', 'Blackpool', 'West Lothian', 'Newbury',
          'South Somerset', 'Dundee', 'Basingstoke and Deane', 'Harrogate', 'Dumfries and Galloway', 'Middlesbrough',
          'Flintshire', 'Rochester upon Medway', 'The Wrekin', 'Falkirk', 'Reading', 'Wokingham', 'North Ayrshire',
          'Windsor and Maidenhead', 'Maidstone', 'Redcar and Cleveland', 'Blackburn', 'Neath Port Talbot', 'Poole',
          'Wealden', 'Bedford', 'Lancaster', 'Newport', 'Canterbury', 'Preston', 'Dacorum', 'Cherwell', 'Mid Sussex',
          'Perth and Kinross', 'Thurrock', 'Tendring', 'Kings Lynn and West Norfolk', 'St Albans', 'Bridgend', 'Halton',
          'Elmbridge','South Cambridgeshire', 'Braintree', 'Norwich', 'Thanet', 'Isle of Wight', 'South Oxfordshire',
          'Guildford','Stafford', 'Powys', 'East Hertfordshire', 'Torbay', 'Wrexham Maelor', 'East Devon', 'Horsham',
          'East Lindsey', 'Warwick', 'East Ayrshire', 'Newcastle under Lyme', 'North Wiltshire', 'South Kesteven',
          'Epping Forest', 'Vale of Glamorgan', 'Reigate and Banstead', 'Chester', 'Mid Bedfordshire',  'St Helens',
          'Nuneaton and Bedworth', 'Gwynedd', 'Havant and Waterloo', 'Teignbridge', 'Cambridge', 'Vale Royal', 'Oxford',
          'Amber Valley', 'North Hertfordshire', 'South Ayrshire', 'Waverley', 'Broadland', 'Crewe and Nantwich',
          'Breckland', 'Ipswich', 'Pembrokeshire', 'Vale of White Horse', 'Salisbury', 'Gedling', 'Eastleigh', 'Conway',
          'Broxtowe', 'Stratford on Avon', 'South Bedfordshire', 'Angus', 'East Hampshire', 'East Dunbartonshire',
          'Sevenoaks', 'Slough', 'Bracknell Forest', 'West Lancashire', 'West Wiltshire', 'Ashfield', 'Lisburn',
          'Scarborough', 'Stroud', 'Wychavon', 'Waveney', 'Exeter', 'Dover', 'Test Valley', 'Gloucester', 'Erewash',
          'Cheltenham', 'Bassetlaw', 'Scottish Borders']


part_of_cities = {'West Lancashire': 'Lancashire', 'Bracknell Forest': 'Bracknell', 'East Hampshire': 'Hampshire',
                  'East Dunbartonshire': 'Dunbartonshire', 'Kingston upon Hull': 'Kingston', 'East Riding': 'Riding',
                  'Stratford on Avon': 'Stratford', 'Crewe and Nantwich': 'Crewe', 'crewe and Nantwich':'Nantwich',
                  'Amber Valley': 'Amber', 'Havant and Waterloo': 'Havant', 'Nuneaton and Bedworth': 'Nuneaton',
                  'nuneaton and Bedworth': 'Bedworth', 'Suffolk Coastal': 'Suffolk', 'reigate and Banstead': 'Banstead',
                  'Reigate and Banstead': 'Reigate', 'Vale of Glamorgan': 'Glamorgan', 'South Kesteven': 'Kesteven',
                  'Newcastle upon Tyne': 'Newcastle', 'Wrexham Maelor': 'Wrexham', 'wrexham Maelor': 'Maelor',
                  'East Devon': 'Devon', 'East Lindsey': 'Lindsey', 'Dumfries and Galloway': 'Galloway',
                  'Isle of Wight': 'Wight', 'South Cambridgeshire': 'Cambridgeshire', 'St Albans': 'Albans',
                  'Kings Lynn and West Norfolk': 'Lynn', 'Kings Lynn & West Norfolk': 'Norfolk', 'Mid Sussex':'Sussex',
                  'Perth and Kinross': 'Perth', 'Perth & Kinross': 'Kinross', 'Windsor and Maidenhead': 'Windsor',
                  'Redcar and Cleveland': 'Redcar', 'Redcar & Cleveland': 'Cleveland', 'Neath Port Talbot': 'Talbot',
                  'The Wrekin': 'Wrekin', 'Rochester upon Medway': 'Rochester', 'Rochester up. Medway': 'Medway',
                  'Dumfries & Galloway': 'Dumfries', 'Basingstoke and Deane': 'Basingstoke', 'St Helens': 'Helens',
                  'Basingstoke & Deane': 'Deane', 'West Lothian': 'Lothian', 'Aylesbury Vale': 'Aylesbury',
                  'Southend on Sea': 'Southend', 'Stockton on Tees': 'Stockton', 'Milton Keynes': 'Milton',
                  'milton keynes': 'Keynes', 'South Gloucestershire': 'Gloucestershire', 'Stoke on Trent': 'Stoke',
                  'Windsor & Maidenhead': 'Maidenhead', 'South Oxfordshire': 'Oxfordshire', 'Stoke Trent': 'Trent'}
