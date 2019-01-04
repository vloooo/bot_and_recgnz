ents = {
    'pst': ["yes", "yeah", "exactly", "you are right", "just so", "ok", "yeh", "yep", "correct", "affirmative",
            "perfect", "still available", "yes, it is", "ok that's great", "perfect, thanks", "thank you for contact me",
            "whats the offer", "thanks for the offer", "cool", "yes please", "why not", "if its free", "accept",
            "precisely", "you may well say so", "available"],

    'ngt': ["sorry it's sold", "no way", "sorry, sold", "it's gone", "sold it", "sold it last week",
            "sold it", "sold", "i don't have a car for sale", "you have missed it", "no",
            "no thanks", "not for me", "i don't think so", "fuck off", "piss off", "that's not my",
            "i don't own", "that's not correct", "that's not", 'is wrong', "i live in",
            "where did you get", "why do you need to know that?", "my mobile is",
            "don't know", "what is service history", 'i have lost', "not interested", "don't", 'changed my mind']
}

subjects = ['my greeting', 'registration number of your car', ' mileage of your car', 'the nearest city to you',
            'phone number to offer', 'your car service history', 'making you offer']

serv_hist = ['1', '2', '3', 'full', 'part', 'no', 'missing', 'lost', 'miss']

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
          'South Gloucestershire',  'Swansea', 'Salford', 'Aberdeenshire', 'Barnsley', 'Tameside', 'Trafford', 'York',
          'Aberdeen', 'Southampton', 'Highland', 'Rochdale', 'Solihull', 'Gateshead', 'Milton Keynes', 'North Tyneside',
          'Calderdale', 'Northampton', 'Portsmouth', 'Warrington', 'North Somerset', 'Bury', 'Luton', 'St Helens',
          'Stockton on Tees', 'Renfrewshire', 'Thamesdown', 'Southend on Sea', 'New Forest', 'Caerphilly', 'Wycombe',
          'Carmarthenshire', 'Bath and North East Somerset', 'Basildon', 'Bournemouth', 'Peterborough', 'Colchester',
          'North East Lincolnshire', 'Chelmsford', 'Brighton', 'South Tyneside', 'Charnwood', 'Aylesbury Vale', 'Swale',
          'Knowsley', 'North Lincolnshire', 'Huntingdonshire', 'Macclesfield', 'Blackpool', 'West Lothian', 'Newbury',
          'South Somerset', 'Dundee', 'Basingstoke and Deane', 'Harrogate', 'Dumfries and Galloway', 'Middlesbrough',
          'Flintshire', 'Rochester upon Medway', 'The Wrekin',  'Falkirk', 'Reading', 'Wokingham', 'North Ayrshire',        ### Bedfordshire Wiltshire Hertfordshire Ayrshire
          'Windsor and Maidenhead', 'Maidstone', 'Redcar and Cleveland', 'Blackburn', 'Neath Port Talbot', 'Poole',
          'Wealden', 'Bedford',  'Lancaster', 'Newport', 'Canterbury', 'Preston', 'Dacorum', 'Cherwell', 'Mid Sussex',
          'Perth and Kinross', 'Thurrock', 'Tendring', 'Kings Lynn and West Norfolk', 'St Albans', 'Bridgend', 'Elmbridge',
          'South Cambridgeshire', 'Braintree', 'Norwich', 'Thanet', 'Isle of Wight',  'South Oxfordshire', 'Guildford',
          'Stafford', 'Powys', 'East Hertfordshire', 'Torbay', 'Wrexham Maelor', 'East Devon', 'East Lindsey', 'Halton',
          'Warwick', 'East Ayrshire', 'Newcastle under Lyme', 'North Wiltshire', 'South Kesteven', 'Epping Forest',
          'Vale of Glamorgan', 'Reigate and Banstead', 'Chester', 'Mid Bedfordshire', 'Suffolk Coastal', 'Horsham',
          'Nuneaton and Bedworth', 'Gwynedd', 'Havant and Waterloo', 'Teignbridge', 'Cambridge', 'Vale Royal', 'Oxford',
          'Amber Valley', 'North Hertfordshire', 'South Ayrshire', 'Waverley', 'Broadland', 'Crewe and Nantwich',
          'Breckland', 'Ipswich', 'Pembrokeshire', 'Vale of White Horse', 'Salisbury', 'Gedling', 'Eastleigh', 'Conway',
          'Broxtowe', 'Stratford on Avon', 'South Bedfordshire', 'Angus', 'East Hampshire', 'East Dunbartonshire',
          'Sevenoaks', 'Slough', 'Bracknell Forest', 'West Lancashire', 'West Wiltshire', 'Ashfield', 'Lisburn',
          'Scarborough', 'Stroud', 'Wychavon', 'Waveney', 'Exeter', 'Dover', 'Test Valley', 'Gloucester', 'Erewash',        ####
          'Cheltenham', 'Bassetlaw', 'Scottish Borders', 'Lancashire', 'Bracknell', 'Hampshire', 'Dunbartonshire',
          'Stratford', 'White Horse', 'Crewe', 'Nantwich', 'Amber', 'Havant', 'Nuneaton', 'Bedworth', 'Suffolk',
          'Reigate', 'Glamorgan', 'Kesteven', 'Newcastle', 'Wrexham', 'Maelor', 'Devon', 'Lindsey', 'Oxfordshire',
          'Wight', 'Cambridgeshire', 'Albans', 'Lynn and Norfolk', 'Perth', 'Kinross', 'Sussex', 'Windsor', 'Maidenhead',
          'Redcar', 'Cleveland', 'Neath Port', 'Talbot', 'Wrekin', 'Rochester', 'Medway']

for i in cities:
    if 'Medway' in i:
        print(i)