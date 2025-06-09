from datetime import datetime

def get_time():
    return str(datetime.now()).split('.')[0].replace(' ', '').replace('-', '').replace(':', '')[2:]