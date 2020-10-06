from lxml import html
import requests
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as bs
import subprocess
lang_dict = {}
with open('lstm/data/wili-2018/labels.csv', 'r') as f:
    for line in f.readlines():
        s = line.split(';')
        code = s[0]
        code_2 = s[2]
        lang_dict[code_2] = code

with open('../twituser-v1/twituser', 'r') as f:
    lines = f.readlines()

f =  open('x_twituser_cleaned.txt', 'w')
f.close()

f =  open('y_twituser.txt', 'w')
f.close()
    


written = ""
for i, line in enumerate(lines): #[:100]:
    d = eval(line)
    clean_1 = d['text'].replace('\n',' ').replace('\r', ' ')

    clean = ""
    for word in clean_1.split():
        if word.startswith('http') or word.startswith('@') or word == 'RT' or word.startswith('“@') or\
            word.startswith('"@'):
            continue 
        clean += word + " "
    clean = clean[:-1]

    with open('x_twituser_cleaned.txt', 'a') as f:
        f.write(clean)
        if '\r' in clean or '\n' in clean: print(clean)
        f.write('\n')
    with open('y_twituser.txt', 'a') as f:
        lan = d['lang']
        f.write(lang_dict[lan])
        f.write('\n')


