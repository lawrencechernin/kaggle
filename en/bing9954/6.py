#FROM https://www.kaggle.com/alphasis/enhance-your-baseline-lb-0-9939
########################################################
#Lawrence: how to handle kilogram vs kilograms vs kg vs kgs vs 1.1kg vs 1.1 kg vs background??
#Lawrence: find error cases on train??
########################################################

import os
import operator
from num2words import num2words
import gc

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False



INPUT_PATH = r'../input'
DATA_INPUT_PATH = r'../input/en_with_types'  # need to download this huge file, see https://storage.googleapis.com/text-normalization/en_with_types.tgz
SUBM_PATH = INPUT_PATH

SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
OTH = str.maketrans("፬一二三四五六七八九", "4123456789") #四=Japanese 4

print('Train start...')

file = "en_train_fixed.csv" # fixed four bad trainings
train = open(os.path.join(INPUT_PATH, "en_train.csv"), encoding='UTF8')
line = train.readline()
res = dict()
total = 0
not_same = 0
while 1:
    line = train.readline().strip()
    if line == '':
        break
    total += 1
    pos = line.find('","')
    text = line[pos + 2:]
    if text[:3] == '","':
        continue
    text = text[1:-1]
    arr = text.split('","')
    #print("1 LINE:", line, "pos:", pos,",text:", text, "arr:", arr)
    if arr[0] != arr[1]:
        not_same += 1
    if arr[0] not in res:
        res[arr[0]] = dict()
        res[arr[0]][arr[1]] = 1
    else:
        if arr[1] in res[arr[0]]:
            res[arr[0]][arr[1]] += 1
        else:
            res[arr[0]][arr[1]] = 1
train.close()
# res contains before and after patterns
print(file + ':\tTotal: {} Have diff before/after values: {}'.format(total, not_same))

files = os.listdir(DATA_INPUT_PATH)
for file in files:
    train = open(os.path.join(DATA_INPUT_PATH, file), encoding='UTF8')
    while 1:
        line = train.readline().strip()
        if line == '':
            break
        total += 1
        pos = line.find('\t')
        text = line[pos + 1:]
        if text[:3] == '':
            continue
        arr = text.split('\t')
        if arr[0] == '<eos>':
            continue
        if arr[1] != '<self>':
            not_same += 1

        if arr[1] == '<self>' or arr[1] == 'sil':
            arr[1] = arr[0]

        if arr[1] == '<self>' or arr[1] == 'sil':
            arr[1] = arr[0]

        if arr[0] not in res:
            res[arr[0]] = dict()
            res[arr[0]][arr[1]] = 1
        else:
            if arr[1] in res[arr[0]]:
                res[arr[0]][arr[1]] += 1
            else:
                res[arr[0]][arr[1]] = 1
    train.close()
    print(file + ':\tTotal: {} Have diff before/after values: {}'.format(total, not_same))
    gc.collect()
    #break #let this be for quick runs

sdict = {}
sdict['km2'] = 'square kilometers'
sdict['km'] = 'kilometers'
sdict['kg'] = 'kilograms'
sdict['lb'] = 'pounds'
sdict['dr'] = 'doctor'
sdict['m²'] = 'square meters'
# added by Lawrence
sdict['lbs'] = 'pounds'
sdict['mi²'] = 'square miles'
sdict['m³'] = 'cubic meters'
sdict['m³/s'] = 'cubic meters per second'
sdict['m/s'] = 'meters per second'
sdict['km/s'] = 'kilometers per second'
sdict['mg'] = 'milligrams'
sdict['µg'] = 'micrograms'
sdict['µm'] = 'micrometers'
sdict['µs'] = 'microseconds'

total = 0
changes = 0
out = open(os.path.join('baseline_ext_en6.csv'), "w", encoding='UTF8')
out.write('"id","after"\n')
test = open(os.path.join(INPUT_PATH, "en_test.csv"), encoding='UTF8')
line = test.readline().strip()
found_direct_in_training=0
line_is_digit=0
line_has_multi_words=0
print("res:", res)
while 1:
    line = test.readline().strip()
    if line == '':
        break

    pos = line.find(',')
    i1 = line[:pos]
    line = line[pos + 1:]

    pos = line.find(',')
    i2 = line[:pos]
    line = line[pos + 1:]

    line = line[1:-1]
    out.write('"' + i1 + '_' + i2 + '",')
    if line in res:
        srtd = sorted(res[line].items(), key=operator.itemgetter(1), reverse=True)
        #print("Found matches of line:", line, "in res , written:", srtd[0][0])
        out.write('"' + srtd[0][0] + '"')
        changes += 1
        found_direct_in_training += 1
    else:
        if len(line) > 1:
            val = line.split(',')
            # this removes commas in a number, e.g. 21,000 ==> 21000
            if len(val) == 2 and val[0].isdigit and val[1].isdigit:
                line = ''.join(val)
        
        if line.isdigit():
            line_is_digit+=1
            srtd = line.translate(SUB)
            srtd = srtd.translate(SUP)
            srtd = srtd.translate(OTH)
            out.write('"' + num2words(float(srtd)) + '"')
            changes += 1
            #print("Changed is a digit, before:", line, ",after:", num2words(float(srtd)))
        elif len(line.split(' ')) > 1:
            val = line.split(' ')
            line_has_multi_words+=1
            for i, v in enumerate(val):
                if v.isdigit():
                    srtd = v.translate(SUB)
                    srtd = srtd.translate(SUP)
                    srtd = srtd.translate(OTH)
                    val[i] = num2words(float(srtd))
                elif v in sdict:
                    val[i] = sdict[v]

            #print("MULTI", line, ",after:", ' '.join(val))
            out.write('"' + ' '.join(val) + '"')
            changes += 1
        else:
            if line == 'inf' or line == '-inf':
                #print("Strange value:", line)
                out.write('"' + line + '"')
            elif is_number(line):
                #print("DEBBB["+line+"]",type(line))
                in_words = num2words(float(line))
                #print("NUM2WORDS, ", line, "==>", in_words)
                out.write('"' + in_words + '"')

            else: 
                print("DARN, no change for line:", line)
                out.write('"' + line + '"')

    out.write('\n')
    total += 1

print('Total: {} Changed: {}'.format(total, changes))
test.close()
out.close()
