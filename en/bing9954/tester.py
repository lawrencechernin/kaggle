#FROM https://www.kaggle.com/alphasis/enhance-your-baseline-lb-0-9939
########################################################
#Lawrence: how to handle kilogram vs kilograms vs kg vs kgs vs 1.1kg vs 1.1 kg vs background??
#Lawrence: find error cases on train??
########################################################

import os
import operator
from num2words import num2words
import gc
import re
from csv import reader


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

def process_test_file(test_file_name,outfile,res):  # use for evaluate training accuracy and producing a submission file on test data
    print("IN process_test_file, test_file_name:["+test_file_name+"]")
    total = 0
    changes = 0
    errors = 0 # for accuracy checking
    out = open(os.path.join(outfile), "w", encoding='UTF8')
    out.write('"id","after"\n')
    print("test_file_name:", test_file_name)
    test = open(os.path.join(INPUT_PATH, test_file_name), encoding='UTF8')
    found_direct_in_training=0
    line_is_digit=0
    line_has_multi_words=0
    splitchar='\t'
    print("splitchar:", splitchar)
    while 1:
        line = test.readline()
        line = line.strip()
        if line == '':
            break
    
        pos = line.find(splitchar)
        class1 = line[:pos]
        line = line[pos + 1:]
    
        pos = line.find(splitchar)
        before = line[:pos]
        line = line[pos + 1:]
    
        after = line
        golden = line
        if after == '<self>':
           after = before
        if after == 'self':
           after = before
        elif after == 'sil':
           after = before
        print("LLL,", line,"b:", before, "after:", after, "class1", class1)


        if before in res:
            srtd = sorted(res[before].items(), key=operator.itemgetter(1), reverse=True)
            modified = srtd[0][0]
            print("MODIFIED 1:", modified)
            out.write('"' + modified + '"')
            changes += 1
            found_direct_in_training += 1
        else:
            if len(before) > 1:
                val = before.split(splitchar)
                # this removes commas in a number, e.g. 21,000 ==> 21000
                if len(val) == 2 and val[0].isdigit and val[1].isdigit:
                    before = ''.join(val)
            
            if before.isdigit():
                line_is_digit+=1
                srtd = before.translate(SUB)
                srtd = srtd.translate(SUP)
                srtd = srtd.translate(OTH)
                modified = num2words(float(srtd))
                out.write('"' + modified + '"')
                print("MODIFIED 2:", modified)
                changes += 1
                #print("Changed is a digit, before:", line, ",after:", num2words(float(srtd)))
            elif len(before.split(' ')) > 1:
                val = before.split(' ')
                line_has_multi_words+=1
                for i, v in enumerate(val):
                    if v.isdigit():
                        srtd = v.translate(SUB)
                        srtd = srtd.translate(SUP)
                        srtd = srtd.translate(OTH)
                        val[i] = num2words(float(srtd))
                    elif v in sdict:
                        val[i] = sdict[v]
    
                print("MULTI before:", before, ",after:", ' '.join(val))
                modified = ' '.join(val)
                out.write('"' + modified + '"')
                print("MIOOODD: ["+modified+"]")
                #line_split = reader([modified])  # use csv reader to split the line more safely, and skip over the punct and before
                print("MODIFIED 3:", modified)
                changes += 1
            else:
                if before == 'inf' or before == '-inf':
                    #print("Strange value:", line)
                    out.write('"' + before + '"')
                    modified = before
                elif is_number(before):
                    #print("DEBBB["+line+"]",type(line))
                    modified = num2words(float(before))
                    print("MODIFIED 4:", modified)
                    #print("NUM2WORDS, ", line, "==>", in_words)
                    out.write('"' + modified + '"')
    
                else: 
                    print("DARN, no change for line:", before)
                    modified = after
                    out.write('"' + modified + '"')
                    print("NO MODIFICATION 5:", modified)
    
        if modified != after:
            errors+=1
            print("Found Errors on input:",before,", modified to:", modified, ",golden:",golden, ",class:", class1)



        out.write('\n')
        total += 1
    
    print('Total: {} Changed: {}'.format(total, changes))
    accuracy = 100* ( 1 - errors*1.0 / total)
    print("ERRORS:", errors, "Accuracy:", accuracy)
    test.close()
    out.close()




print('Train start...')

file = "en_train.csv"
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
    break #let this be for quick runs

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

print("Evaluating accuracy on training data...")
outfile="baseline_ext_en7_temp.csv" # for debugging purposes
test_file='output-00000-of-00100' # run on training so we can see accuracy
process_test_file(test_file,outfile,res)


