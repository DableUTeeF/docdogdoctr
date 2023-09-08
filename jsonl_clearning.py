import json
import os


def sub(text):
    out = ''
    for char in text:
        if char in vocabs:
            out += char
    return out


if __name__ == '__main__':
    src_dir = "/media/palm/Data/ocr/"
    charmap = open('/home/palm/PycharmProjects/ABINet/data/charset_enth.txt').read().split('\n')
    charset = [x.split('	')[1] for x in charmap]
    vocabs = ''.join(charset)
    jsonl = '/home/palm/PycharmProjects/capocr/data/train.jsonl'
    labels = open(jsonl).read().split('\n')[:-1]
    with open('data/train.jsonl', 'w') as wr:
        for line in labels:
            data = json.loads(line)
            img_name = data['filename'].replace('data/', 'data2/')
            label = data['text']
            if not os.path.exists(os.path.join(src_dir, img_name)):
                continue
            label = sub(label)
            wr.write(json.dumps({'filename': img_name, 'text': label}))
            wr.write('\n')
