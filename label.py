import re
import time


def main():
    data, labels = '', ''
    with open('data/short_test/short.txt', encoding='utf-8') as f:
        for line in f:
            def repl_cb(item):
                match = item.group(1)
                if match == ' ':
                    return match
                return item.group(0)
            line = re.sub(r'( ){2,}', repl_cb, line)
            data += line
    labelg = ''
    for i in range(len(data) - 1):
        cur, nxt = data[i], data[i + 1]
        if cur == ' ':
            continue
        elif cur == '\n':
            labelg += '\n'
        if nxt in (' ', '\n'):
            labelg += '1'
        else:
            labelg += '0'
    labelg += '1'
    with open('data/short_test/short_label.txt', encoding='utf-8') as f:
        for line in f:
            labels += line
    data_no_spaca = data.replace(' ', '')
    data_no_space = ''
    with open('data/short_test/short_no_space.txt', encoding='utf-8') as f:
        for line in f:
            data_no_space += line
    assert(len(labelg) == len(labels))
    assert(labelg == labels)


if __name__ == "__main__":
    main()
