import os.path as path


def split_line(line):
    """
    :param line: from an input data file.
    :return: lower-cased words split by whitespaces.
    """
    return line.split('\t')[3].strip().lower().split(' ')


def clean_words(words):
    """
    :param words: a list of raw words.
    :return: a list of words where each word is cleaned from special symbols.
    """
    for w in words:
        w = w.strip('".\'?)(:,!\\[]=/')
        if w.endswith('\'s'):
            w = w[:len(w)-2]
        if w is not '':
            yield w


def settings_file():
    """
    :return: project settings from the 'SETTINGS.json' file
    """
    return path.join(path.dirname(path.realpath(__file__)), '..', 'SETTINGS.json')
