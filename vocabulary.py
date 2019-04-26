import json

from common import clean_words, settings_file, split_line


with open(settings_file()) as settings_f:
    settings = json.load(settings_f)
    # datasets = ['train', 'test', 'validate']
    datasets = ['train']
    words = set()
    for ds in datasets:
        with open(settings[ds]) as in_f:
            for line in in_f:
                words.update(clean_words(split_line(line)))

    vocabulary = list(words)
    vocabulary.sort()
    with open(settings['vocabulary'], mode='w') as out_f:
        for v in vocabulary:
            print(v, file=out_f)
