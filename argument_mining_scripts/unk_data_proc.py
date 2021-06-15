import stanza
import sys
import codecs
import pandas as pd
from content_word_utils import retrieve
stanza.download('en')


path = sys.argv[1]
file = sys.argv[2]

def process(path, file, keep_top10=False, keep_common=False):
    if ".csv" in file:
        data = pd.read_csv(path+file)
        texts = texts = data.text.values
        ids = data.index
        labels = data.main_label
        file = file[:len(file)-4]+".tsv"
    else:
        data = codecs.open(path+file, "r", "utf-8").read().split("\n")
        texts = [data[i].split("\t")[2] for i in range(len(data)-1)]
        labels = [data[i].split("\t")[1] for i in range(len(data)-1)]
        ids = [data[i].split("\t")[0] for i in range(len(data)-1)]

    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos')

    to_replace = ["NOUN", "VERB", "ADJ"] # content words: nouns, verbs, adjectives

    if keep_top10:
        topics = ["abortion", "cloning", "death_penalty",
                  "gun_control", "marijuana_legalization",
                  "school_uniforms", "minimum_wage", "nuclear_energy"]
        for i in topics:
            if i in path:
                topic = i
        print(topic)
        wordset = retrieve(topic, n=10, content_words_only=True, no_n_limit=False, ignore_freqs=True)

    common_cont_words = {'political', 'single', 'debate', 'had', 'asked', 'made',
                        'policy', 'last', 'legal', 'cause', 'long', 'few', 'said',
                        'want', 'person', 'issue', 'say', 'group', 'possible',
                        'use', 'people', 'believe', 'good', 'have', 'fact', 'point',
                        'society', 'time', 'such', 'going', 'put', 'used', 'come',
                        'based', 'question', 'think', 'example', 'part', 'other',
                        'are', 'year', 'including', 'argument', 'only', 'way',
                        'effects', 'go', 'many', 'support', 'more', 'several',
                        'end', 'has', 'day', 'see', 'need', 'make', 'get', 'means',
                        'public', 'is', 'high', 'help', 'money', 'find', 'found',
                        'same'}

    if keep_top10:
        savefilename = path+"UNKtop10topicdependentincluded"+file
    elif keep_common:
        savefilename = path+"UNKcommonContentWords"+file
    else:
        savefilename = path+"UNK"+file

    with open(savefilename, "w+", encoding='utf-8') as f:
        for indel, text in enumerate(texts):
            doc = nlp(text)
            tokens = [word.text for sent in doc.sentences for word in sent.words]
            u_pos = [word.upos for sent in doc.sentences for word in sent.words]
            #x_pos = [word.xpos for sent in doc.sentences for word in sent.words]

            new_sentence = []
            for i, pos in enumerate(u_pos):
                if pos in to_replace:
                    if keep_top10:
                        if tokens[i] in wordset:
                            new_sentence.append(tokens[i])
                        else:
                            new_sentence.append("[UNK]")
                    elif keep_common:
                        if tokens[i] in common_cont_words:
                            new_sentence.append(tokens[i])
                        else:
                            new_sentence.append("[UNK]")
                    else:
                        new_sentence.append("[UNK]")
                else:
                    new_sentence.append(tokens[i])
            new_sentence = " ".join(new_sentence)
            f.write("%s\t%s\t%s\n" % (str(ids[indel]),str(labels[indel]), new_sentence))

process(path, file, keep_common=sys.argv[3])
