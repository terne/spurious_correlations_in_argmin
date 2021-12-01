import json
import pandas as pd

def retrieve(topic, n=20, content_words_only=False, no_n_limit=False, ignore_freqs=False, normed=False):
    # for using content words rather than all words
    if normed:
        file = "content_word_freq_rankings/ST_"+topic+"_top_ranked_words_NORMALIZED.json"
    elif content_words_only:
        file = "content_word_freq_rankings/ST_"+topic+"_top_ranked_content_words.json"
    else:
        file = "content_word_freq_rankings/ST_"+topic+"_top_ranked_words.json"
    print(file)
    f = open(file)
    j_obj = json.load(f)
    f.close()

    f = open("content_word_freq_rankings/ST_"+topic+"_all_words_frequencies.json")
    freqs = json.load(f)
    f.close()

    if no_n_limit:
        wset = set([i[0] for i in j_obj])
    else:
        wset = set([i[0] for i in j_obj[:n]])
    if ignore_freqs:
        words = [(i[0], (round(i[1],2))) for i in j_obj[:n]]
    else:
        words = [(i[0], (round(i[1],2), freqs[i[0]])) for i in j_obj[:n]]
    #counts = [freqs[w] for w in words]
    for i in words:
        print(i)
    print("Number of unique words in heldout dev set:", len(freqs))
    print("Total number of words in heldout dev set:", sum(freqs.values()))
    return wset


def common_content_words(n=20):
    topics = ["abortion", "cloning", "death_penalty",
              "gun_control", "marijuana_legalization",
              "school_uniforms", "minimum_wage", "nuclear_energy"]
    #all content words
    setlist = [retrieve(topic, content_words_only=True, no_n_limit=True, normed=False) for topic in topics]
    print(len(setlist))
    u = set.intersection(*setlist)
    return u


def class_weight_list(model, topic, c, n, content_words_only=False):
    if content_words_only:
        f = open("content_word_freq_rankings/"+model+"_"+topic+"_"+c+'_ranked_content_words.json')
    else:
        f = open("content_word_freq_rankings/"+model+"_"+topic+"_"+c+'_ranked_words.json')
    j_obj = json.load(f)
    f.close()
    words = [i[0] for i in j_obj[:n]]
    return words


def class_weight_overview(model, topic, n, content_words_only=True):
    classes = ["NoArgument", "Argument_against", "Argument_for"]
    top_words_NoArgument = class_weight_list(model, topic, c="NoArgument", n=n, content_words_only=content_words_only)
    top_words_Argument_for = class_weight_list(model, topic, c="Argument_for", n=n, content_words_only=content_words_only)
    top_words_Argument_against = class_weight_list(model, topic, c="Argument_against", n=n, content_words_only=content_words_only)
    df = pd.DataFrame(list(zip(top_words_NoArgument, top_words_Argument_for, top_words_Argument_against)), columns=["NoArgument", "Argument_for", "Argument_against"])
    return df


def normalized_rankings(model, topic):
    f = open("content_word_freq_rankings/ST_"+topic+"_top_ranked_words.json")
    j_obj = json.load(f)
    f.close()

    f = open("content_word_freq_rankings/ST_"+topic+"_all_words_frequencies.json")
    freqs = json.load(f)
    f.close()

    d = {}
    for i in j_obj:
        d[i[0]] = d.get(i[0].lower(),i[1]/freqs[i[0]])

    with open("content_word_freq_rankings/"+model+"_"+topic+"_top_ranked_words_NORMALIZED.json", "w") as f:
        json.dump(sorted(d.items(), key=lambda x: x[1], reverse=True),f)


def top_10_most_important_save():
    topics = ["abortion", "cloning", "death_penalty",
              "gun_control", "marijuana_legalization",
              "school_uniforms", "minimum_wage", "nuclear_energy"]

    wordsets = [retrieve(topic, n=10, content_words_only=True, no_n_limit=False, ignore_freqs=True) for topic in topics]
    return wordsets

#print()
#topic = "nuclear_energy"
#print(topic)
#df = class_weight_overview("ST", topic, 20, content_words_only=False)
#print(df)
#print(df.to_latex(index=False))

#df = class_weight_overview("MT", topic, 20, content_words_only=False)
#print(df)

#retrieve(topic, n=20, content_words_only=False, no_n_limit=False, ignore_freqs=False, normed=True)

#common_content_words(n=100)


#for topic in ["abortion", "cloning", "death_penalty",
    #      "gun_control", "marijuana_legalization",
    #      "school_uniforms", "minimum_wage", "nuclear_energy"]:
    #      normalized_rankings("ST", topic)
    #      normalized_rankings("MT", topic)
