import numpy as np
import json
import codecs
import stanza
import time
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

"""

a) Which content words lead to better results than with the UNK model?
- accumulate lime weights for sentences S, for the class which was predicted for sentence s, for all s in S.

b) How much weight does content words have in the single task and multi-task models?
- For each sentence and each class, calculate:
fraction of content words’ LIME (normalised) weights / fraction of content words
Example:
Gun:0.3 control:0.3 should:0.1 be:0.1 enforced:0.2
"Gun", "control" and "enforced" are content words, then:
((0.3+0.3+0.2)/1) / (3/5) = 0.8 / 0.6 = 1 1/3

"""


""" a """

def word_ranking(lime_weights,topic, model="ST"):
    lime_accum_towardspredictedlabel = {}
    lime_accum_content_words = {}
    word_freqs = {}
    assert len(lime_weights) == len(sentences)
    for i in range(len(sentences)):
        posses = pos_sentences[i]
        toks = tokens[i]
        #for tok in toks:
        #    word_freqs[tok.lower()] = word_freqs.get(tok.lower(),0)+1
        l = predictions[i]
        content_words = [tok.lower() for i, tok in enumerate(toks) if posses[i] in content_word_pos]
        for j in lime_weights[str(i)][ind2label[l]]:
            word_freqs[j[0].lower()] = word_freqs.get(j[0].lower(),0)+1
            lime_accum_towardspredictedlabel[j[0].lower()] = lime_accum_towardspredictedlabel.get(j[0].lower(),0)+float(j[1])
            if j[0].lower() in content_words:
                lime_accum_content_words[j[0].lower()] = lime_accum_content_words.get(j[0].lower(),0)+float(j[1])

    #print(lime_accum_towardspredictedlabel)
    #print(sorted(lime_accum_towardspredictedlabel.items(), key=lambda x: x[1], reverse=True))
    with open("content_word_freq_rankings/"+model+"_"+topic+"_top_ranked_words.json", "w") as f:
        json.dump(sorted(lime_accum_towardspredictedlabel.items(), key=lambda x: x[1], reverse=True),f)
    with open("content_word_freq_rankings/"+model+"_"+topic+"_top_ranked_content_words.json", "w") as f:
        json.dump(sorted(lime_accum_content_words.items(), key=lambda x: x[1], reverse=True),f)
    with open("content_word_freq_rankings/"+model+"_"+topic+"_all_words_frequencies.json", "w") as f:
        json.dump(word_freqs,f)



def content_word_ranking(lime_weights, topic):
    lime_accum_better_sentences = {}
    for indel in mismatch:
        posses = pos_sentences[indel]
        toks = tokens[indel]
        content_words = [tok.lower() for i, tok in enumerate(toks) if posses[i] in content_word_pos]
        assert predictions[indel]==labels[indel]
        y = predictions[indel]
        y_string = ind2label[y]
        for j in lime_weights[str(indel)][y_string]:
            if j[0].lower() in content_words:
                lime_accum_better_sentences[j[0].lower()] = lime_accum_better_sentences.get(j[0].lower(),0)+float(j[1])
    #print(sorted(lime_accum_better_sentences.items(), key=lambda x: x[1], reverse=True))
    with open("content_word_freq_rankings/"+topic+'_top_ranked_content_words_of_better_preds.json', 'w') as f:
        json.dump(sorted(lime_accum_better_sentences.items(), key=lambda x: x[1], reverse=True), f)




def b(lime_weights, mismatch):
    """
    b)
    how much weight does content words have?
    """
    scores_0 = []
    scores_1 = []
    scores_2 = []
    for i in range(len(sentences)):
        posses = pos_sentences[i]
        toks = tokens[i]
        content_words = [tok.lower() for i, tok in enumerate(toks) if posses[i] in content_word_pos]
        #print(content_words)

        for c in ["NoArgument", "Argument_against", "Argument_for"]:
            content_word_lime_weights = [j[1] for j in lime_weights[str(i)][c] if j[0].lower() in content_words]
            sent_lime_weights = [j[1] for j in lime_weights[str(i)][c]]#sum lime vægt på alle ord i sætningen
            if perc_contentwords[i]>0:
                sent_score = (sum(content_word_lime_weights)/sum(sent_lime_weights))/ perc_contentwords[i]
            else:
                sent_score = 0
            if c=="NoArgument":
                scores_0.append(sent_score)
            elif c=="Argument_against":
                scores_1.append(sent_score)
            elif c=="Argument_for":
                scores_2.append(sent_score)


    print("mean score of sentences where standard ST model was better than UNK model",np.mean(np.array(scores_1)[mismatch])) # mean score of sentences where standard ST model was better then UNK model
    print("mean score of all other sentences",np.mean([i for indel, i in enumerate(scores_1) if indel not in mismatch])) # mean score of all other sentences.
    return scores_0, scores_1, scores_2, np.mean(np.array(scores_1)[mismatch]), np.mean([i for indel, i in enumerate(scores_1) if indel not in mismatch])

def b2(lime_weights, sentences, f=np.abs):
    sentence_scores = []
    for i in range(len(sentences)):
        posses = pos_sentences[i]
        toks = tokens[i]
        content_words = [tok.lower() for i, tok in enumerate(toks) if posses[i] in content_word_pos]
        #lime_weights[str(i)]["Argument_against"].sort()
        #lime_weights[str(i)]["Argument_for"].sort()
        content_word_lime_weights_1 = [j[1] for j in sorted(lime_weights[str(i)]["Argument_against"]) if j[0].lower() in content_words]
        sent_lime_weights_1 = [j[1] for j in sorted(lime_weights[str(i)]["Argument_against"])]
        content_word_lime_weights_2 = [j[1] for j in sorted(lime_weights[str(i)]["Argument_for"]) if j[0].lower() in content_words]
        sent_lime_weights_2 = [j[1] for j in sorted(lime_weights[str(i)]["Argument_for"])]

        content_word_lime_weights = [max([i,j]) for i,j in list(zip(f(content_word_lime_weights_1),f(content_word_lime_weights_2)))]
        sent_lime_weights = [max([i,j]) for i,j in list(zip(f(sent_lime_weights_1),f(sent_lime_weights_2)))]
        if perc_contentwords[i]>0:
            sent_score = (sum(content_word_lime_weights)/sum(sent_lime_weights))/ perc_contentwords[i]
        else:
            sent_score = 0
        sentence_scores.append(sent_score)
    return sentence_scores




def class_weights(lime_weights, c, topic, model="ST"):
    class_w = {}
    class_content_w = {}
    for i in range(len(sentences)):
        #l = predictions[i]
        posses = pos_sentences[i]
        toks = tokens[i]
        content_words = [tok.lower() for i, tok in enumerate(toks) if posses[i] in content_word_pos]
        for j in lime_weights[str(i)][c]:
            class_w[j[0].lower()] = class_w.get(j[0].lower(),0)+float(j[1])
            if j[0].lower() in content_words:
                class_content_w[j[0].lower()] = class_content_w.get(j[0].lower(),0)+float(j[1])
    with open("content_word_freq_rankings/"+model+"_"+topic+"_"+c+'_ranked_words.json', 'w') as f:
        json.dump(sorted(class_w.items(), key=lambda x: x[1], reverse=True), f)
    with open("content_word_freq_rankings/"+model+"_"+topic+"_"+c+'_ranked_content_words.json', 'w') as f:
        json.dump(sorted(class_content_w.items(), key=lambda x: x[1], reverse=True), f)


def weight_of_arg_words(lime_weights):
    arg_words = ["if", "would", "potential", "could", "for", "that"]

    # Claim Indicators based on Stab and Gurevich (2014), adjusted to unigrams
    sg = ["accordingly", "result","consequently", "conclude", "that", "clearly",
    "demonstrates", "entails", "follows", "hence", "however", "implies",
    "fact", "opinion", "conclusion", "indicates",
    "follows", "probable",
    "should", "clear", "i", "believe", "mean", "think", "must",
    "contrary", "points", "proves", "shows",
    "suggests", "obvious", "explanation","point",
    "therefore", "thus", "truth", "sum", "may"]

    premise_indicators = ["because", "since", "whereas", "given",
    "reason", "for"]

    conclusion_indicators = ["therefore", "consequently", "hence", "thus",
    "conclusion", "accordingly", "follows", "that","result", "conclusion"]

    all = sg+premise_indicators+conclusion_indicators+arg_words
    u = set(all)
    print("set of argument words:",u)

    argwords_0 = []
    argwords_1 = []
    argwords_2 = []
    all_argwords_present = []
    for i in range(len(sentences)):
        for c in ["NoArgument", "Argument_against", "Argument_for"]:
            arg_word_lime_weights = [j[1] for j in lime_weights[str(i)][c] if j[0].lower() in u]
            arg_word_lime_words = [j[0].lower() for j in lime_weights[str(i)][c] if j[0].lower() in u]
            if len(arg_word_lime_weights)>0:
                if c=="NoArgument":
                    argwords_0.extend(arg_word_lime_weights)
                    all_argwords_present.extend(arg_word_lime_words)
                elif c=="Argument_against":
                    argwords_1.extend(arg_word_lime_weights)
                elif c=="Argument_for":
                    argwords_2.extend(arg_word_lime_weights)
    return argwords_0, argwords_1, argwords_2, all_argwords_present


content_word_pos = ["NOUN", "VERB", "ADJ"]
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos')

topics = ["abortion", "cloning", "death_penalty",
          "gun_control", "marijuana_legalization",
          "school_uniforms", "minimum_wage", "nuclear_energy"]

means_ST_0 = []
means_ST_1 = []
means_ST_2 = []
means_MT_0 = []
means_MT_1 = []
means_MT_2 = []
better_than_UNK_means = []
others_means = []


ARG_means_ST_0 = []
ARG_means_ST_1 = []
ARG_means_ST_2 = []
ARG_means_MT_0 = []
ARG_means_MT_1 = []
ARG_means_MT_2 = []

max_abs_sent_scores = []
MT_max_abs_sent_scores = []


d = []
#topic = topics[3] # temp for testing code
for indel,topic in enumerate(topics):
    # change to use dev sets
    ST_predictions_file = "../argument_mining_output/"+topic+"/2018/UKP_heldout_dev_scores.json"
    ST_UNK_predictions_file = "../argument_mining_output/UNK_models/UKP/"+topic+"/2018/UNKUKP_heldout_dev_scores.json"
    data_file = "../argument_mining_data/UKPArgMin/"+topic+"/UKP_heldout_dev.tsv"
    lime_file = "content_word_analysis_lime_output/heldout_dev/"+topic+"_2018__0_lime_output.txt"
    MT_lime_file = "content_word_analysis_lime_output/heldout_dev/VacClaim_UKP_IBM_AQ_"+topic+"_2018__0_lime_output.txt"

    f = open(str(ST_predictions_file))
    j_obj = json.load(f)
    predictions = j_obj["predictions"]
    pred_indices = j_obj["uids"]
    f.close()

    f_unk = open(ST_UNK_predictions_file)
    unk_j_obj = json.load(f_unk)
    unk_predictions = unk_j_obj["predictions"]
    f_unk.close()

    f_lime = open(lime_file)
    lime_weights = json.load(f_lime)
    f_lime.close()

    f_MT_lime = open(MT_lime_file)
    MT_lime_weights = json.load(f_MT_lime)
    f_MT_lime.close()

    data = codecs.open(str(data_file), "r", "utf-8").read().split("\n")
    sentences = [data[i].split("\t")[2] for i in range(len(data)-1)]
    labels = [data[i].split("\t")[1] for i in range(len(data)-1)]
    labels = [labels[int(i)] for i in pred_indices]
    label2ind = {"NoArgument":0, "Argument_against":1, "Argument_for":2}
    ind2label = {0:"NoArgument", 1:"Argument_against", 2:"Argument_for"}
    labels = [label2ind[i] for i in labels]

    print(len(labels))
    print(len(predictions))


    """ 0 """
    time_s = time.time()
    tokens = []
    pos_sentences = []
    for text in sentences:
        doc = nlp(text)
        toks = [word.text for sent in doc.sentences for word in sent.words]
        tokens.append(toks)
        pos_sents = [word.upos for sent in doc.sentences for word in sent.words]
        pos_sentences.append(pos_sents)
    time_e = time.time()
    print("Time: ", time_e-time_s)
    num_content_words_per_sentence = [len([j for j in i if j in content_word_pos]) for i in pos_sentences]
    perc_contentwords = [num_content_words_per_sentence[indel]/len(i) for indel, i in enumerate(pos_sentences)]
    """ 0 end"""
    # indices of sentences where the standard model predicted correctly while the UNK model did not
    mismatch = [int(i) for i in pred_indices if labels[int(i)]==predictions[int(i)] if predictions[int(i)]!=unk_predictions[int(i)]]
    print(len(mismatch))
    print(mismatch)
    # view mismatch preds:
    #preds_mismatch = np.array(predictions)[mismatch]
    #print(preds_mismatch)
    #unk_preds_mismatch = np.array(unk_predictions)[mismatch]
    #print(unk_preds_mismatch)

    #word_ranking(lime_weights, topic)
    #word_ranking(MT_lime_weights, topic, model="MT")

    scores_0, scores_1, scores_2, better_than_UNK_mean, others_mean = b(lime_weights, mismatch)
    MT_scores_0, MT_scores_1, MT_scores_2, _, _= b(MT_lime_weights,mismatch)

    means_ST_0.append(np.mean(scores_0))
    means_ST_1.append(np.mean(scores_1))
    means_ST_2.append(np.mean(scores_2))
    means_MT_0.append(np.mean(MT_scores_0))
    means_MT_1.append(np.mean(MT_scores_1))
    means_MT_2.append(np.mean(MT_scores_2))
    better_than_UNK_means.append(better_than_UNK_mean)
    others_means.append(others_mean)

    ST_sent_scores = b2(lime_weights, sentences)
    MT_sent_scores = b2(MT_lime_weights, sentences)
    max_abs_sent_scores.append(np.mean(ST_sent_scores))
    MT_max_abs_sent_scores.append(np.mean(MT_sent_scores))

    #plt.scatter(np.arange(len(scores_1)),scores_1, c="g", label="single task")
    #plt.scatter(np.arange(len(scores_1)),MT_scores_1, c="r", label="multitask")
    #plt.boxplot([scores_0, MT_scores_0], showfliers=False)
    #plt.legend()
    #plt.savefig(topic+"_sentence_scores.png")

    argwords_0, argwords_1, argwords_2, all_argwords_present = weight_of_arg_words(lime_weights)
    print(len(argwords_0), len(argwords_1), len(argwords_2), len(all_argwords_present))
    print("all arg words present in this topic (dev set):",all_argwords_present)
    MT_argwords_0, MT_argwords_1, MT_argwords_2, _ = weight_of_arg_words(MT_lime_weights)

    #plt.boxplot([argwords_0, MT_argwords_0] positions=[indel])


    print("ST arg",np.mean(argwords_0), np.mean(argwords_1), np.mean(argwords_2))
    print("MT arg",np.mean(MT_argwords_0), np.mean(MT_argwords_1), np.mean(MT_argwords_2))
    for i in range(len(all_argwords_present)):
        d.append({
        "model":"ST",
        "topic": topic,
        "class_0_weight":argwords_0[i],
        "class_1_weight":argwords_1[i],
        "class_2_weight":argwords_2[i],
        "word": all_argwords_present[i]
        })
        d.append({
        "model":"MT",
        "topic":topic,
        "class_0_weight":MT_argwords_0[i],
        "class_1_weight":MT_argwords_1[i],
        "class_2_weight":MT_argwords_2[i],
        "word": all_argwords_present[i]
        })
df = pd.DataFrame(d)

print(df.head())

#df.to_csv("lime_weights_for_argument_words.csv",index=False)

print("ST NoArg mean sentence scores: ", means_ST_0)
print("ST Arg against mean sentence scores: ",means_ST_1)
print("ST Arg for mean sentence scores: ", means_ST_2)

print("MT NoArg mean sentence scores: ", means_MT_0)
print("MT Arg against mean sentence scores: ",means_MT_1)
print("MT Arg for mean sentence scores: ", means_MT_2)

print("ST-MT Diff. NoArg:", stats.wilcoxon(means_ST_0,means_MT_0))
print("ST-MT Diff. Arg Against:", stats.wilcoxon(means_ST_1,means_MT_1))
print("ST-MT Diff. Arg for:", stats.wilcoxon(means_ST_2,means_MT_2))

print("ST Better than UNK mean sentence scores:", better_than_UNK_means)
print("ST Others mean sentence scores:", others_means)
print("Test differnce between mean scores of sentences where ST was better than UNK and those where it wasn't: ",stats.wilcoxon(better_than_UNK_means,others_means))


print()
print()
print("ST Max abs sent scores of argument classes:", max_abs_sent_scores)
print("MT Max abs sent scores of argument classes:", MT_max_abs_sent_scores)
print("Diff between ST and MT with max abs scores:",stats.wilcoxon(max_abs_sent_scores,MT_max_abs_sent_scores))

#word_ranking(lime_weights, topic)
#word_ranking(MT_lime_weights, topic, model="MT")

#content_word_ranking(lime_weights, topic)

#for c in ["NoArgument", "Argument_against", "Argument_for"]:
#    class_weights(lime_weights, c, topic, model="ST")
#    class_weights(MT_lime_weights, c, topic, model="MT")
