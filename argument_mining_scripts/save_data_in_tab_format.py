import pandas as pd
import numpy as np
import codecs
import os
import sys

#root = "./" # put in location of repo

data_path = os.path.join("..","argument_mining_data/")
#model_path = root+"mt-dnn/"
#script_path = root+"spurious_correlations_in_argmin/argument_mining_scripts/"
#output_dir= root+"spurious_correlations_in_argmin/argument_mining_output"

def prepare_dfs(topic, set_name, nrows=None, two_label=False):
    #data = codecs.open(data_path+"/complete/"+topic+".tsv", "r", "utf-8").read().split("\n")
    data = codecs.open(data_path+"ArgMin/"+topic+".tsv", "r", "utf-8").read().split("\n")
    data = [data[i].split("\t") for i in range(len(data)-1)]
    df = pd.DataFrame(data[1:], columns=data[0])
    premises = list(df[df.set==set_name].sentence)
    labels = list(df[df.set==set_name].annotation)
    if two_label==True:
        labels = [i if i=="NoArgument" else "Argument" for i in labels]
    return labels, premises


def collect_data(set_name, list_name, topics, include_topic=False, two_label=False):
    for topic in topics:
        label, premise = prepare_dfs(topic, set_name, two_label)
        for (i, j) in zip(label, premise):
            if include_topic==True:
                list_name.append([i,j,topic])
            else:
                list_name.append([i,j])


def save_files_in_tab_format(savefile_name,list_name, include_topic=False):
    with open(savefile_name, "w+", encoding='utf-8') as f:
        for indel, i in enumerate(list_name):
            if include_topic==True:
                f.write("%s\t%s\t%s\t%s\n" % (str(indel),str(i[0]), i[1], i[2]))
            else:
                f.write("%s\t%s\t%s\n" % (str(indel),str(i[0]), i[1]))


indel = sys.argv[1]
i = sys.argv[2]

UKP_topics = ["abortion", "cloning", "death_penalty", "gun_control", "marijuana_legalization", "school_uniforms", "minimum_wage", "nuclear_energy"]
#for indel, i in enumerate(UKP_topics):

train_top = [UKP_topics[i] for i,j in enumerate(UKP_topics) if i!=indel]
held_out_top = [i]
print("held out topic: ", i)

UKP_train = []
UKP_dev = []
UKP_test = []
heldout_test = []

#collect_data("train", UKP_train, train_top, include_topic=True, two_label=True)
#collect_data("val", UKP_dev, train_top, include_topic=True, two_label=True)
#collect_data("test", UKP_test, train_top, include_topic=True, two_label=True)
#collect_data("test", heldout_test, held_out_top, include_topic=True, two_label=True)

collect_data("train", UKP_train, train_top, include_topic=False, two_label=False)
collect_data("val", UKP_dev, train_top, include_topic=False, two_label=False)
collect_data("test", UKP_test, train_top, include_topic=False, two_label=False)
collect_data("test", heldout_test, held_out_top, include_topic=False, two_label=False)

#collect_data("val", heldout_test, held_out_top, include_topic=False, two_label=False)

save_files_in_tab_format(data_path+"UKPArgMin/"+i+"/UKP_train.tsv",UKP_train, include_topic=False)
save_files_in_tab_format(data_path+"UKPArgMin/"+i+"/UKP_dev.tsv",UKP_dev, include_topic=False)
save_files_in_tab_format(data_path+"UKPArgMin/"+i+"/UKP_test.tsv",UKP_test, include_topic=False)
save_files_in_tab_format(data_path+"UKPArgMin/"+i+"/UKP_heldout_test.tsv",heldout_test, include_topic=False)

#save_files_in_tab_format(data_path+"UKPArgMin/"+i+"/UKP_heldout_dev.tsv",heldout_test, include_topic=False)
