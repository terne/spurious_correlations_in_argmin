import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
import json
import codecs
import sys


def UKP_eval(test_scores_file, test_file, n_class, output_dir, topic):
    f = open(str(test_scores_file))
    j_obj = json.load(f)
    predictions = j_obj["predictions"]
    pred_indices = j_obj["uids"]
    f.close()
    #labels = pd.read_csv(test_file, sep="\t", header=None)[1]

    data = codecs.open(str(test_file), "r", "utf-8").read().split("\n")
    labels = [data[i].split("\t")[1] for i in range(len(data)-1)]
    labels = [labels[int(i)] for i in pred_indices] # find labels by index and sort accordingly

    if int(n_class)>2:
        label2ind = {"NoArgument":0, "Argument_against":1, "Argument_for":2}
        labels = [label2ind[i] for i in labels]
    else:
        label2ind = {"NoArgument":0, "Argument":1}
        labels = [label2ind[i] for i in labels]

    F1 = f1_score(labels,predictions, average="macro")
    print("macro F1:", F1)
    F1_weigthed = f1_score(labels,predictions, average="weighted")
    print("weighted macro F1:", F1_weigthed)

    if int(n_class)>2:
        per_label_p_r = precision_recall_fscore_support(labels,predictions, average=None,labels=[0,1,2])

        NoArgument_precision = per_label_p_r[0][0]
        Argument_for_precision = per_label_p_r[0][2]
        Argument_against_precision = per_label_p_r[0][1]

        NoArgument_recall = per_label_p_r[1][0]
        Argument_for_recall = per_label_p_r[1][2]
        Argument_against_recall = per_label_p_r[1][1]
        # Print the precision and recall, among other metrics
        print(classification_report(labels, predictions, digits=3))

        with open(str(output_dir)+"/results.txt", "a+") as f:
            f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
            str(str(topic)),
            str(F1),
            str(NoArgument_precision),
            str(Argument_for_precision),
            str(Argument_against_precision),
            str(NoArgument_recall),
            str(Argument_for_recall),
            str(Argument_against_recall)
            ))
    else:
        per_label_p_r = precision_recall_fscore_support(labels,predictions, average=None,labels=[0,1])

        NoArgument_precision = per_label_p_r[0][0]
        Argument_precision = per_label_p_r[0][1]

        NoArgument_recall = per_label_p_r[1][0]
        Argument_recall = per_label_p_r[1][1]
        # Print the precision and recall, among other metrics
        print(classification_report(labels, predictions, digits=3))

        with open(str(output_dir)+"/results.txt", "a+") as f:
            f.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (
            str(str(topic)),
            str(F1),
            str(NoArgument_precision),
            str(Argument_precision),
            str(NoArgument_recall),
            str(Argument_recall),
            ))

    # Print the confusion matrix
    print(confusion_matrix(labels, predictions))

    return F1

UKP_eval(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
