For Q1, to run code:

python q1_classifier.py -p <pvalue> -f1 <train_dataset> -f2 <test_dataset> -o <output_file> -t <decision_tree>

where:
<train_dataset> and <test_dataset> are .csv files
<output_file> is a csv file containing the predicted labels for the test dataset (same format as the original testlabs.csv)
<decision_tree> is your decision tree file name
<pvalue> is the p-value threshold as described above

In addition, as the training labels file and test labels file are not mentioned in instruction, two files must be named to "trainlabs.csv" and "testlabs.csv" for running code.