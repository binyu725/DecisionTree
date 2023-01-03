import sys, math, csv

# TODO: laplace smoothing

def populate_emails(file):
    emails = []
    with open(file,'r') as f:
        for line in f.readlines():
            words = line.split(" ")
            email = (words[0], words[1], dict()) # (id, label, {words: freq})
            for i in range(2, len(words), 2):
                email[2][words[i]] = int(words[i + 1])
            emails.append(email)
    return emails

def main():

    if(len(sys.argv) != 7):
        print('Invalid input, follow the following format:')
        print('Usage: python q2_classifier.py -f1 <train_dataset> -f2 <test_dataset> -o <output_file>')
        return

    f_train = sys.argv[2]
    f_test = sys.argv[4]
    output = sys.argv[6]

    emails = populate_emails(f_train)
    # count spams and hams so we can get P(w|spam), P(w|ham)
    spam_dict, ham_dict = dict(), dict()
    total_spam = 0
    for (id, label, freq_map) in emails:
        # add to ham/spam dictionary.
        # keep # of occurence of spam and ham for future computation
        if label == 'spam':
            for k in freq_map:
                spam_dict[k] = spam_dict.get(k, 0) + freq_map.get(k, 0)
            total_spam += 1
        else: 
            for k in freq_map:
                ham_dict[k] = ham_dict.get(k, 0) + freq_map.get(k, 0)
    total_ham = len(emails) - total_spam
    # Prior, P(Y)
    prior_spam = total_spam / len(emails)
    prior_ham = 1 - prior_spam
    ###---------------------------Naive Bayes Classifier------------------------###
    output_list = []
    emails = populate_emails(f_test)

    accurate_prediction = 0
    alpha = 1
    for email in emails:
        log_total_spam = 0
        log_total_ham = 0
        # apply Naive bayes with log, namely, P(Y|x1,x2,..) = P(Y)*Σlog(P(xi|Y))
        # Also, apply laplace smoothing on conditionals
        for word in email[2]:
            # P(W | Y=spam)
            if word in spam_dict:
                # number of word in spam emails / total # of spams
                p_word_given_spam = math.log((spam_dict[word] + alpha )/ (total_spam + len(spam_dict) * alpha) )
            else:
                p_word_given_spam = math.log(alpha / (total_spam + len(spam_dict) * alpha))
            log_total_spam += p_word_given_spam
             
            # P(W | Y=ham)
            if word in ham_dict:
                # number of word in ham emails / total # of ham
                p_word_given_ham = math.log((ham_dict[word] + alpha) / (total_ham + len(ham_dict) * alpha))                
            else:
                p_word_given_ham = math.log(alpha / (total_ham + len(ham_dict) * alpha))
            log_total_ham += p_word_given_ham
            
        # P(Y=spam|W) = P(Y=spam) * P(W | Y=spam)
        # P(Y=ham|W) = P(Y=ham) * P(W | Y=ham)
        posterior_spam = prior_spam * log_total_spam
        posterior_ham = prior_ham * log_total_ham
        
        # Accuracy and output file handling
        result = None
        if posterior_spam > posterior_ham:
            result = 'spam'
        else:
            result = 'ham'
        output_list.append([email[0], result])
        if (email[1] == result):
            accurate_prediction += 1

    accuracy = accurate_prediction / len(emails)
    print(f'Accuracy: {accuracy}')

    with open(output + ".csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(output_list)

main()