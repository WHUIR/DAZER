
def get_word2id(word2id_path):
    word2id = {}
    with open(word2id_path,'r',encoding='gbk') as f:
        for line in f:
            w,id = line.strip().split(' ')
            word2id[w] = int(id)
    return word2id

def get_labels(label_dict_path,word2id_path):
    #use the label-dict file and word2id file to get label_dict, reverse_label_dict and label_list
    #which is useful in our DAZER model
    label_dict = {}
    reverse_label_dict = {}
    label_list = []
    word2id = get_word2id(word2id_path)
    with open(label_dict_path,'r') as f:
        for line in f:
            c_name,words = line.strip().split('/')
            ids = [word2id[w] for w in words.split(' ')]
            label_dict[c_name] = ids
            label_list.append(c_name)
            ids_str = ','.join([str(x) for x in ids])
            reverse_label_dict[ids_str] = c_name
    return label_dict, reverse_label_dict, label_list

def get_label_index(label_list, zsl_num,zsl_type):
    #get the index of zeroshot label
    #below is the experiments setting of 20NG in our ACL paper, you should change them in your own dataset
    
    #e.g., zeroshot_labels_1[0] = [['sci.space'],['comp.graphics']]
    #it means we use label "sci.space" for zeroshot experiments
    #and randomly pick label 'comp.graphics' to prevent overfitting
    #please refer to the "Evaluation protocol" part of our paper
    
    zeroshot_labels_1 = [
                 [['sci.space'],['comp.graphics']],
                 [['rec.sport.baseball'],['talk.politics.misc']],
                 [['sci.med'],['rec.autos']],
                 [['comp.sys.ibm.pc.hardware'],['rec.sport.hockey']],
                 ]

    zeroshot_labels_2= [
                    [['sci.med','sci.space'],['talk.politics.guns']],
                    [['alt.atheism','sci.electronics'],['comp.sys.ibm.pc.hardware']],
                    [['soc.religion.christian','talk.politics.mideast'],['rec.sport.baseball']],
                    [['rec.sport.baseball','rec.sport.hockey'],['comp.sys.mac.hardware']]
                    ]

    zeroshot_labels_3 = [
                     [['comp.sys.ibm.pc.hardware','comp.windows.x','sci.electronics'],['talk.politics.mideast']],
                    ]

    zeroshot_labels = [zeroshot_labels_1,zeroshot_labels_2,zeroshot_labels_3]

    z_labels = zeroshot_labels[zsl_num-1][zsl_type-1][0] + zeroshot_labels[zsl_num-1][zsl_type-1][1] 
    label_test = []
    for _l in label_list:
        if _l not in z_labels:
            label_test.append(_l)
    indexs = list(range(len(label_test)))
    zip_label_index = zip(label_test, indexs)
    return dict(list(zip_label_index))



