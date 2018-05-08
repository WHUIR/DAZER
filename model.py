import tensorflow as tf
import numpy as np
import time
import get_label

import sys
import argparse
from traitlets.config.loader import PyFileConfigLoader
from traitlets.config import Configurable
from traitlets import (
    Int,
    Float,
    Bool,
    Unicode,
)

class DataGenerator(Configurable):
    #params for data generator
    max_q_len = Int(10, help='max q len').tag(config=True)
    max_d_len = Int(500, help='max document len').tag(config=True)
    q_name = Unicode('q')
    d_name = Unicode('d')
    q_str_name = Unicode('q_str')
    q_lens_name = Unicode('q_lens')
    aux_d_name = Unicode('d_aux')
    vocabulary_size = Int(2000000).tag(config=True)

    def __init__(self, **kwargs):
        #init the data generator
        super(DataGenerator, self).__init__(**kwargs)
        print ("generator's vocabulary size: ", self.vocabulary_size)

    def pairwise_reader(self, pair_stream, batch_size, with_idf=False):
        #generate the batch of x,y in training time
        l_q = []
        l_q_str = []
        l_d = []
        l_d_aux = []
        l_y = []
        l_q_lens = []
        for line in pair_stream:
            cols = line.strip().split('\t')
            y = float(1.0)
            l_q_str.append(cols[0])
            q = np.array([int(t) for t in cols[0].split(',') if int(t) < self.vocabulary_size])
            t1 = np.array([int(t) for t in cols[1].split(',') if int(t) < self.vocabulary_size])
            t2 = np.array([int(t) for t in cols[2].split(',') if int(t) < self.vocabulary_size])

            #padding
            v_q = np.zeros(self.max_q_len)
            v_d = np.zeros(self.max_d_len)
            v_d_aux = np.zeros(self.max_d_len)

            v_q[:min(q.shape[0], self.max_q_len)] = q[:min(q.shape[0], self.max_q_len)]
            v_d[:min(t1.shape[0], self.max_d_len)] = t1[:min(t1.shape[0], self.max_d_len)]
            v_d_aux[:min(t2.shape[0], self.max_d_len)] = t2[:min(t2.shape[0], self.max_d_len)]

            l_q.append(v_q)
            l_d.append(v_d)
            l_d_aux.append(v_d_aux)
            l_y.append(y)
            l_q_lens.append(len(q))

            if len(l_q) >= batch_size:
                Q = np.array(l_q,  dtype=int,)
                D = np.array(l_d,  dtype=int,)
                D_aux = np.array(l_d_aux, dtype=int,)
                Q_lens = np.array(l_q_lens, dtype=int,)
                Y = np.array(l_y,  dtype=int,)
                X = {self.q_name: Q, self.d_name: D, self.aux_d_name: D_aux, self.q_lens_name: Q_lens, self.q_str_name: l_q_str}
                yield X, Y
                l_q, l_d, l_d_aux, l_y, l_q_lens, l_ids, l_q_str = [], [], [], [], [], [], []
        if l_q:
            Q = np.array(l_q,  dtype=int,)
            D = np.array(l_d,  dtype=int,)
            D_aux = np.array(l_d_aux,  dtype=int,)
            Q_lens = np.array(l_q_lens, dtype=int,)
            Y = np.array(l_y,  dtype=int,)
            X = {self.q_name: Q, self.d_name: D, self.aux_d_name: D_aux, self.q_lens_name: Q_lens, self.q_str_name: l_q_str}
            yield X, Y

    def test_pairwise_reader(self, pair_stream, batch_size):
        #generate the batch of x,y in test time
        l_q = []
        l_q_lens = []
        l_d = []

        for line in pair_stream:
            cols = line.strip().split('\t')
            q = np.array([int(t) for t in cols[0].split(',') if int(t) < self.vocabulary_size])
            t = np.array([int(t) for t in cols[1].split(',') if int(t) < self.vocabulary_size])

            v_q = np.zeros(self.max_q_len)
            v_d = np.zeros(self.max_d_len)

            v_q[:min(q.shape[0], self.max_q_len)] = q[:min(q.shape[0], self.max_q_len)]
            v_d[:min(t.shape[0], self.max_d_len)] = t[:min(t.shape[0], self.max_d_len)]

            l_q.append(v_q)
            l_d.append(v_d)
            l_q_lens.append(len(q))

            if len(l_q) >= batch_size:
                Q = np.array(l_q,  dtype=int,)
                D = np.array(l_d,  dtype=int,)
                Q_lens = np.array(l_q_lens, dtype=int,)
                X = {self.q_name: Q, self.d_name: D, self.q_lens_name: Q_lens}
                yield X
                l_q, l_d, l_q_lens = [], [], []
        if l_q:
            Q = np.array(l_q,  dtype=int,)
            D = np.array(l_d,  dtype=int,)
            Q_lens = np.array(l_q_lens, dtype=int,)
            X = {self.q_name: Q, self.d_name: D, self.q_lens_name: Q_lens}
            yield X

class BaseNN(Configurable):
    #params of base deeprank model
    max_q_len = Int(10, help='max q len').tag(config=True)
    max_d_len = Int(50, help='max document len').tag(config=True)
    batch_size = Int(16, help="minibatch size").tag(config=True)
    max_epochs = Float(10, help="maximum number of epochs").tag(config=True)
    eval_frequency = Int(10000, help="print out minibatch every * epoches").tag(config=True)
    checkpoint_steps = Int(10000, help="store trained model every * epoches").tag(config=True)

    def __init__(self, **kwargs):
        super(BaseNN, self).__init__(**kwargs)
        # generator
        self.data_generator = DataGenerator(config=self.config)
        self.val_data_generator = DataGenerator(config=self.config)   #validation in training stage is full test data in 20ng
        self.test_data_generator = DataGenerator(config=self.config)  #test is zeros shot test data in 20ng (delete docs of zero shot label)

    @staticmethod
    def weight_variable(shape,name):
        tmp = np.sqrt(3.0) / np.sqrt(shape[0] + shape[1])
        initial = tf.random_uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial_value=initial,name=name)

    def gen_query_mask(self, Q):
        mask = np.zeros((self.batch_size, self.max_q_len))
        for b in range(len(Q)):
            for q in range(len(Q[b])):
                if Q[b][q] > 0:
                    mask[b][q] = 1

        return mask

    def gen_doc_mask(self, D):
        mask = np.zeros((self.batch_size, self.max_d_len))
        for b in range(len(D)):
            for q in range(len(D[b])):
                if D[b][q] > 0:
                    mask[b][q] = 1

        return mask

class DAZER(BaseNN):
    #params of zeroshot document filtering model
    embedding_size = Int(300, help="embedding dimension").tag(config=True)
    vocabulary_size = Int(2000000, help="vocabulary size").tag(config=True)
    kernal_width = Int(5, help='kernal width').tag(config=True)
    kernal_num = Int(50, help='number of kernal').tag(config=True)
    regular_term = Float(0.01, help='param for controlling wight of L2 loss').tag(config=True)
    maxpooling_num = Int(3, help='number of k-maxpooling').tag(config=True)
    decoder_mlp1_num = Int(75, help='number of hidden units of first mlp in relevance aggregation part').tag(config=True)
    decoder_mlp2_num = Int(1, help='number of hidden units of second mlp in relevance aggregation part').tag(config=True)
    emb_in = Unicode('None', help="initial embedding. Terms should be hashed to ids.").tag(config=True)
    model_learning_rate = Float(0.001, help="learning rate of model").tag(config=True)
    adv_learning_rate = Float(0.001, help='learning rate of adv classifier').tag(config=True)
    epsilon = Float(0.00001, help="Epsilon for Adam").tag(config=True)
    label_dict_path = Unicode('None', help='label dict path').tag(config=True)
    word2id_path = Unicode('None', help='word2id path').tag(config=True)
    train_class_num = Int(16, help='num of class in training data').tag(config=True)
    adv_term = Float(0.2, help='regular term of adversrial loss').tag(config=True)
    zsl_num = Int(1, help='num of zeroshot label').tag(config=True)
    zsl_type = Int(1, help='type of zeroshot label setting').tag(config=True)

    def __init__(self, **kwargs):
        #init the DAZER model
        super(DAZER, self).__init__(**kwargs)
        print ("trying to load initial embeddings from:  ", self.emb_in)
        if self.emb_in != 'None':
            self.emb = self.load_word2vec(self.emb_in)
            self.embeddings = tf.Variable(tf.constant(self.emb, dtype='float32', shape=[self.vocabulary_size + 1, self.embedding_size]),trainable=False)
            print ("Initialized embeddings with {0}".format(self.emb_in))
        else:
            self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size + 1, self.embedding_size], -1.0, 1.0))

        #variables of the DAZER model
        self.query_gate_weight = BaseNN.weight_variable((self.embedding_size, self.kernal_num),'gate_weight')
        self.query_gate_bias = tf.Variable(initial_value=tf.zeros((self.kernal_num)),name='gate_bias')
        self.adv_weight = BaseNN.weight_variable((self.decoder_mlp1_num,self.train_class_num),name='adv_weight')
        self.adv_bias = tf.Variable(initial_value=tf.zeros((1,self.train_class_num)),name='adv_bias')
        #get the label information to help adversarial learning
        self.label_dict, self.reverse_label_dict, self.label_list = get_label.get_labels(self.label_dict_path, self.word2id_path)
        self.label_index_dict = get_label.get_label_index(self.label_list, self.zsl_num, self.zsl_type)

    def load_word2vec(self, emb_file_path):
        emb = np.zeros((self.vocabulary_size + 1, self.embedding_size))
        nlines = 0
        with open(emb_file_path) as f:
            for line in f:
                nlines += 1
                if nlines == 1:
                    continue
                items = line.split()
                tid = int(items[0])
                if tid > self.vocabulary_size:
                    print (tid)
                    continue
                vec = np.array([float(t) for t in items[1:]])
                emb[tid, :] = vec
                if nlines % 20000 == 0:
                    print ("load {0} vectors...".format(nlines))
        return emb

    def gen_adv_query_mask(self, q_ids):
        q_mask = np.zeros((self.batch_size, self.train_class_num))
        for batch_num, b_q_id in enumerate(q_ids):
            c_name = self.reverse_label_dict[b_q_id]
            c_index = self.label_index_dict[c_name]
            q_mask[batch_num][c_index] = 1
        return q_mask

    def get_class_gate(self,class_vec, emb_d):
        '''
        compute the gate in kernal space
        :param class_vec: avg emb of seed words
        :param emb_d: emb of doc
        :return:the class gate [batchsize,d_len,kernal_num]
        '''
        gate1 = tf.expand_dims(tf.matmul(class_vec, self.query_gate_weight), axis=1)
        bias = tf.expand_dims(self.query_gate_bias,axis=0)
        gate = tf.add(gate1, bias)
        return tf.sigmoid(gate)

    def L2_model_loss(self):
        all_para = [v for v in tf.trainable_variables() if 'b' not in v.name and 'adv' not in v.name]
        loss = 0.
        for each in all_para:
            loss += tf.nn.l2_loss(each)
        return loss

    def L2_adv_loss(self):
        all_para = [v for v in tf.trainable_variables() if 'b' not in v.name and 'adv' in v.name]
        loss = 0.
        for each in all_para:
            loss += tf.nn.l2_loss(each)
        return loss

    def train(self, train_pair_file_path, val_pair_file_path, checkpoint_dir, load_model=False):

        input_q = tf.placeholder(tf.int32, shape=[self.batch_size,self.max_q_len])
        input_pos_d = tf.placeholder(tf.int32, shape=[self.batch_size,self.max_d_len])
        input_neg_d = tf.placeholder(tf.int32, shape=[self.batch_size,self.max_d_len])
        q_lens = tf.placeholder(tf.float32, shape=[self.batch_size,])
        q_mask = tf.placeholder(tf.float32, shape=[self.batch_size,self.max_q_len])
        pos_d_mask = tf.placeholder(tf.float32, shape=[self.batch_size,self.max_d_len])
        neg_d_mask = tf.placeholder(tf.float32, shape=[self.batch_size,self.max_d_len])
        input_q_index = tf.placeholder(tf.int32, shape=[self.batch_size,self.train_class_num])

        emb_q = tf.nn.embedding_lookup(self.embeddings,input_q)
        class_vec_sum = tf.reduce_sum(
            tf.multiply(emb_q,tf.expand_dims(q_mask,axis=-1)),
            axis=1
        )

        #get class vec
        class_vec = tf.div(class_vec_sum,tf.expand_dims(q_lens,-1))
        emb_pos_d = tf.nn.embedding_lookup(self.embeddings,input_pos_d)
        emb_neg_d = tf.nn.embedding_lookup(self.embeddings,input_neg_d)

        #get query gate
        pos_query_gate = self.get_class_gate(class_vec, emb_pos_d)
        neg_query_gate = self.get_class_gate(class_vec, emb_neg_d)

        # CNN for document
        pos_mult_info = tf.multiply(tf.expand_dims(class_vec, axis=1), emb_pos_d)
        pos_sub_info = tf.expand_dims(class_vec,axis=1) - emb_pos_d
        pos_conv_input = tf.concat([emb_pos_d,pos_mult_info,pos_sub_info], axis=-1)
        
        neg_mult_info = tf.multiply(tf.expand_dims(class_vec, axis=1), emb_neg_d)
        neg_sub_info = tf.expand_dims(class_vec,axis=1) - emb_neg_d
        neg_conv_input = tf.concat([emb_neg_d,neg_mult_info,neg_sub_info], axis=-1)


        #in fact that's 1D conv, but we implement it by conv2d
        pos_conv = tf.layers.conv2d(
            inputs = tf.expand_dims(pos_conv_input,axis=-1),
            filters = self.kernal_num,
            kernel_size=[self.kernal_width,self.embedding_size*3],
            strides = [1,self.embedding_size*3],
            padding = 'SAME',
            trainable = True,
            name='doc_conv'
        )

        neg_conv = tf.layers.conv2d(
            inputs = tf.expand_dims(neg_conv_input,axis=-1),
            filters = self.kernal_num,
            kernel_size=[self.kernal_width,self.embedding_size*3],
            strides = [1,self.embedding_size*3],
            padding = 'SAME',
            trainable = True,
            name='doc_conv',
            reuse=True
        )
        #shape=[batch,max_dlen,1,kernal_num]
        #reshape to [batch,max_dlen,kernal_num]
        rs_pos_conv = tf.squeeze(pos_conv)
        rs_neg_conv = tf.squeeze(neg_conv)

        #query_gate elment-wise multiply rs_pos_conv
        pos_gate_conv = tf.multiply(pos_query_gate, rs_pos_conv)
        neg_gate_conv = tf.multiply(neg_query_gate, rs_neg_conv)

        #K-max_pooling
        #transpose to [batch,knum,dlen],then get max k in each kernal filter
        transpose_pos_gate_conv = tf.transpose(pos_gate_conv, perm=[0,2,1])
        transpose_neg_gate_conv = tf.transpose(neg_gate_conv, perm=[0,2,1])

        #shape = [batch,k_num,maxpolling_num]
        #the k-max pooling here is implemented by function top_k, so the relative position information is ignored
        pos_kmaxpooling,_ = tf.nn.top_k(
            input=transpose_pos_gate_conv,
            k=self.maxpooling_num,
        )
        neg_kmaxpooling,_ = tf.nn.top_k(
            input=transpose_neg_gate_conv,
            k=self.maxpooling_num,
        )

        pos_encoder = tf.reshape(pos_kmaxpooling, shape=(self.batch_size,-1))
        neg_encoder = tf.reshape(neg_kmaxpooling, shape=(self.batch_size,-1))

        pos_decoder_mlp1 = tf.layers.dense(
            inputs=pos_encoder,
            units=self.decoder_mlp1_num,
            activation=tf.nn.tanh,
            trainable=True,
            name='decoder_mlp1'
        )

        neg_decoder_mlp1 = tf.layers.dense(
            inputs=neg_encoder,
            units=self.decoder_mlp1_num,
            activation=tf.nn.tanh,
            trainable=True,
            name='decoder_mlp1',
            reuse=True
        )

        pos_decoder_mlp2 = tf.layers.dense(
            inputs=pos_decoder_mlp1,
            units=self.decoder_mlp2_num,
            activation=tf.nn.tanh,
            trainable=True,
            name='decoder_mlp2'
        )

        neg_decoder_mlp2 = tf.layers.dense(
            inputs=neg_decoder_mlp1,
            units=self.decoder_mlp2_num,
            activation=tf.nn.tanh,
            trainable=True,
            name='decoder_mlp2',
            reuse=True
        )

        score_pos = pos_decoder_mlp2
        score_neg = neg_decoder_mlp2

        hinge_loss = tf.reduce_mean(tf.maximum(0.0, 1 - score_pos + score_neg))
        adv_prob = tf.nn.softmax(tf.add(tf.matmul(pos_decoder_mlp1, self.adv_weight), self.adv_bias))
        log_adv_prob = tf.log(adv_prob)
        adv_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(log_adv_prob, tf.cast(input_q_index,tf.float32)), axis=1, keep_dims=True))
        L2_adv_loss = self.regular_term*self.L2_adv_loss()

        #to apply GRL, we use two seperate optimizers for adversarial classifier and the rest part of DAZER
        #optimizer for adversarial classifier
        adv_var_list = [v for v in tf.trainable_variables() if 'adv' in v.name]
        adv_opt = tf.train.AdamOptimizer(learning_rate=self.adv_learning_rate, epsilon=self.epsilon).minimize(loss=(-1 * adv_loss + L2_adv_loss), var_list=adv_var_list)

        #optimizer for rest part of DAZER model
        L2_model_loss = self.regular_term*self.L2_model_loss()
        model_var_list = [v for v in tf.trainable_variables() if 'adv' not in v.name]
        loss = hinge_loss + L2_model_loss + (adv_loss * self.adv_term)
        model_opt = tf.train.AdamOptimizer(learning_rate=self.model_learning_rate, epsilon=self.epsilon).minimize(loss = loss, var_list = model_var_list)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        val_results = []
        save_num = 0
        save_var = [v for v in tf.trainable_variables()]

        # Create a local session to run the training.
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver(max_to_keep=50,var_list=save_var)
            start_time = time.time()
            if not load_model:
                print ("Initializing a new model...")
                init = tf.global_variables_initializer()
                sess.run(init)
                print('New model initialized!')
            else:
                #to load trained model, and keep training
                #remember to change the name of ckpt file
                init = tf.global_variables_initializer()
                sess.run(init)
                saver.restore(sess, checkpoint_dir+'/zsl25.ckpt')
                print ("model loaded!")

            # Loop through training steps.
            step = 0
            loss_list = []
            for epoch in range(int(self.max_epochs)):
                epoch_val_loss = 0
                epoch_loss = 0
                epoch_hinge_loss = 0.
                epoch_adv_loss = 0
                epoch_s = time.time()
                pair_stream = open(train_pair_file_path)

                for BATCH in self.data_generator.pairwise_reader(pair_stream, self.batch_size):
                    step += 1
                    X, Y = BATCH
                    query = X[u'q']
                    str_query = X[u'q_str']
                    q_index = self.gen_adv_query_mask(str_query)
                    pos_doc = X[u'd']
                    neg_doc = X[u'd_aux']
                    train_q_lens = X[u'q_lens']
                    M_query = self.gen_query_mask(query)
                    M_pos = self.gen_doc_mask(pos_doc)
                    M_neg = self.gen_doc_mask(neg_doc)

                    if X[u'q_lens'].shape[0] != self.batch_size:
                        continue
                    train_feed_dict = {input_q:query,
                                       input_pos_d:pos_doc,
                                       q_lens:train_q_lens,
                                       input_neg_d:neg_doc,
                                       q_mask:M_query,
                                       pos_d_mask:M_pos,
                                       neg_d_mask:M_neg,
                                       input_q_index: q_index}

                    _1,l,hinge_l,_2,adv_l  = sess.run([model_opt,loss,hinge_loss,adv_opt,adv_loss], feed_dict=train_feed_dict)
                    epoch_loss += l
                    epoch_hinge_loss += hinge_l
                    epoch_adv_loss += adv_l

                if (epoch + 1) % self.eval_frequency == 0:
                    #after eval_frequency epochs we run model on val dataset
                    val_start = time.time()
                    val_pair_stream = open(val_pair_file_path)
                    for BATCH in self.val_data_generator.pairwise_reader(val_pair_stream, self.batch_size):
                        X_val,Y_val = BATCH
                        query = X_val[u'q']
                        pos_doc = X_val[u'd']
                        neg_doc = X_val[u'd_aux']
                        val_q_lens = X_val[u'q_lens']
                        M_query = self.gen_query_mask(query)
                        M_pos = self.gen_doc_mask(pos_doc)
                        M_neg = self.gen_doc_mask(neg_doc)
                        if X_val[u'q'].shape[0] != self.batch_size:
                            continue
                        train_feed_dict = {input_q:query,
                                           input_pos_d:pos_doc,
                                           input_neg_d:neg_doc,
                                           q_lens:val_q_lens,
                                           q_mask:M_query,
                                           pos_d_mask:M_pos,
                                           neg_d_mask:M_neg}

                        # Run the graph and fetch some of the nodes.
                        v_loss = sess.run(hinge_loss, feed_dict=train_feed_dict)
                        epoch_val_loss += v_loss
                        val_results.append(epoch_val_loss)

                    val_end = time.time()
                    print('---Validation:epoch %d, %.1f ms , val_loss are %f' % (epoch+1,val_end-val_start,epoch_val_loss))
                    sys.stdout.flush()
                loss_list.append(epoch_loss)
                epoch_e = time.time()
                print('---Train:%d epoches cost %f seconds, hinge cost = %f  model cost = %f, adv cost = %f...'%(epoch+1,epoch_e-epoch_s,epoch_hinge_loss, epoch_loss,epoch_adv_loss))
                # save model after checkpoint_steps epochs
                if (epoch+1)%self.checkpoint_steps == 0:
                    save_num += 1
                    saver.save(sess, checkpoint_dir + 'zsl'+str(epoch+1)+'.ckpt')
                pair_stream.close()

            with open('save_training_loss.txt','w') as f:
                for index,_loss in enumerate(loss_list):
                    f.write('epoch'+str(index+1)+', loss:'+str(_loss)+'\n')

            with open('save_val_cost.txt','w') as f:
                for index, v_l in enumerate(val_results):
                    f.write('epoch'+str((index+1)*self.eval_frequency)+' val loss:'+str(v_l)+'\n')

            # end training
            end_time = time.time()
            print('All costs %f seconds...'%(end_time-start_time))

    def test(self, test_point_file_path, test_size, output_file_path, checkpoint_dir=None, load_model=False):

        input_q = tf.placeholder(tf.int32, shape=[self.batch_size,self.max_q_len])
        input_pos_d = tf.placeholder(tf.int32, shape=[self.batch_size,self.max_d_len])
        q_lens = tf.placeholder(tf.float32, shape=[self.batch_size,])
        q_mask = tf.placeholder(tf.float32, shape=[self.batch_size,self.max_q_len])
        pos_d_mask = tf.placeholder(tf.float32, shape=[self.batch_size,self.max_d_len])

        emb_q = tf.nn.embedding_lookup(self.embeddings,input_q)
        class_vec_sum = tf.reduce_sum(
            tf.multiply(emb_q,tf.expand_dims(q_mask,axis=-1)),
            axis=1
        )

        class_vec = tf.div(class_vec_sum,tf.expand_dims(q_lens,axis=-1))
        emb_pos_d = tf.nn.embedding_lookup(self.embeddings,input_pos_d)

        #get query gate
        query_gate = self.get_class_gate(class_vec, emb_pos_d)
        pos_mult_info = tf.multiply(tf.expand_dims(class_vec, axis=1), emb_pos_d)
        pos_sub_info = tf.expand_dims(class_vec, axis=1) - emb_pos_d
        pos_conv_input = tf.concat([emb_pos_d,pos_mult_info, pos_sub_info], axis=-1)

        # CNN for document
        pos_conv = tf.layers.conv2d(
            inputs = tf.expand_dims(pos_conv_input,axis=-1),
            filters = self.kernal_num,
            kernel_size=[self.kernal_width,self.embedding_size*3],
            strides = [1,self.embedding_size*3],
            padding = 'SAME',
            trainable = True,
            name='doc_conv'
        )

        #shape=[batch,max_dlen,1,kernal_num]
        #reshape to [batch,max_dlen,kernal_num]
        rs_pos_conv = tf.squeeze(pos_conv)

        #query_gate elment-wise multiply rs_pos_conv
        #[batch,kernal_num] , [batch,max_dlen,kernal_num]
        pos_gate_conv = tf.multiply(query_gate, rs_pos_conv)

        #K-max_pooling
        #transpose to [batch,knum,dlen],then get max k in each kernal filter
        transpose_pos_gate_conv = tf.transpose(pos_gate_conv, perm=[0,2,1])

        #[batch,k_num,maxpolling_num]
        pos_kmaxpooling,_ = tf.nn.top_k(
            input=transpose_pos_gate_conv,
            k=self.maxpooling_num,
        )
        pos_encoder = tf.reshape(pos_kmaxpooling, shape=(self.batch_size,-1))

        pos_decoder_mlp1 = tf.layers.dense(
            inputs=pos_encoder,
            units=self.decoder_mlp1_num,
            activation=tf.nn.tanh,
            trainable=True,
            name='decoder_mlp1'
        )

        pos_decoder_mlp2 = tf.layers.dense(
            inputs=pos_decoder_mlp1,
            units=self.decoder_mlp2_num,
            activation=tf.nn.tanh,
            trainable=True,
            name='decoder_mlp2'
        )

        score_pos = pos_decoder_mlp2
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        save_var = [v for v in tf.trainable_variables()]
        # Create a local session to run the testing.
        for i in range(int(self.max_epochs/self.checkpoint_steps)):
            with tf.Session(config=config) as sess:
                test_point_stream = open(test_point_file_path)
                outfile = open(output_file_path+'-epoch'+str(self.checkpoint_steps*(i+1))+'.txt', 'w')
                saver = tf.train.Saver(var_list=save_var)

                if load_model:
                    p = checkpoint_dir + 'zsl'+str(self.checkpoint_steps*(i+1))+'.ckpt'
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    saver.restore(sess, p)
                    print ("data loaded!")
                else:
                    init = tf.global_variables_initializer()
                    sess.run(init)

                # Loop through training steps.
                for b in range(int(np.ceil(float(test_size)/self.batch_size))):
                    X = next(self.test_data_generator.test_pairwise_reader(test_point_stream, self.batch_size))
                    if(X[u'q'].shape[0] != self.batch_size):
                        continue
                    query = X[u'q']
                    pos_doc = X[u'd']
                    test_q_lens = X[u'q_lens']
                    M_query = self.gen_query_mask(query)
                    M_pos = self.gen_doc_mask(pos_doc)
                    test_feed_dict = {input_q: query,
                                       input_pos_d: pos_doc,
                                       q_lens: test_q_lens,
                                       q_mask: M_query,
                                      pos_d_mask: M_pos}

                    # Run the graph and fetch some of the nodes.
                    scores = sess.run(score_pos, feed_dict=test_feed_dict)

                    for score in scores:
                        outfile.write('{0}\n'.format(score[0]))

                outfile.close()
                test_point_stream.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file_path")

    parser.add_argument("--train", action='store_true')
    parser.add_argument("--train_file", '-f', help="train_pair_file_path")
    parser.add_argument("--validation_file", '-v', help="val_pair_file_path")
    parser.add_argument("--train_size", '-z', type=int, help="number of train samples")
    parser.add_argument("--load_model", '-l', action='store_true')

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_file")
    parser.add_argument("--test_size", type=int, default=0)
    parser.add_argument("--output_score_file", '-o')
    parser.add_argument("--emb_file_path", '-e')
    parser.add_argument("--checkpoint_dir", '-s', help="store data to here")

    args = parser.parse_args()

    conf = PyFileConfigLoader(args.config_file_path).load_config()

    if args.train:
        nn = DAZER(config=conf)
        nn.train(train_pair_file_path=args.train_file,
                 val_pair_file_path=args.validation_file,
                 checkpoint_dir=args.checkpoint_dir,
                 load_model=args.load_model)
    else:
        nn = DAZER(config=conf)
        nn.test(test_point_file_path=args.test_file,
                test_size=args.test_size,
                output_file_path=args.output_score_file,
                load_model=True,
                checkpoint_dir=args.checkpoint_dir)

