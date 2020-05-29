# -*- coding: utf-8 -*- 

import tensorflow as tf
import pandas as pd
import numpy as np
import sys,os

from flask import Flask
from flask_restful import Resource, Api
from flask_restful import reqparse

app = Flask(__name__)
api = Api(app)

def sigma(x):
    # sigmoid function
    # σ(x) = 1 / (1 + exp(-x))
    return 1. / (1. + tf.exp(-x))

def pred_to_list(pred):
    pred_list=[]
    for i in range(len(pred)):
        temp=[]
        temp.append(pred[i])
        pred_list.append(temp)
    return pred_list

def data_embedding(nb_classes,x_colum):
    
    X = tf.placeholder(tf.float32, [None, x_colum])
    y = tf.placeholder(tf.int32, [None, 1])

    target = tf.one_hot(y, nb_classes)
    target = tf.reshape(target, [-1, nb_classes])
    target = tf.cast(target, tf.float32)
    
    Y_one_hot = tf.one_hot(y, nb_classes)  
    Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
    
    return X, y, target,Y_one_hot

def layer_structed(X, y, target, nb_classes, x_colum):
    
    keep_prob = tf.placeholder(tf.float32)
    
    W1 = tf.get_variable("W1", shape=[x_colum, x_colum],initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([x_colum]), name='bias1')
    l1 = tf.sigmoid(tf.matmul(X, W1) + b1)
    l1 = tf.nn.dropout(l1, keep_prob=keep_prob)

    W2 = tf.get_variable("W2", shape=[x_colum, x_colum],initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([x_colum]), name='bias2')
    l2 = tf.sigmoid(tf.matmul(l1, W2) + b2)
    l2 = tf.nn.dropout(l2, keep_prob=keep_prob)

    W3 = tf.get_variable("W3", shape=[x_colum, x_colum],initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([x_colum]), name='bias3')
    l3 = tf.sigmoid(tf.matmul(l2, W3) + b3)
    l3 = tf.nn.dropout(l3, keep_prob=keep_prob)

    W4 = tf.get_variable("W4", shape=[x_colum, x_colum],initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([x_colum]), name='bias4')
    l4 = tf.sigmoid(tf.matmul(l3, W4) + b4)
    l4 = tf.nn.dropout(l2, keep_prob=keep_prob)

    W5 = tf.get_variable("W5", shape=[x_colum, nb_classes],initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([nb_classes]), name='bias5')
    #     y_pred = tf.sigmoid(tf.matmul(l4, W5) + b5)
    
    # Forward propagtion
    layer_5 = tf.matmul(X, W5) + b5
    y_pred = sigma(layer_5)
    
    return W5, b5, layer_5, y_pred, keep_prob

def pred_by_restore(checkpoint_path, W5, b5, X, temp, y):
    
    predict_list=[]
    
    #hypothesis
    hypothesis = tf.nn.sigmoid(tf.matmul(X, W5) + b5)
    
    #prediction
    prediction = tf.argmax(hypothesis, 1) 
    
    #sess
    sess = tf.Session()
    
    #restore
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

    pred = sess.run(prediction, feed_dict={ X : temp})
    pred_list = sess.run(hypothesis,  feed_dict={ X: temp, y: pred_to_list(pred)}).tolist()

    for i in range(len(pred_list)):
        temp=[]
        pred_list_sort, pred_list_index = sorted(pred_list[i],reverse=True),[]
        
        for j in range(len(pred_list[i])):
            pred_list_index.append(pred_list[i].index(pred_list_sort[j]))
            
        temp.append(pred_list_sort)
        temp.append(pred_list_index)
        predict_list.append(temp)
        
    return predict_list

def hyun2(model_0,model_1,model_2,model_3,model_4):
    model_list=[]
    for i in range(len(model_0)):
        model_list_temp=[]
        frist_intserrup=0
        for j0 in range(6):
            if model_0[i][1][j0] == 5 :
                frist_intserrup=j0
                break
            else :model_list_temp.append(model_0[i][1][j0])
        for j1 in range(6):
            if model_1[i][1][j1] == 5 :break
            else :model_list_temp.append(model_1[i][1][j1]+5)
        for j2 in range(6):
            if model_2[i][1][j2] == 5 :break
            else :model_list_temp.append(model_2[i][1][j2]+10)
        for j3 in range(6):
            if model_3[i][1][j3] == 5 :break
            else :model_list_temp.append(model_3[i][1][j3]+15)
        for j4 in range(6):
            if model_4[i][0][j4] < 0.5 :
                for j5 in range(6-len(model_list_temp)):
                    try :
                        model_list_temp.append(model_0[i][1][frist_intserrup+j5+1])
                    except IndexError:
                        pass
            else : model_list_temp.append(model_4[i][1][j4]+20)
        model_list_temp=model_list_temp[:6]
        model_list.append(model_list_temp)
        
    return model_list


def load_model(nb_classes=6,x_colum=1444,input_string=''):
    

    temp2 = input_string.split(",")
    temp3 = np.array(temp2)
    temp4 = temp3.astype(np.float32)
    df_input = (pd.DataFrame(temp4)).T
    
    #data_embedding
    X, y, target,Y_one_hot = data_embedding(nb_classes,x_colum)
    
    #layer_structed
    W5, b5, layer_5, y_pred, keep_prob = layer_structed(X, y, target, nb_classes, x_colum)
    
    model_0 = pred_by_restore('./model/model_0',W5, b5, X, df_input,y)
    model_1 = pred_by_restore('./model/model_1',W5, b5, X, df_input,y)
    model_2 = pred_by_restore('./model/model_2',W5, b5, X, df_input,y)
    model_3 = pred_by_restore('./model/model_3',W5, b5, X, df_input,y)
    model_4 = pred_by_restore('./model/model_4',W5, b5, X, df_input,y)
    
    model_list = hyun2(model_0,model_1,model_2,model_3,model_4)
    
    return(model_list[0])

class Predict_label(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('input', type=str)
        args = parser.parse_args()

        text = args['input']
        
        Predict_label=load_model(input_string=text)
        
        
        
        return {'Predict_label': str(Predict_label)}

api.add_resource(Predict_label, '/a')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

    
