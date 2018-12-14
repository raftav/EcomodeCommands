import numpy as np
import glob
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import tensorflow as tf


#################################
# Serializer
#################################
def serialize_sequence(audio_sequence,labels):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(audio_sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)

    # Feature lists for the two sequential features of our example
    fl_audio_feat = ex.feature_lists.feature_list["audio_feat"]
    fl_audio_labels = ex.feature_lists.feature_list["audio_labels"]
    print('label from serializing = ',label)
    print('')
    fl_audio_labels.feature.add().float_list.value.append(label)

    for audio_feat in audio_sequence:
        fl_audio_feat.feature.add().float_list.value.extend(audio_feat)    

    return ex

commands_dict={}

with open('command_mapping.txt','r') as fp:

    row=fp.readline()
    while row:
        cmd_index,cmd_string,cmd_label = row.split('\t')
        commands_dict[cmd_index]=cmd_label.replace('\n','')
        row = fp.readline()


val_files_augment = glob.glob('/home/local/IIT/rtavarone/Data/PreProcessEcomodeItalian/NO_CLUTCH/VAL_FEAT/*.dat')
filelist=open('val_files_list.txt','w')
for index,file in enumerate(val_files_augment):
    print('serializing VAL file {} of {}'.format(index,len(val_files_augment)))
    filelist.write(file+'\t'+str(index)+'\n')

    features = np.genfromtxt(file,dtype=float)

    print('feature shape = ',features.shape)

    for key,value in commands_dict.items():
        if key in file:
            label = float(value)

    filename='VAL_FEAT_TFRECORDS/sequence_full_{:04d}.tfrecords'.format(index)
    fp = open(filename,'w')
    writer = tf.python_io.TFRecordWriter(fp.name)
    serialized_sentence = serialize_sequence(features,label)

    # write to tfrecord
    writer.write(serialized_sentence.SerializeToString())
    writer.close()
    fp.close()
filelist.close()


test_files = glob.glob('/home/local/IIT/rtavarone/Data/PreProcessEcomodeItalian/NO_CLUTCH/TEST_FEAT_IREC/*.dat')
filelist=open('test_irec_files_list.txt','w')
for index,file in enumerate(test_files):
    print('serializing TEST file {} of {}'.format(index,len(test_files)))
    filelist.write(file+'\t'+str(index)+'\n')

    features = np.genfromtxt(file,dtype=float)


    for key,value in commands_dict.items():
        if key in os.path.basename(file):
            label = float(value)
            print('filename[{}] key [{}] label[{}]'.format(os.path.basename(file),key,label))

    filename='TEST_FEAT_IREC_TFRECORDS/sequence_full_{:04d}.tfrecords'.format(index)
    fp = open(filename,'w')
    writer = tf.python_io.TFRecordWriter(fp.name)
    serialized_sentence = serialize_sequence(features,label)

    # write to tfrecord
    writer.write(serialized_sentence.SerializeToString())
    writer.close()
    fp.close()
filelist.close()


val_files = glob.glob('/home/local/IIT/rtavarone/Data/PreProcessEcomodeItalian/NO_CLUTCH/VAL_FEAT_IREC/*.dat')
filelist=open('val_irec_files_list.txt','w')
for index,file in enumerate(val_files):
    print('serializing VAL file {} of {}'.format(index,len(val_files)))
    filelist.write(file+'\t'+str(index)+'\n')

    features = np.genfromtxt(file,dtype=float)


    for key,value in commands_dict.items():
        if key in os.path.basename(file):
            label = float(value)
            print('filename[{}] key [{}] label[{}]'.format(os.path.basename(file),key,label))

    filename='VAL_FEAT_IREC_TFRECORDS/sequence_full_{:04d}.tfrecords'.format(index)
    fp = open(filename,'w')
    writer = tf.python_io.TFRecordWriter(fp.name)
    serialized_sentence = serialize_sequence(features,label)

    # write to tfrecord
    writer.write(serialized_sentence.SerializeToString())
    writer.close()
    fp.close()
filelist.close()


train_files = glob.glob('/home/local/IIT/rtavarone/Data/PreProcessEcomodeItalian/NO_CLUTCH/TRAIN_FEAT_IREC_AUGMENT/*.dat')
filelist=open('train_irec_files_augment_list.txt','w')
for index,file in enumerate(train_files):
    print('serializing TRAIN file {} of {}'.format(index,len(train_files)))
    filelist.write(file+'\t'+str(index)+'\n')

    features = np.genfromtxt(file,dtype=float)

    for key,value in commands_dict.items():
        if key in os.path.basename(file):
            label = float(value)
            print('filename[{}] key [{}] label[{}]'.format(os.path.basename(file),key,label))

    filename='TRAIN_FEAT_IREC_AUGMENT_TFRECORDS/sequence_full_{:04d}.tfrecords'.format(index)
    fp = open(filename,'w')
    writer = tf.python_io.TFRecordWriter(fp.name)
    serialized_sentence = serialize_sequence(features,label)

    # write to tfrecord
    writer.write(serialized_sentence.SerializeToString())
    writer.close()
    fp.close()
filelist.close()