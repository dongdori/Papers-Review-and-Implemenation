import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import transformer
from transformer import BertTokenizer

from config import config
from padding_func import create_masks

class Dataset(Sequence):
    def __init__(self, config, dataframe):
        self.src_maxlen = config['src_MAXLEN']
        self.tar_maxlen = config['tar_MAXLEN']
        self.batch_size = config['BATCH_SIZE']
        self.df = dataframe
        self.tz_eng = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tz_fch = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    
    def __len__(self):
        return len(self.df) // self.batch_size
    
    def __getitem__(self, index):
        # get id of source sentence
        src = np.array(self.df.iloc[index*self.batch_size:(index+1)*self.batch_size, 0].apply(self.tz_eng.encode_plus,
                                                                                                 max_length = self.src_maxlen,
                                                                                                 pad_to_max_length = True,
                                                                                                 return_attention_mask = False).apply(lambda x: x['input_ids']).tolist())
        # get id of target sentence
        tar = np.array(self.df.iloc[index*self.batch_size:(index+1)*self.batch_size, 1].apply(self.tz_fch.encode_plus,
                                                                                                 max_length = self.tar_maxlen,
                                                                                                 pad_to_max_length = True,
                                                                                                 return_attention_mask = False).apply(lambda x: x['input_ids']).tolist())
        
        # create mask
        src_pad, combined_mask, tar_pad = create_masks(src, tar)
        # defining target_id for teacher forcing
        def make_target_output(tar):
            return tf.Variable(tar[1:], dtype = tf.int32)
        def make_target_input(tar):
            return tf.Variable(tar[:-1], dtype = tf.int32)
        src_input = np.array(src)
        tar_output = tar[:,1:]
        tar_input = tar[:,:-1]
        
        return tf.Variable(src_input, dtype = tf.int32), tf.Variable(tar_input, dtype = tf.int32), tf.Variable(tar_output, dtype = tf.int32),src_pad, combined_mask, tar_pad
      
train_data = Dataset(config, tr)
val_data = Dataset(config, val)
        
