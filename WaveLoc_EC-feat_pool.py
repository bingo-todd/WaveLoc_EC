import numpy as np
import pickle
# import matplotlib
import matplotlib.pyplot as plt
import os
import configparser
import time
import tensorflow as tf
import gammatone.filters as gt_filters
from BasicTools.get_file_path import get_file_path
from BasicTools.Dataset import Dataset
from BasicTools.ProgressBar import ProgressBar
from BasicTools.Logger import Logger
from LocTools.add_log import add_log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# matplotlib.use('Agg')


class WaveLoc_NN(tf.keras.Model):
    def __init__(self, config):
        super(WaveLoc_NN, self).__init__()
        self.fs = np.int32(config['fs'])
        self.cf_low = np.int32(config['cf_low'])
        self.cf_high = np.int32(config['cf_high'])
        self.n_band = np.int32(config['n_band'])
        self.filter_len = np.int32(config['filter_len'])
        self.n_azi = np.int32(config['n_azi'])
        self.n_unit_fcn = np.int32(config['n_unit_fcn'])
        self.n_feat_filter = np.int32(config['n_feat_filter'])
        self.feat_filter_len = np.int32(config['feat_filter_len'])
        self.pool_len = np.int32(config['pool_len'])
        self.cnn_coef = tf.constant(np.float32(config['feat_filter_len']))
        self._build_model()

    def get_gtf_kernel(self):
        cfs = gt_filters.erb_space(self.cf_low, self.cf_high, self.n_band)
        sample_times = np.arange(0, self.filter_len, 1)/self.fs
        irs = np.zeros((self.filter_len, self.n_band), dtype=np.float32)
        EarQ = 9.26449
        minBW = 24.7
        order = 1
        N = 4
        for band_i in range(self.n_band):
            b = 1.019*((cfs[band_i]/EarQ)**order+minBW**order)**(1/order)
            numerator = np.multiply(sample_times**(N-1),
                                    np.cos(2*np.pi*cfs[band_i]*sample_times))
            denominator = np.exp(2*np.pi*b*sample_times)
            irs[:, band_i] = np.divide(numerator, denominator)
        gain = np.max(np.abs(np.fft.fft(irs, axis=0)), axis=0)
        irs_gain_norm = np.divide(np.flipud(irs), gain)
        kernel = np.concatenate((irs_gain_norm,
                                 np.zeros((self.filter_len, self.n_band))),
                                axis=0)
        self.cfs = cfs
        return kernel

    def _concat_layers(self, layer_all, x):
        if layer_all is None:
            return x
        else:
            input_tmp = x
            for layer in layer_all:
                output_tmp = layer(input_tmp)
                input_tmp = output_tmp
            return output_tmp

    def _max_normalization(self, x):
        amp_max = tf.reduce_max(
                    tf.reduce_max(
                        tf.reduce_max(
                            tf.abs(x),
                            axis=1, keepdims=True),
                        axis=2, keepdims=True),
                    axis=3, keepdims=True)
        return tf.divide(x, amp_max)

    def _band_max_normalization(self, x):
        amp_max = tf.reduce_max(
            tf.reduce_max(tf.abs(x), axis=1, keepdims=True),
            axis=2, keepdims=True)
        return tf.divide(x, amp_max)

    def _neg_log(self, x):
        return 1-self.cnn_coef*tf.math.log(1+x**2)

    def _build_model(self):
        tf.random.set_seed(1)
        kernel_initializer = tf.constant_initializer(self.get_gtf_kernel())
        gtf_kernel_len = 2*self.filter_len
        # Gammatome filter layer
        gt_layer = tf.keras.layers.Conv2D(
            filters=self.n_band,
            kernel_size=[gtf_kernel_len, 1],
            strides=[1, 1],
            padding='same',
            kernel_initializer=kernel_initializer,
            trainable=False,
            use_bias=False)
        gt_layer_norm = self._band_max_normalization
        gt_layer_all = [gt_layer, gt_layer_norm]

        # convolve layer
        band_layer_all = []
        for i in range(self.n_band):
            band_layer1 = tf.keras.layers.Conv2D(
                filters=self.n_feat_filter,
                kernel_size=[self.feat_filter_len, 2],
                strides=[np.int(self.feat_filter_len/2), 1],
                activation=self._neg_log)
            band_layer2 = tf.keras.layers.MaxPool2D(
                [self.pool_len, 1], [self.pool_len, 1])
            band_layer3 = tf.keras.layers.Flatten()
            band_layer4 = tf.keras.layers.Dense(
                units=128, activation=tf.nn.relu)
            band_layer5 = tf.keras.layers.Dense(
                units=32, activation=tf.nn.relu)
            band_layer_all.append(
                [band_layer1, band_layer2, band_layer3,
                 band_layer4, band_layer5])
        #
        loc_fcn_layer1 = tf.keras.layers.Dense(
            units=self.n_unit_fcn, activation=tf.nn.relu)
        loc_fcn_layer2 = tf.keras.layers.Dense(
            units=self.n_unit_fcn, activation=tf.nn.relu)
        loc_ouptut_layer = tf.keras.layers.Dense(
            units=self.n_azi, activation=tf.nn.softmax)
        loc_fcn_layer_all = [loc_fcn_layer1, loc_fcn_layer2, loc_ouptut_layer]

        self.gt_layer_all = gt_layer_all
        self.band_layer_all = band_layer_all
        self.loc_fcn_layer_all = loc_fcn_layer_all

    def random_run(self):
        x_tmp = np.random.rand(5, self.filter_len*2, 2, 1)
        self.call(x_tmp)

    def get_gtf(self):
        self.random_run()
        kernel = self.gt_layer_all[0].get_weights()[0]
        return kernel

    def get_feat_extractors_weights(self):
        self.random_run()
        weights = []
        for band_i in range(self.n_band):
            weights_band_i = []
            for layer in self.band_layer_all[band_i]:
                weights_band_i.append(layer.get_weights())
            weights.append(weights_band_i)
        return weights

    def get_feat_extractors_cnn_weights(self):
        self.random_run()
        weights = [self.band_layer_all[band_i][0].get_weights()
                   for band_i in range(self.n_band)]
        return weights

    def get_decision_device_weights(self):
        self.random_run()
        weights = []
        for layer in self.loc_fcn_layer_all:
            weights.append(layer.get_weights())
        return weights

    def set_feat_extractors_weights(self, weights):
        self.random_run()
        for band_i in range(self.n_band):
            for layer_i, layer in enumerate(self.band_layer_all[band_i]):
                layer.set_weights(weights[band_i][layer_i])

    def set_decision_device_weights(self, weights):
        self.random_run()
        for layer_i, layer in enumerate(self.loc_fcn_layer_all):
            layer.set_weights(weights[layer_i])

    def cal_feat(self, x):
        x_band_all = self.gt_layer_all[0](x)[:, self.filter_len:, :, :]
        gt_layer_output = self._concat_layers(
            self.gt_layer_all[1:], x_band_all)
        feat_bands = []
        for band_i in range(self.n_band):
            band_input = tf.expand_dims(
                gt_layer_output[:, :, :, band_i], axis=-1)
            feat = self._concat_layers(
                self.band_layer_all[band_i][:1], band_input)
            feat_bands.append(np.squeeze(np.asarray(feat)))
        return feat_bands

    def call(self, x):

        x_band_all = self.gt_layer_all[0](x)[:, self.filter_len:, :, :]
        gt_layer_output = self._concat_layers(self.gt_layer_all[1:],
                                              x_band_all)
        band_out_all = []
        for band_i in range(self.n_band):
            band_input = tf.expand_dims(
                gt_layer_output[:, :, :, band_i], axis=-1)
            band_output = self._concat_layers(
                self.band_layer_all[band_i], band_input)
            band_out_all.append(band_output)

        band_out_all_concat = tf.concat(band_out_all, axis=1)
        y_loc_est = self._concat_layers(
            self.loc_fcn_layer_all, band_out_all_concat)
        return y_loc_est


class Model(object):
    def __init__(self, file_reader, log_path, config_path=None,
                 test_mode=False):
        self.test_mode = test_mode

        # constant settings
        config = configparser.ConfigParser()
        config.read(config_path)
        self._load_cfg(config_path)
        self.epsilon = 1e-20
        self.file_reader = file_reader
        self.nn = WaveLoc_NN(config['model'])

        self.ckpt = tf.train.Checkpoint(model=self.nn)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                       self.model_dir,
                                                       max_to_keep=10)

        self.logger = Logger(log_path, True)

    def _load_cfg(self, config_path):
        if config_path is not None and os.path.exists(config_path):
            config = configparser.ConfigParser()
            config.read(config_path)
            # settings for model
            self.fs = np.int32(config['model']['fs'])
            self.n_band = np.int32(config['model']['n_band'])
            self.cf_low = np.int32(config['model']['cf_low'])
            self.cf_high = np.int32(config['model']['cf_high'])
            self.frame_len = np.int32(config['model']['frame_len'])
            self.shift_len = np.int32(config['model']['shift_len'])
            self.filter_len = np.int32(config['model']['filter_len'])
            self.n_azi = np.int32(config['model']['n_azi'])
            # settings for training
            self.model_dir = config['train']['model_dir']
            self.batch_size = np.int32(config['train']['batch_size'])
            self.init_lr = np.float32(config['train']['init_lr'])
            self.max_epoch = np.int32(config['train']['max_epoch'])
            self.is_print_log = config['train']['is_print_log'] == 'True'
            self.train_file_list = config['train']['train_file_list']
            self.valid_file_list = config['train']['valid_file_list']
        else:
            print(config_path)
            raise OSError

    def _cal_cross_entropy(self, y_est, y_gt):
        cross_entropy = -tf.reduce_mean(
                            tf.reduce_sum(
                                tf.multiply(
                                    y_gt, tf.math.log(y_est+1e-20)),
                                axis=1))
        return cross_entropy

    def _cal_mse(self, y_est, y_gt):
        mse = tf.reduce_mean(tf.reduce_sum((y_gt-y_est)**2, axis=1))
        return mse

    def _cal_loc_rmse(self, y_est, y_gt):
        azi_est = tf.argmax(y_est, axis=1)
        azi_gt = tf.argmax(y_est, axis=1)
        diff = tf.cast(azi_est - azi_gt, tf.float32)
        return tf.sqrt(tf.reduce_mean(diff**2))

    def _cal_cp(self, y_est, azi_gt):
        equality = tf.equal(tf.argmax(y_est, axis=1),
                            tf.argmax(y_est, axis=1))
        cp = tf.reduce_mean(tf.cast(equality, tf.float32))
        return cp

    def load_model(self, load_best=True):
        """load model"""
        if load_best:
            with open(f'{self.model_dir}/best_model', 'r') as tmp_file:
                model_path = tmp_file.readlines()[0].strip()
        else:
            model_path = self.ckpt_manager.latest_checkpoint
        self.ckpt.restore(model_path)
        print(f'load model from {model_path}')

    def _train_record_init(self, model_dir, is_load_model):
        if is_load_model:
            record_info = np.load(os.path.join(model_dir, 'train_record.npz'))
            loss_record = record_info['loss_record']
            lr_value = record_info['lr']
            cur_epoch = record_info['cur_epoch']
            best_epoch = record_info['best_epoch']
            min_loss = record_info['min_loss']
        else:
            loss_record = np.zeros(self.max_epoch)
            lr_value = self.init_lr
            min_loss = np.infty
            cur_epoch = -1
            best_epoch = 0
        return [loss_record, lr_value,
                cur_epoch, best_epoch, min_loss]

    def _get_file_path(self, set_dir):
        if isinstance(set_dir, list):
            dir_all = set_dir
        else:
            dir_all = [set_dir]
        file_paths = []
        for dir_tmp in dir_all:
            file_paths_tmp = get_file_path(dir_tmp, suffix='.wav',
                                           is_absolute=True)
            file_paths.extend(file_paths_tmp)
        if len(file_paths) < 1:
            raise Exception(f'empty dir: {set_dir}')
        if self.test_mode:
            file_paths = file_paths[:10]
        return file_paths

    def run_optimization(self, x, y, optimizer):
        with tf.GradientTape() as g:
            y_est = self.nn(x)
            loss = self._cal_cross_entropy(y_est, y)
        gradients = g.gradient(loss, self.nn.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.nn.trainable_variables))
        return loss

    def test(self, load_weights=False,
             feat_extractors_weights_path=None,
             decision_device_weights_path=None, **other_params):
        if load_weights:
            with open(feat_extractors_weights_path, 'rb') as weights_f:
                weights = pickle.load(weights_f)
                self.nn.set_feat_extractors_weights(weights)
            print('load feat_extractors_weights from '
                  + feat_extractors_weights_path)
            with open(decision_device_weights_path, 'rb') as weights_f:
                weights = pickle.load(weights_f)
                self.nn.set_decision_device_weights(weights)
            print('load decision_device_weights from  '
                  + decision_device_weights_path)

        with open(self.valid_file_list, 'r') as f:
            valid_file_paths = [item.strip() for item in f.readlines()]
        self.dataset_valid = Dataset(self.file_reader,
                                     valid_file_paths,
                                     self.batch_size*5,
                                     self.batch_size,
                                     [[self.frame_len, 2, 1],
                                      [self.n_azi]])
        loss = self.validate()
        print(loss)

    def train(self):

        is_load_model = os.path.exists(f'{self.model_dir}/train_record.npz')

        if is_load_model:
            self.load_model(load_best=False)

        [loss_record,
         lr, cur_epoch,
         best_epoch, min_loss] = self._train_record_init(self.model_dir,
                                                         is_load_model)

        with open(self.train_file_list, 'r') as f:
            train_file_paths = [item.strip() for item in f.readlines()]
        self.logger.info(f'{len(train_file_paths)} files in Train set')
        self.dataset_train = Dataset(self.file_reader,
                                     train_file_paths,
                                     self.batch_size*5,
                                     self.batch_size,
                                     [[self.frame_len, 2, 1],
                                      [self.n_azi]])

        with open(self.valid_file_list, 'r') as f:
            valid_file_paths = [item.strip() for item in f.readlines()]
        self.logger.info(f'{len(valid_file_paths)} files in Valid set')
        self.dataset_valid = Dataset(self.file_reader,
                                     valid_file_paths,
                                     self.batch_size*5,
                                     self.batch_size,
                                     [[self.frame_len, 2, 1],
                                      [self.n_azi]])

        print('start training')
        # pb = ProgressBar(self.max_epoch)
        # pb.value = cur_epoch
        for epoch in range(cur_epoch+1, self.max_epoch):
            # pb.update()
            t_start = time.time()
            optimizer = tf.optimizers.Adam(lr)
            self.dataset_train.reset()
            while not self.dataset_train.is_finish():
                x, y = self.dataset_train.next_batch()
                self.run_optimization(x, y, optimizer)
            # model test
            loss_record[epoch] = self.validate()
            # write to log
            iter_time = time.time()-t_start
            self.logger.info(
                ' '.join((f'epoch:{epoch}',
                          f'lr:{lr}',
                          f'time:{iter_time:.2f}\n')))
            self.logger.info(f'\t loss_loc:{loss_record[epoch]}')

            # save in each epoch in case of interruption
            cur_model_path = self.ckpt_manager.save()
            np.savez(os.path.join(self.model_dir, 'train_record'),
                     loss_record=loss_record,
                     lr=lr,
                     cur_epoch=epoch,
                     best_epoch=best_epoch,
                     min_loss=min_loss)

            # update min_loss
            if min_loss > loss_record[epoch]:
                self.logger.info('find new optimal\n')
                best_epoch = epoch
                min_loss = loss_record[epoch]
                with open(f'{self.model_dir}/best_model', 'w') as tmp_file:
                    tmp_file.write(cur_model_path)

            # early stop
            n_epoch_stop = 5
            if epoch-best_epoch > n_epoch_stop:
                print(epoch, best_epoch)
                print('early stop\n', min_loss)
                self.logger.info('early stop{}\n'.format(min_loss))
                break

            # learning rate decay
            n_epoch_decay = 2
            if epoch >= n_epoch_decay:  # no better performance in 2 epoches
                min_loss_local = np.min(
                    loss_record[epoch-n_epoch_decay+1:epoch+1])
                if loss_record[epoch-n_epoch_decay] < min_loss_local:
                    lr = lr*.5

        if True:
            fig, ax = plt.subplots(1, 1, sharex=True, tight_layout=True)
            ax.plot(loss_record, label='loc')
            ax.legend()
            ax.set_ylabel('cost')
            fig_path = os.path.join(self.model_dir, 'train_curve.png')
            plt.savefig(fig_path)

    def validate(self):
        loss = 0.
        n_sample = 0
        self.dataset_valid.reset()
        while not self.dataset_valid.is_finish():
            x, y = self.dataset_valid.next_batch()
            n_sample_tmp = x.shape[0]
            y_est = self.nn(x)
            # print(np.argmax(y, axis=1))
            # print(np.argmax(y_est, axis=1))
            loss_tmp = self._cal_cross_entropy(y_est, y)
            loss = loss + loss_tmp*n_sample_tmp
            n_sample = n_sample + n_sample_tmp
        loss = loss/n_sample
        return loss

    def estimate(self, x, batch_size=-1):
        if batch_size == -1:
            # feed all sampels to nn
            y_est = self.nn(x)
        else:
            n_sample = x.shape[0]
            y_est = np.zeros([n_sample, self.n_azi])
            for sample_i in range(0, n_sample, batch_size):
                batch_slice = slice(sample_i, sample_i+batch_size)
                y_est[batch_slice] = self.nn(x[batch_slice])
        return y_est

    def localize(self, x, batch_size=-1):
        x = np.expand_dims(x, axis=-1)
        return self.estimate(x, batch_size)

    def evaluate(self, record_paths, log_path):
        """
        """
        logger = open(log_path, 'x')
        n_file = len(record_paths)
        print(f'n_file: {n_file}')
        pb = ProgressBar(n_file)
        for record_i, record_path in enumerate(record_paths):
            x, y = self.file_reader(record_path)
            n_batch = np.int(np.ceil(x.shape[0]/self.batch_size))
            y_est = []
            for batch_i in range(n_batch):
                batch_slice = slice(batch_i*self.batch_size,
                                    (batch_i+1)*self.batch_size)
                y_est_batch = self.nn(x[batch_slice])
                y_est.append(y_est_batch)

            add_log(logger, record_path, tf.concat(y_est, axis=0))
            pb.update()

        logger.close()
