# encoding:utf-8
import sys
import time
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from dataset import Dataset
# from vocab import Vocab
from Agent2 import Agent


# 定义模型参数
def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('GACM')
    # parser.add_argument('--prepare', action='store_true',
    #                     help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--pretrain', action='store_true',
                        help='pretrain the model')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--rank', action='store_true',
                        help='rank on train set')
    parser.add_argument('--gpu', type=str, default='0,1',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--g_lr', type=float, default=0.001,
                                help='learning rate of generator')
    train_settings.add_argument('--d_lr', type=float, default=0.01,
                                help='learning rate of discriminator')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--momentum', type=float, default=0.99,
                                help='momentum')
    train_settings.add_argument('--dropout_rate', type=float, default=0.5,
                                help='dropout rate')
    train_settings.add_argument('--alpha', type=float, default=0.5,
                                help='policy_surr')
    train_settings.add_argument('--beta', type=float, default=0.5,
                                help='policy entropy')
    train_settings.add_argument('--gamma', type=float, default=0.99,
                                help='discount factor')
    train_settings.add_argument('--tau', type=float, default=0.95,
                                help='gae')
    train_settings.add_argument('--clip_epsilon', type=float, default=0.2,
                                help='ppo')
    train_settings.add_argument('--batch_size', type=int, default=20,
                                help='train batch size')
    train_settings.add_argument('--num_steps', type=int, default=200000,
                                help='number of training steps')
    train_settings.add_argument('--num_train_files', type=int, default=40,
                                help='number of training files')
    train_settings.add_argument('--num_dev_files', type=int, default=40,
                                help='number of dev files')
    train_settings.add_argument('--num_test_files', type=int, default=40,
                                help='number of test files')
    train_settings.add_argument('--minimum_occurrence', type=int, default=1,
                                help='minimum_occurrence for NDCG')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', default='GACM',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=100,
                                help='size of the embeddings')
    model_settings.add_argument('--gru_hidden_size', type=int, default=64,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--critic_hidden_size', type=int, default=[64, 32], nargs='+',
                                help='size of critic hidden units')
    model_settings.add_argument('--actor_hidden_size', type=int, default=[64, 32], nargs='+',
                                help='size of actor hidden units')
    model_settings.add_argument('--max_d_num', type=int, default=10,
                                help='max number of docs in a session')
    # model_settings.add_argument('--max_q_len', type=int, default=20,
    #                             help='max length of question')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_dirs', nargs='+',
                               default=['./data/train_per_query.txt'],
                               help='list of dirs that contain the preprocessed train data')
    path_settings.add_argument('--dev_dirs', nargs='+',
                               default=['./data/dev_per_query.txt'],
                               help='list of dirs that contain the preprocessed dev data')
    path_settings.add_argument('--test_dirs', nargs='+',
                               default=['./data/test_per_query.txt'],
                               help='list of dirs that contain the preprocessed test data')
    path_settings.add_argument('--human_label_dir', default='./data/human_label.txt',
                               help='the dir to Human Label txt file')
    # path_settings.add_argument('--qfreq_file', help='the file of query frequency')
    # path_settings.add_argument('--dfreq_file', help='the file of doc frequency')
    # path_settings.add_argument('--brc_dir', default='../data/baidu',
    #                            help='the dir with preprocessed baidu reading comprehension data')
    # path_settings.add_argument('--vocab_dir', default='../data/vocab/',
    #                            help='the dir to save vocabulary')
    path_settings.add_argument('--load_dir', default='./data/models/',
                               help='the dir to load models')
    path_settings.add_argument('--save_dir', default='./data/models/',
                               help='the dir to save models')
    path_settings.add_argument('--result_dir', default='./data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='./data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')

    path_settings.add_argument('--eval_freq', type=int, default=100,
                               help='the frequency of evaluating on the dev set when training')
    path_settings.add_argument('--check_point', type=int, default=1000,
                               help='the frequency of saving model')
    path_settings.add_argument('--patience', type=int, default=3,
                               help='lr half when more than the patience times of evaluation\' loss don\'t decrease')
    path_settings.add_argument('--lr_decay', type=float, default=0.5,
                               help='lr decay')
    # path_settings.add_argument('--min_cnt', type=int, default=0,
    #                            help='min_cnt')
    path_settings.add_argument('--load_model', type=int, default=-1,
                               help='load model global step')
    path_settings.add_argument('--data_parallel', type=bool, default=True,
                               help='data_parallel')
    path_settings.add_argument('--use_gpu', type=bool, default=True,
                               help='use_gpu')
    path_settings.add_argument('--gpu_num', type=int, default=1,
                               help='gpu_num')

    return parser.parse_args()



def rank(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("GACM")
    logger.info('Checking the data files...')
    for data_path in args.dev_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    # logger.info('Load data_set and vocab...')
    # with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
    #     vocab = pickle.load(fin)
    #     logger.info('Vocab size is {}'.format(vocab.size()))
    dataset = Dataset(args, test_dirs=args.test_dirs, isRank=True)
    logger.info('Initialize the model...')
    model = Model(args, len(dataset.qid_query), len(dataset.uid_url),  len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1
    logger.info('Restoring the model...')
    model.load_model(model_dir=args.load_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Training the model...')
    dev_batches = dataset.gen_mini_batches('test', args.batch_size, shuffle=False)
    model.evaluate(dev_batches, dataset, result_dir=args.result_dir,
        result_prefix='rank.predicted.{}.{}.{}'.format(args.algo, args.load_model, time.time()))
    logger.info('Done with model ranking!')


def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("GACM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    # logger.info('Load data_set and vocab...')
    # with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
    #     vocab = pickle.load(fin)
    #     logger.info('Vocab size is {}'.format(vocab.size()))
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs)
    logger.info('Initialize the model...')
    model = Agent(args, len(dataset.qid_query), len(dataset.uid_url),  len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    if args.load_model > -1:
        logger.info('Restoring the model...')
        model.load_model(model_dir=args.load_dir, model_prefix=args.algo, global_step=args.load_model, load_optimizer=False)
    logger.info('Training the model...')
    model.train(dataset)
    logger.info('Done with model training!')


def pretrain(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("GACM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    # logger.info('Load data_set and vocab...')
    # with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
    #     vocab = pickle.load(fin)
    #     logger.info('Vocab size is {}'.format(vocab.size()))
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs)
    logger.info('Initialize the model...')
    model = Agent(args, len(dataset.qid_query), len(dataset.uid_url),  len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    if args.load_model > -1:
        logger.info('Restoring the model...')
        model.load_model(model_dir=args.load_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Training the model...')
    model.pretrain(dataset)
    logger.info('Done with model training!')


def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("GACM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    # logger.info('Load data_set and vocab...')
    # with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
    #     vocab = pickle.load(fin)
    #     logger.info('Vocab size is {}'.format(vocab.size()))

    assert len(args.dev_dirs) > 0, 'No dev files are provided.'
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs)
    logger.info('Restoring the model...')
    model = Model(args, len(dataset.qid_query), len(dataset.uid_url),  len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    model.load_model(model_dir=args.load_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Evaluating the model on dev set...')
    dev_batches = dataset.gen_mini_batches('dev', args.batch_size, shuffle=False)
    dev_loss = model.evaluate(dev_batches, dataset, result_dir=args.result_dir,
        result_prefix='dev.predicted.{}.{}.{}'.format(args.algo, args.load_model, time.time()))
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Predicted results are saved to {}'.format(os.path.join(args.result_dir)))


def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("GACM")
    logger.info('Checking the data files...')
    for data_path in args.dev_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)

    dataset = Dataset(args, dev_dirs=args.dev_dirs, isRank=True)
    logger.info('Initialize the model...')
    model = Model(args, len(dataset.qid_query), len(dataset.uid_url), len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1
    logger.info('Restoring the model...')
    model.load_model(model_dir=args.load_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Training the model...')
    dev_batches = dataset.gen_mini_batches('test', args.batch_size, shuffle=False)
    model.evaluate(dev_batches, dataset, result_dir=args.result_dir,
                   result_prefix='rank.predicted.{}.{}.{}'.format(args.algo, args.load_model, time.time()))
    logger.info('Done with model ranking!')


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()
    assert args.batch_size % args.gpu_num == 0
    assert args.gru_hidden_size % 2 == 0

    # create a logger
    logger = logging.getLogger("GACM")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))


    logger.info('Checking the directories...')
    for dir_path in [args.save_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.pretrain:
        pretrain(args)
    if args.predict:
        predict(args)
    if args.rank:
        rank(args)
    logger.info('run done.')


if __name__ == '__main__':
    run()
