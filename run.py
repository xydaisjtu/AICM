# encoding:utf-8
import sys
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
import importlib
from utils import *

Dataset = None
Agent = None

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('GACM')
    parser.add_argument('--pretrain', action='store_true',
                        help='pretrain the model')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--test', action='store_true',
                        help='test the model')
    parser.add_argument('--rank', action='store_true',
                        help='rank on train set')
    parser.add_argument('--rank_cheat', action='store_true',
                        help='rank on train set in a cheating way')
    parser.add_argument('--generate_click_seq', action='store_true',
                        help='generate click sequence based on model itself')
    parser.add_argument('--generate_click_seq_cheat', action='store_true',
                        help='generate click sequence based on ground truth data')
    parser.add_argument('--generate_synthetic_dataset', action='store_true',
                        help='generate synthetic dataset for reverse ppl')
    parser.add_argument('--use_gpu', action='store_true',
                        help='use gpu instead of cpu')
    parser.add_argument('--gpu_num', type=int, default=1,
                        help='gpu_num')
    parser.add_argument('--data_parallel', action='store_true',
                        help='data_parallel')
    parser.add_argument('--dataset_version', type=int, default=1,
                        help='version number of the dataset that is used')
    parser.add_argument('--agent_version', type=int, default=1,
                        help='version number of the agent that is used')

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
    train_settings.add_argument('--num_train_files', type=int, default=1,
                                help='number of training files')
    train_settings.add_argument('--num_dev_files', type=int, default=1,
                                help='number of dev files')
    train_settings.add_argument('--num_test_files', type=int, default=1,
                                help='number of test files')
    train_settings.add_argument('--num_label_files', type=int, default=1,
                                help='number of label files')
    train_settings.add_argument('--minimum_occurrence', type=int, default=1,
                                help='minimum_occurrence for NDCG')
    train_settings.add_argument('--g_step', type=int, default=4,
                                help='generator is updated g_step times during one epoch')
    train_settings.add_argument('--d_step', type=int, default=1,
                                help='synthetic trajectory is generated d_step times during one epoch')
    train_settings.add_argument('--k', type=int, default=1,
                                help='discriminator is updated k times during one epoch')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', default='GACM',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=100,
                                help='size of the embeddings')
    model_settings.add_argument('--gru_hidden_size', type=int, default=64,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--critic_hidden_size', type=int, nargs='+', default=[64, 32],
                                help='size of critic hidden units')
    model_settings.add_argument('--max_d_num', type=int, default=10,
                                help='max number of docs in a session')

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
    path_settings.add_argument('--label_dirs', nargs='+',
                                default=['data/human_label_for_GACM.txt'],
                                help='list of dirs that contain the preprocessed label data')
    path_settings.add_argument('--human_label_dir', default='./data/human_label.txt',
                                help='the dir to Human Label txt file')
    path_settings.add_argument('--load_dir', default='./outputs/models/',
                                help='the dir to load models')
    path_settings.add_argument('--save_dir', default='./outputs/models/',
                                help='the dir to save models')
    path_settings.add_argument('--result_dir', default='./outputs/results/',
                                help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='./outputs/summary/',
                                help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_dir', default='./outputs/log/',
                                help='path of the log file. If not set, logs are printed to console')

    path_settings.add_argument('--eval_freq', type=int, default=10,
                                help='the frequency of evaluating on the dev set when training')
    path_settings.add_argument('--check_point', type=int, default=1000,
                                help='the frequency of saving model')
    path_settings.add_argument('--patience', type=int, default=3,
                                help='lr half when more than the patience times of evaluation\' loss don\'t decrease')
    path_settings.add_argument('--lr_decay', type=float, default=0.5,
                                help='lr decay')
    path_settings.add_argument('--load_model', type=int, default=-1,
                                help='load model at global step')
    path_settings.add_argument('--load_pretrain_model', type=int, default=-1,
                                help='load the pretrained model at global step')

    return parser.parse_args()

def pretrain(args):
    """
    pretrain the model
    """
    logger = logging.getLogger("GACM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs + args.test_dirs + args.label_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    assert len(args.train_dirs) > 0, 'No train files are provided.'
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs, label_dirs=args.label_dirs)
    logger.info('Initialize the model...')
    model = Agent(args, len(dataset.qid_query), len(dataset.uid_url),  len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    if args.load_pretrain_model > -1:
        logger.info('Reloading the pretrain model...')
        model.load_model(model_dir=args.load_dir, model_prefix=args.algo, global_step=args.load_pretrain_model, load_optimizer=True)
    logger.info('Pretraining the model...')
    model.pretrain(dataset)
    logger.info('Done with model pretraining!')

def train(args):
    """
    trains the model
    """
    logger = logging.getLogger("GACM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs + args.test_dirs + args.label_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    assert len(args.train_dirs) > 0, 'No train files are provided.'
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs, label_dirs=args.label_dirs)
    logger.info('Initialize the model...')
    model = Agent(args, len(dataset.qid_query), len(dataset.uid_url), len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    if args.load_pretrain_model > -1:
        logger.info('Reloading the pretrain model...')
        model.load_model(model_dir=args.load_dir, model_prefix=args.algo, global_step=args.load_pretrain_model, load_optimizer=False)
    elif args.load_model > -1:
        logger.info('Restoring the model...')
        model.load_model(model_dir=args.load_dir, model_prefix=args.algo, global_step=args.load_model, load_optimizer=True)
    logger.info('Training the model...')
    model.train(dataset)
    logger.info('Done with model training!')

def test(args):
    """
    test the model
    """
    logger = logging.getLogger("GACM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs + args.test_dirs + args.label_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    assert len(args.train_dirs) > 0, 'No train files are provided.'
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs, label_dirs=args.label_dirs)
    logger.info('Initialize the model...')
    model = Agent(args, len(dataset.qid_query), len(dataset.uid_url),  len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1
    logger.info('Restoring the model...')
    model.load_model(model_dir=args.load_dir, model_prefix=args.algo, global_step=args.load_model, load_optimizer=False)
    logger.info('Computing the Log Likelihood and Perplexity...')
    test_batches = dataset.gen_mini_batches('test', args.batch_size, shuffle=False)
    test_loss, perplexity, perplexity_at_rank = model.evaluate(test_batches, dataset, result_dir=args.result_dir,
                                                                result_prefix='test.predicted.{}.{}.{}'.format(args.algo, args.load_model, time.time()))
    logger.info('Loss on test set: {}'.format(test_loss))
    logger.info('perplexity on test set: {}'.format(perplexity))
    logger.info('perplexity at rank: {}'.format(perplexity_at_rank))
    logger.info('Predicted results are saved to {}'.format(os.path.join(args.result_dir)))
    logger.info('Done with model testing!')

def rank(args):
    """
    cheat on ranking performance on test files
    """
    logger = logging.getLogger("GACM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs + args.test_dirs + args.label_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs, label_dirs=args.label_dirs)
    logger.info('Initialize the model...')
    model = Agent(args, len(dataset.qid_query), len(dataset.uid_url), len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1
    logger.info('Restoring the model...')
    model.load_model(model_dir=args.load_dir, model_prefix=args.algo, global_step=args.load_model, load_optimizer=False)
    logger.info('Start computing NDCG@k for ranking performance (cheat)')
    label_batches = dataset.gen_mini_batches('label', 1, shuffle=False)
    trunc_levels = [1, 3, 5, 10]
    ndcgs_version1, ndcgs_version2 = model.ndcg(label_batches, dataset)
    for trunc_level in trunc_levels:
        ndcg_version1, ndcg_version2 = ndcgs_version1[trunc_level], ndcgs_version2[trunc_level]
        logger.info("NDCG@{}: {}, {}".format(trunc_level, ndcg_version1, ndcg_version2))
    logger.info('【{}, {}】'.format(args.load_model, args.minimum_occurrence))
    logger.info('Done with model testing!')


def generate_synthetic_dataset(args):
    """
    generate synthetic dataset for reverse ppl
    """
    logger = logging.getLogger("GACM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs + args.test_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    assert len(args.test_dirs) > 0, 'No test files are provided.'
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs)
    logger.info('Initialize the model...')
    model = Agent(args, len(dataset.qid_query), len(dataset.uid_url), len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1
    logger.info('Restoring the model...')
    model.load_model(model_dir=args.load_dir, model_prefix=args.algo, global_step=args.load_model, load_optimizer=False)

    synthetic_types = ['deterministic', 'stochastic']
    shuffle_splits = [None, [1, 11], [1, 6, 11]]
    amplifications = [1, 7]
    for synthetic_type in synthetic_types:
        for shuffle_split in shuffle_splits:
            for amplification in amplifications:
                #synthetic_type = 'deterministic'
                #shuffle_split = None
                #amplification = 1
                file_path = os.path.join(args.load_dir, '..', 'synthetic')
                model.generate_synthetic_dataset('test', dataset, file_path, 
                                                'synthetic_{}_{}_{}.txt'.format(synthetic_type[0].upper(), str(shuffle_split), amplification), 
                                                synthetic_type=synthetic_type, shuffle_split=shuffle_split, amplification=amplification)
                # exit()
    logger.info('Done with click sequence generation.')

def run():
    """
    Prepares and runs the whole system.
    """
    # get arguments
    args = parse_args()
    assert args.batch_size % args.gpu_num == 0
    assert args.gru_hidden_size % 2 == 0

    # create a logger
    logger = logging.getLogger("GACM")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    check_path(args.save_dir)
    check_path(args.load_dir)
    check_path(args.result_dir)
    check_path(args.summary_dir)
    if args.log_dir:
        check_path(args.log_dir)
        file_handler = logging.FileHandler(args.log_dir + time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())) + '.txt')
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
    
    global Dataset
    global Agent
    logger.info('Agent version: {}.0'.format(args.agent_version))
    logger.info('Dataset version: {}.0'.format(args.dataset_version))
    logger.info('Checking the directories...')
    Dataset = importlib.import_module('dataset{}'.format(args.dataset_version)).Dataset
    Agent = importlib.import_module('Agent{}'.format(args.agent_version)).Agent
    
    if args.pretrain:
        pretrain(args)
    if args.train:
        train(args)
    if args.test:
        test(args)
    if args.rank:
        rank(args)
    if args.generate_synthetic_dataset:
        generate_synthetic_dataset(args)
    logger.info('run done.')

if __name__ == '__main__':
    run()
