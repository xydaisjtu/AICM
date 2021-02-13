import os
import time
import logging
import json
import math
import numpy as np
import torch
import copy
import math
import random
from torch.autograd import Variable
from tqdm import tqdm
from Generator import Policy
from Discriminator import Discriminator
from tensorboardX import SummaryWriter
from torch import nn
from ndcg import RelevanceEstimator
from TianGong_HumanLabel_Parser import TianGong_HumanLabel_Parser
from utils import *

MINF = 1e-30

class Agent(object):
    def __init__(self, args, query_size, doc_size, vtype_size):
        # logging
        self.logger = logging.getLogger("GACM")

        # basic config
        self.args = args
        self.use_cuda = torch.cuda.is_available() if args.use_gpu else False
        self.gru_hidden_size = args.gru_hidden_size
        self.optim_type = args.optim
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        self.weight_decay = args.weight_decay
        self.eval_freq = args.eval_freq
        self.global_step = args.load_model if args.load_model > -1 else 0
        self.patience = args.patience
        self.max_d_num = args.max_d_num
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma =args.gamma
        self.tau = args.tau
        self.clip_epsilon = args.clip_epsilon
        self.writer = None
        if args.train or args.pretrain:
            self.writer = SummaryWriter(self.args.summary_dir)

        # Networks
        self.policy = Policy(self.args, query_size, doc_size, vtype_size)
        self.discrim = Discriminator(self.args, query_size, doc_size, vtype_size)

        if args.data_parallel:
            self.policy = nn.DataParallel(self.policy)
            self.discrim = nn.DataParallel(self.discrim)
        if self.use_cuda:
            self.policy = self.policy.cuda()
            self.discrim = self.discrim.cuda()

        self.policy_optimizer = self.create_train_op(self.policy, self.g_lr)
        self.discrim_optimizer = self.create_train_op(self.discrim, self.d_lr)
        self.discrim_criterion = nn.BCELoss()

        # for NDCG@k
        self.relevance_queries = TianGong_HumanLabel_Parser().parse(args.human_label_dir)
        self.relevance_estimator = RelevanceEstimator(args.minimum_occurrence)
        self.trunc_levels = [1, 3, 5, 10]

    def compute_loss(self, pred_scores, target_scores):
        """
        The loss function
        """
        total_loss = 0.
        loss_list = []
        cnt = 0
        for batch_idx, scores in enumerate(target_scores):
            cnt += 1
            loss = 0.
            for position_idx, score in enumerate(scores[2:]):
                if score == 0:
                    loss -= torch.log(pred_scores[batch_idx, position_idx, 0].view(1) + MINF)
                else:
                    loss -= torch.log(pred_scores[batch_idx, position_idx, 1].view(1) + MINF)
            loss_list.append(loss.data[0])
            total_loss += loss
        total_loss /= cnt
        return total_loss, loss_list

    def compute_perplexity(self, pred_scores, target_scores):
        '''
        Compute the perplexity
        '''
        perplexity_at_rank = [0.0] * self.max_d_num  # 10 docs per query
        total_num = 0
        for batch_idx, scores in enumerate(target_scores):
            total_num += 1
            for position_idx, score in enumerate(scores[2:]):
                if score == 0:
                    perplexity_at_rank[position_idx] += torch.log2(pred_scores[batch_idx, position_idx, 0].view(1) + MINF)
                else:
                    perplexity_at_rank[position_idx] += torch.log2(pred_scores[batch_idx, position_idx, 1].view(1) + MINF)
        return total_num, perplexity_at_rank

    def create_train_op(self, model, learning_rate):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'rprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        return optimizer

    def adjust_learning_rate(self, optimizer, decay_rate=0.99):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    def _train_epoch(self, train_batches, data, max_metric_value, metric_save, patience, step_pbar):
        """
        Trains the model for a single epoch.
        """
        evaluate = True
        exit_tag = False
        num_steps = self.args.num_steps
        check_point, batch_size = self.args.check_point, self.args.batch_size
        save_dir, save_prefix = self.args.save_dir, self.args.algo

        for bitx, batch in enumerate(train_batches):
            if evaluate and self.global_step % self.eval_freq == 0:
                if data.dev_set is not None:
                    dev_batches = data.gen_mini_batches('dev', 31928, shuffle=False)
                    dev_loss, dev_perplexity, dev_perplexity_at_rank = self.evaluate(dev_batches, data, result_dir=self.args.result_dir, stop=-1,
                                                                                    result_prefix='train_dev.predicted.{}.{}'.format(
                                                                                        self.args.algo, self.global_step))

                    test_batches = data.gen_mini_batches('test', 41405, shuffle=False)
                    test_loss, test_perplexity, test_perplexity_at_rank = self.evaluate(test_batches, data, result_dir=self.args.result_dir, stop=-1, 
                                                                                        result_prefix='train_test.predicted.{}.{}'.format(
                                                                                            self.args.algo, self.global_step))

                    self.writer.add_scalar("dev/loss", dev_loss, self.global_step)
                    self.writer.add_scalar("dev/perplexity", dev_perplexity, self.global_step)
                    self.writer.add_scalar("test/loss", test_loss, self.global_step)
                    self.writer.add_scalar("test/perplexity", test_perplexity, self.global_step)

                    if dev_loss < metric_save:
                        metric_save = dev_loss
                        patience = 0
                    else:
                        patience += 1

                    # Trick: do not decay d_lr help convergence
                    if patience >= self.patience:
                        self.adjust_learning_rate(self.policy_optimizer, self.args.lr_decay)
                        self.g_lr *= self.args.lr_decay
                        #self.d_lr *= self.args.lr_decay
                        self.writer.add_scalar('train/g_lr', self.g_lr, self.global_step)
                        #self.writer.add_scalar('train/d_lr', self.d_lr, self.global_step)
                        metric_save = dev_loss
                        patience = 0
                        self.patience += 1
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')

            self.global_step += 1
            step_pbar.update(1)
            t0 = time.time()
            QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
            UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
            VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
            PRE_CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64)[:, :-1]))
            CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64)[:, 1:]))

            # generate trajectories
            actor_rnn_state = Variable(torch.zeros(1, QIDS.shape[0], self.gru_hidden_size))
            critic_rnn_state = Variable(torch.zeros(1, QIDS.shape[0], self.gru_hidden_size))
            CLICK_ = torch.zeros(QIDS.shape[0], 1, dtype=CLICKS.dtype)
            logits = torch.zeros(QIDS.shape[0], 0, 2)
            values = torch.zeros(QIDS.shape[0], 0)
            CLICKS_ = Variable(torch.zeros((QIDS.shape[0], 0), dtype=CLICKS.dtype))
            if self.use_cuda:
                QIDS, UIDS, VIDS, PRE_CLICKS, CLICKS = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), PRE_CLICKS.cuda(), CLICKS.cuda()
                actor_rnn_state, critic_rnn_state, CLICK_ = actor_rnn_state.cuda(), critic_rnn_state.cuda(), CLICK_.cuda()
                logits, values, CLICKS_ = logits.cuda(), values.cuda(), CLICKS_.cuda()
            self.policy.eval()
            for i in range(self.max_d_num + 1):
                logit, value, actor_rnn_state, critic_rnn_state = self.policy(QIDS[:, i:i+1], 
                                                                                UIDS[:, i:i+1], 
                                                                                VIDS[:, i:i+1], 
                                                                                CLICK_, 
                                                                                actor_rnn_state, 
                                                                                critic_rnn_state)
                if i > 0:
                    CLICK_ = torch.distributions.Categorical(logit).sample()
                    logits = torch.cat([logits, logit], dim=1)
                    values = torch.cat([values, value], dim=1)
                    CLICKS_ = torch.cat([CLICKS_, CLICK_], dim=1)

            if self.use_cuda:
                CLICKS_ = torch.cat((torch.zeros((CLICKS_.shape[0], 1), dtype=CLICKS_.dtype, device=torch.device('cuda')), CLICKS_), dim=1)
            else:
                CLICKS_ = torch.cat((torch.zeros((CLICKS_.shape[0], 1), dtype=CLICKS_.dtype), CLICKS_), dim=1)
            
            t1 = time.time()

            '''update discriminator'''
            self.discrim.train()
            self.discrim_optimizer.zero_grad()
            g_o = self.discrim(QIDS, UIDS, VIDS, CLICKS_)
            g_o_target = torch.ones((QIDS.shape[0], g_o.shape[1]))
            e_o = self.discrim(QIDS, UIDS, VIDS, CLICKS)
            e_o_target = torch.zeros((QIDS.shape[0], e_o.shape[1]))
            if self.use_cuda:
                g_o_target, e_o_target = g_o_target.cuda(), e_o_target.cuda()
            
            discrim_loss = self.discrim_criterion(g_o, g_o_target) + self.discrim_criterion(e_o, e_o_target)
            discrim_loss.backward()
            self.discrim_optimizer.step()
            self.writer.add_scalar('train/d_loss', discrim_loss.data, self.global_step)

            t2 = time.time()


            '''estimate advantage'''
            with torch.no_grad():
                self.discrim.eval()
                rewards = -torch.log(self.discrim(QIDS, UIDS, VIDS, CLICKS_))
                deltas = torch.zeros(rewards.shape)
                advantages = torch.zeros(rewards.shape)
                prev_value = torch.zeros(rewards.shape[0])
                prev_advantage = torch.zeros(rewards.shape[0])
                if self.use_cuda:
                    deltas, advantages = deltas.cuda(), advantages.cuda()
                    prev_value, prev_advantage = prev_value.cuda(), prev_advantage.cuda()

                for i in reversed(range(rewards.size(1))):
                    deltas[:, i] = rewards[:, i] + self.gamma * prev_value - values[:, i]
                    advantages[:, i] = deltas[:, i] + self.gamma * self.tau * prev_advantage
                    prev_value = values[:, i]
                    prev_advantage = advantages[:, i]

                returns = values + advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + MINF)

                fixed_log_probs = torch.distributions.Categorical(logits).log_prob(CLICKS_[:, 1:])


            t3 = time.time()

            '''PPO update'''
            self.policy.train()
            optim_epochs = 4
            optim_batchsize = 512
            optim_iter_num = int(math.ceil(QIDS.shape[0] / optim_batchsize))
            if self.use_cuda:
                CLICKS_ = torch.cat((torch.zeros((CLICKS_.shape[0], 1), dtype=CLICKS_.dtype, device=torch.device('cuda')), CLICKS_), dim=1)
            else:
                CLICKS_ = torch.cat((torch.zeros((CLICKS_.shape[0], 1), dtype=CLICKS_.dtype), CLICKS_), dim=1)
            for _ in range(optim_epochs):
                perm = np.arange(QIDS.shape[0])
                np.random.shuffle(perm)

                QIDS, UIDS, VIDS, PRE_CLICKS, CLICKS, CLICKS_, advantages, returns, fixed_log_probs = \
                    QIDS[perm].clone(), UIDS[perm].clone(), VIDS[perm].clone(), PRE_CLICKS[perm].clone(), \
                    CLICKS[perm].clone(), CLICKS_[perm].clone(), advantages[perm].clone(), returns[perm].clone(), fixed_log_probs[perm].clone()

                for i in range(optim_iter_num):
                    ind = slice(i * optim_batchsize, min((i + 1) * optim_batchsize, QIDS.shape[0]))
                    qids_b, uids_b, vids_b, pclicks_b, clicks_b, clicks__b, advantage_b, returns_b, fixed_log_probs_b = \
                        QIDS[ind], UIDS[ind], VIDS[ind], CLICKS_[ind, :-1], CLICKS[ind], CLICKS_[ind, 2:], \
                            advantages[ind], returns[ind], fixed_log_probs[ind]

                    logits, values_pred, _, _ = self.policy(qids_b, uids_b, vids_b, pclicks_b)
                    dist = torch.distributions.Categorical(logits)


                    '''update critic'''
                    value_loss = (values_pred - returns_b).pow(2).mean()
                    '''optimizer policy'''
                    log_probs_b = dist.log_prob(clicks__b)
                    ratio = torch.exp(log_probs_b - fixed_log_probs_b)
                    surr1 = ratio * advantage_b
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage_b
                    policy_surr = -torch.min(surr1, surr2).mean()
                    pe = dist.entropy().mean()
                    loss = value_loss + self.alpha * policy_surr - self.beta * pe

                    t4 = time.time()
                    self.policy_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                    self.policy_optimizer.step()
                    g_loss, _ = self.compute_loss(logits, clicks_b)

                    t5 = time.time()
                    self.writer.add_scalar('train/g_loss', g_loss.data, self.global_step)
                    self.writer.add_scalar('train/g_valueloss', value_loss.data, self.global_step)
                    self.writer.add_scalar('train/g_policysurr', policy_surr.data, self.global_step)
                    self.writer.add_scalar('train/g_entropy', pe.data, self.global_step)

            t6 = time.time()

            if check_point > 0 and self.global_step % check_point == 0:
                self.save_model(save_dir, save_prefix)
            if self.global_step >= num_steps:
                exit_tag = True

        return max_metric_value, exit_tag, metric_save, patience

    def train(self, data):
        max_metric_value, epoch, patience, metric_save = 0., 0, 0, 1e10
        step_pbar = tqdm(total=self.args.num_steps)
        exit_tag = False
        self.writer.add_scalar('train/g_lr', self.g_lr, self.global_step)
        self.writer.add_scalar('train/d_lr', self.d_lr, self.global_step)
        while not exit_tag:
            epoch += 1
            train_batches = data.gen_mini_batches('train', self.args.batch_size, shuffle=True)
            max_metric_value, exit_tag, metric_save, patience = self._train_epoch(train_batches, data,
                                                                                max_metric_value, metric_save,
                                                                                patience, step_pbar)

    def pretrain(self, data):
        max_metric_value, epoch, patience, metric_save = 0., 0, 0, 1e10
        step_pbar = tqdm(total=self.args.num_steps)
        exit_tag = False
        evaluate = True
        num_steps = self.args.num_steps
        check_point, batch_size = self.args.check_point, self.args.batch_size
        save_dir, save_prefix = self.args.save_dir, self.args.algo
        self.writer.add_scalar('pretrain/g_lr', self.g_lr, self.global_step)
        self.writer.add_scalar('pretrain/d_lr', self.d_lr, self.global_step)
        while not exit_tag:
            epoch += 1
            train_batches = data.gen_mini_batches('train', self.args.batch_size, shuffle=True)
            for b_itx, batch in enumerate(train_batches):
                self.global_step += 1
                step_pbar.update(1)
                QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
                UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
                VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
                CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:, :-1])
                if self.use_cuda:
                    QIDS, UIDS, VIDS, CLICKS = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda()

                self.policy.train()
                self.policy_optimizer.zero_grad()
                pred_logits, _, _, _ = self.policy(QIDS, UIDS, VIDS, CLICKS)
                loss, loss_list = self.compute_loss(pred_logits, batch['clicks'])
                loss.backward()
                self.policy_optimizer.step()
                self.writer.add_scalar('pretrain/g_loss', loss.data[0], self.global_step)

                self.policy.eval()
                rnn_state = Variable(torch.zeros(1, QIDS.shape[0], self.gru_hidden_size * 2))
                CLICK_ = torch.zeros(QIDS.shape[0], 1, dtype=CLICKS.dtype)
                if self.use_cuda:
                    rnn_state, CLICK_ = rnn_state.cuda(), CLICK_.cuda()
                click_list = []
                for i in range(self.max_d_num + 1):
                    logit, value, rnn_state = self.policy(QIDS[:, i:i + 1], UIDS[:, i:i + 1], VIDS[:, i:i + 1], CLICK_, rnn_state=rnn_state)
                    if i > 0:
                        CLICK_ = torch.distributions.Categorical(logit).sample()
                        click_list.append(CLICK_)

                CLICKS_ = torch.squeeze(torch.stack(click_list, dim=1))
                CLICKS_ = torch.cat((torch.zeros((CLICKS_.shape[0], 1), dtype=CLICKS_.dtype), CLICKS_), dim=1)

                self.discrim.train()
                self.discrim_optimizer.zero_grad()
                g_o = self.discrim(QIDS, UIDS, VIDS, CLICKS_)
                e_o = self.discrim(QIDS, UIDS, VIDS, CLICKS)
                discrim_loss = self.discrim_criterion(g_o, torch.ones((QIDS.shape[0], g_o.shape[1]))) + \
                                self.discrim_criterion(e_o, torch.zeros((QIDS.shape[0], e_o.shape[1])))
                discrim_loss.backward()
                self.discrim_optimizer.step()
                self.writer.add_scalar('pretrain/d_loss', discrim_loss.data, self.global_step)

                if evaluate and self.global_step % self.eval_freq == 0:
                    if data.dev_set is not None:
                        dev_batches = data.gen_mini_batches('dev', batch_size, shuffle=False)
                        dev_loss, dev_perplexity, dev_perplexity_at_rank = self.evaluate(dev_batches, data,
                                                                                        result_dir=self.args.result_dir, stop=-1,
                                                                                        result_prefix='train_dev.predicted.{}.{}'.format(
                                                                                            self.args.algo, self.global_step))
                        
                        test_batches = data.gen_mini_batches('test', batch_size, shuffle=False)
                        test_loss, test_perplexity, test_perplexity_at_rank = self.evaluate(test_batches, data,
                                                                                            result_dir=self.args.result_dir, stop=-1,
                                                                                            result_prefix='train_test.predicted.{}.{}'.format(
                                                                                                self.args.algo, self.global_step))

                        self.writer.add_scalar("dev/loss", dev_loss, self.global_step)
                        self.writer.add_scalar("dev/perplexity", dev_perplexity, self.global_step)
                        self.writer.add_scalar("test/loss", test_loss, self.global_step)
                        self.writer.add_scalar("test/perplexity", test_perplexity, self.global_step)

                        for trunc_level in self.trunc_levels:
                            ndcg_version1, ndcg_version2 = self.relevance_estimator.evaluate(self, data, self.relevance_queries, trunc_level)
                            self.writer.add_scalar("NDCG_version1/{}".format(trunc_level), ndcg_version1, self.global_step)
                            self.writer.add_scalar("NDCG_version2/{}".format(trunc_level), ndcg_version2, self.global_step)

                        if dev_loss < metric_save:
                            metric_save = dev_loss
                            patience = 0
                        else:
                            patience += 1
                        if patience >= self.patience:
                            self.adjust_learning_rate(self.policy_optimizer, self.args.lr_decay)
                            self.g_lr *= self.args.lr_decay
                            self.writer.add_scalar('pretrain/lr', self.g_lr, self.global_step)
                            metric_save = dev_loss
                            patience = 0
                            self.patience += 1
                    else:
                        self.logger.warning('No dev set is loaded for evaluation in the dataset!')
                if check_point > 0 and self.global_step % check_point == 0:
                    self.save_model(save_dir, save_prefix)
                if self.global_step >= num_steps:
                    exit_tag = True


    def evaluate(self, eval_batches, dataset, result_dir=None, result_prefix=None, stop=-1):
        #eval_ouput = []
        total_loss, total_num, perplexity_num = 0., 0, 0
        perplexity_at_rank = [0.0] * self.max_d_num
        for b_itx, batch in enumerate(eval_batches):
            if b_itx == stop:
                break
            if b_itx % 5000 == 0:
                self.logger.info('Evaluation step {}.'.format(b_itx))
            QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
            UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
            VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
            CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:, :-1])
            if self.use_cuda:
                QIDS, UIDS, VIDS, CLICKS = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda()

            self.policy.eval()
            pred_logits, _, _, _ = self.policy(QIDS, UIDS, VIDS, CLICKS)
            loss, loss_list = self.compute_loss(pred_logits, batch['clicks'])
            tmp_num, tmp_perplexity_at_rank = self.compute_perplexity(pred_logits, batch['clicks'])
            perplexity_num += tmp_num
            perplexity_at_rank = [perplexity_at_rank[i] + tmp_perplexity_at_rank[i] for i in range(10)]
            '''for pred_metric, data, pred_logit in zip(loss_list, batch['raw_data'], pred_logits.data.cpu().numpy().tolist()):
                eval_ouput.append([data['session_id'], data['query'], data['urls'][1:], 
                                    data['vtypes'][1:], data['clicks'][2:], pred_logit, pred_metric])'''
            total_loss += loss.data[0] * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

        '''if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.txt')
            with open(result_file, 'w') as fout:
                for sample in eval_ouput:
                    fout.write('\t'.join(map(str, sample)) + '\n')
            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))'''

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        assert total_num == perplexity_num
        ave_span_loss = 1.0 * total_loss / total_num
        perplexity_at_rank = [2 ** (-x / perplexity_num) for x in perplexity_at_rank]
        perplexity = sum(perplexity_at_rank) / len(perplexity_at_rank)
        return ave_span_loss, perplexity, perplexity_at_rank

    def predict_relevance(self, qid, uid, vid):
        qids = [[qid, qid]]
        uids = [[0, uid]]
        vids = [[0, vid]]
        clicks = [[0, 0, 0]]
        QIDS = Variable(torch.from_numpy(np.array(qids, dtype=np.int64)))
        UIDS = Variable(torch.from_numpy(np.array(uids, dtype=np.int64)))
        VIDS = Variable(torch.from_numpy(np.array(vids, dtype=np.int64)))
        CLICKS = Variable(torch.from_numpy(np.array(clicks, dtype=np.int64))[:, :-1])
        if self.use_cuda:
            QIDS, UIDS, VIDS, CLICKS = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda()
        self.policy.eval()
        pred_logits, _ , _, _ = self.policy(QIDS, UIDS, VIDS, CLICKS)
        return pred_logits[0, 0, 1]

    def generate_click_seq(self, eval_batches, file_path, file_name):
        check_path(file_path)
        data_path = os.path.join(file_path, file_name)
        file = open(data_path, 'w')
        for b_itx, batch in enumerate(eval_batches):
            if b_itx % 5000 == 0:
                self.logger.info('Generating click sequence at step: {}.'.format(b_itx))
            QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
            UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
            VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
            CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64)))
            if self.use_cuda:
                QIDS, UIDS, VIDS, CLICKS = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda()

            self.policy.eval()
            rnn_state = Variable(torch.zeros(1, QIDS.shape[0], self.gru_hidden_size * 2))
            CLICK_ = torch.zeros(QIDS.shape[0], 1, dtype=CLICKS.dtype)
            if self.use_cuda:
                rnn_state, CLICK_ = rnn_state.cuda(), CLICK_.cuda()
            logit_list = []
            click_list = []
            for i in range(self.max_d_num + 1):
                logit, value, rnn_state = self.policy(QIDS[:, i:i + 1], UIDS[:, i:i + 1], VIDS[:, i:i + 1], CLICK_, rnn_state=rnn_state)
                if i > 0:
                    logit = logit[:, :, 1:].squeeze(2)
                    CLICK_ = (logit > 0.5).type(CLICKS.dtype)
                    logit_list.append(logit)
                    click_list.append(CLICK_)

            logits = torch.cat(logit_list, dim=1).cpu().detach().numpy().tolist()
            CLICKS_ = torch.cat(click_list, dim=1).cpu().numpy().tolist()
            CLICKS = CLICKS[:, 2:].cpu().numpy().tolist()
            assert len(CLICKS[0]) == 10
            
            for logit, CLICK_, CLICK in zip(logits, CLICKS_, CLICKS):
                file.write('{}\t{}\t{}\n'.format(str(logit), str(CLICK_), str(CLICK)))

    def generate_click_seq_cheat(self, eval_batches, file_path, file_name):
        check_path(file_path)
        data_path = os.path.join(file_path, file_name)
        file = open(data_path, 'w')
        for b_itx, batch in enumerate(eval_batches):
            if b_itx % 5000 == 0:
                self.logger.info('Generating click sequence at step: {}.'.format(b_itx))
            QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
            UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
            VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
            CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:, :-1])
            true_clicks = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64)))
            if self.use_cuda:
                QIDS, UIDS, VIDS, CLICKS, true_clicks = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda(), true_clicks.cuda()

            self.policy.eval()
            pred_logits, _, _, _ = self.policy(QIDS, UIDS, VIDS, CLICKS)
            pred_logits = pred_logits[:, :, 1:].squeeze(2)
            pred_clicks = (pred_logits > 0.5).type(true_clicks.dtype).cpu().numpy().tolist()
            pred_logits = pred_logits.detach().cpu().numpy().tolist()
            true_clicks = true_clicks[:, 2:].cpu().numpy().tolist()
            
            for logit, pred_click, true_click in zip(pred_logits, pred_clicks, true_clicks):
                file.write('{}\t{}\t{}\n'.format(str(logit), str(pred_click), str(true_click)))

    def save_model(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        torch.save(self.policy.state_dict(), os.path.join(model_dir, '{}_policy_{}.model'.format(model_prefix, self.global_step)))
        torch.save(self.policy_optimizer.state_dict(), os.path.join(model_dir, '{}_policy_{}.optimizer'.format(model_prefix, self.global_step)))
        torch.save(self.discrim.state_dict(), os.path.join(model_dir, '{}_discrim_{}.model'.format(model_prefix, self.global_step)))
        torch.save(self.discrim_optimizer.state_dict(), os.path.join(model_dir, '{}_discrim_{}.optimizer'.format(model_prefix, self.global_step)))
        self.logger.info('Model and optimizer saved in {}, with prefix {} and global step {}.'.format(model_dir, model_prefix, self.global_step))

    def load_model(self, model_dir, model_prefix, global_step, load_optimizer=True):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        optimizer_path = [os.path.join(model_dir, '{}_{}_{}.optimizer'.format(model_prefix, type, global_step)) for type in ['policy', 'discrim']]
        if load_optimizer:
            self.policy_optimizer.load_state_dict(torch.load(optimizer_path[0]))
            self.discrim_optimizer.load_state_dict(torch.load(optimizer_path[1]))
            self.logger.info('Optimizer restored from {}, with prefix {} and global step {}.'.format(model_dir, model_prefix, global_step))
        
        model_path = [os.path.join(model_dir, '{}_{}_{}.model'.format(model_prefix, type, global_step)) for type in ['policy', 'discrim']]
        if not os.path.isfile(model_path[0]) or not os.path.isfile(model_path[1]):
            self.logger.info('Load file not found. Try to load the best model files.')
            model_path[0] = os.path.join(model_dir, '{}_best_policy.model'.format(model_prefix))
            model_path[1] = os.path.join(model_dir, '{}_best_discrim.model'.format(model_prefix))
        if self.use_cuda:
            state_dict = [torch.load(model_path[i]) for i in [0, 1]]
        else:
            state_dict = [torch.load(model_path[i], map_location=lambda storage, loc: storage) for i in [0, 1]]
        print(state_dict[0].items())
        if self.args.data_parallel:
            state_dict = [{'module.{}'.format(k) if 'module' not in k else k:v for k, v in state_dict[i].items()} for i in [0, 1]]
        else:
            state_dict = [{k.replace('module.', ''):v for k, v in state_dict[i].items()} for i in [0, 1]]
        print(state_dict[0].items())
        self.policy.load_state_dict(state_dict[0])
        self.discrim.load_state_dict(state_dict[1])
        self.logger.info('Model restored from {}, with prefix {} and global step {}.'.format(model_dir, model_prefix, global_step))
