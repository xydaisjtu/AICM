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
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
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
                    dev_loss, dev_perplexity, dev_perplexity_at_rank = self.evaluate(dev_batches, data)
                    #print('dev loss=%s' % dev_loss, 'dev ppl=%s' % dev_perplexity, 'dev ppl at rank=', dev_perplexity_at_rank)

                    test_batches = data.gen_mini_batches('test', 41405, shuffle=False)
                    test_loss, test_perplexity, test_perplexity_at_rank = self.evaluate(test_batches, data)
                    #print('test loss=%s' % test_loss, 'dev ppl=%s' % test_perplexity, 'dev ppl at rank=' , test_perplexity_at_rank)

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
                    # Trick: do not decay d_lr help convergence
                    if patience >= self.patience:
                        #self.adjust_learning_rate(self.discrim_optimizer, self.args.lr_decay)
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
            QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
            UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
            VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
            PRE_CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64)[:, :-1]))
            CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64)[:, 1:]))

            # generate trajectories
            for __ in range(self.args.d_step):
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

                '''update discriminator'''
                for _ in range(self.args.k):
                    self.discrim.train()
                    self.discrim_optimizer.zero_grad()
                    g_o, _ = self.discrim(QIDS, UIDS, VIDS, CLICKS_)
                    g_o_target = torch.ones((QIDS.shape[0], g_o.shape[1]))
                    e_o, _ = self.discrim(QIDS, UIDS, VIDS, CLICKS)
                    e_o_target = torch.zeros((QIDS.shape[0], e_o.shape[1]))
                    if self.use_cuda:
                        g_o_target, e_o_target = g_o_target.cuda(), e_o_target.cuda()
                    
                    discrim_loss = self.discrim_criterion(g_o, g_o_target) + self.discrim_criterion(e_o, e_o_target)
                    discrim_loss.backward()
                    self.discrim_optimizer.step()
                    self.writer.add_scalar('train/d_loss', discrim_loss.data, self.global_step)

            '''estimate advantage'''
            with torch.no_grad():
                self.discrim.eval()
                rewards = -torch.log(self.discrim(QIDS, UIDS, VIDS, CLICKS_)[0])
                # print(rewards.shape, values.shape)
                #print(tensor_type)
                #exit(0)
                deltas = torch.zeros(rewards.shape)
                advantages = torch.zeros(rewards.shape)
                prev_value = torch.zeros(rewards.shape[0])
                prev_advantage = torch.zeros(rewards.shape[0])
                if self.use_cuda:
                    deltas, advantages = deltas.cuda(), advantages.cuda()
                    prev_value, prev_advantage = prev_value.cuda(), prev_advantage.cuda()
                '''print(deltas)
                print(advantages)
                print(prev_value)
                print(prev_advantage)
                exit(0)'''

                for i in reversed(range(rewards.size(1))):
                    deltas[:, i] = rewards[:, i] + self.gamma * prev_value - values[:, i]
                    advantages[:, i] = deltas[:, i] + self.gamma * self.tau * prev_advantage
                    prev_value = values[:, i]
                    prev_advantage = advantages[:, i]

                returns = values + advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + MINF)
                # advantages = (returns - returns.mean())/returns.std()

                fixed_log_probs = torch.distributions.Categorical(logits).log_prob(CLICKS_[:, 1:])

            '''PPO update'''
            self.policy.train()
            optim_batchsize = 512
            optim_iter_num = int(math.ceil(QIDS.shape[0] / optim_batchsize))
            if self.use_cuda:
                CLICKS_ = torch.cat((torch.zeros((CLICKS_.shape[0], 1), dtype=CLICKS_.dtype, device=torch.device('cuda')), CLICKS_), dim=1)
            else:
                CLICKS_ = torch.cat((torch.zeros((CLICKS_.shape[0], 1), dtype=CLICKS_.dtype), CLICKS_), dim=1)
            for _ in range(self.args.g_step):
                perm = np.arange(QIDS.shape[0])
                np.random.shuffle(perm)

                QIDS, UIDS, VIDS, PRE_CLICKS, CLICKS, CLICKS_, advantages, returns, fixed_log_probs = \
                    QIDS[perm].clone(), UIDS[perm].clone(), VIDS[perm].clone(), PRE_CLICKS[perm].clone(), \
                    CLICKS[perm].clone(), CLICKS_[perm].clone(), advantages[perm].clone(), returns[perm].clone(), fixed_log_probs[perm].clone()

                #print(QIDS)
                #exit(0)

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

                    self.policy_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                    self.policy_optimizer.step()
                    g_loss, _ = self.compute_loss(logits, clicks_b)

                    self.writer.add_scalar('train/g_loss', g_loss.data, self.global_step)
                    self.writer.add_scalar('train/g_valueloss', value_loss.data, self.global_step)
                    self.writer.add_scalar('train/g_policysurr', policy_surr.data, self.global_step)
                    self.writer.add_scalar('train/g_entropy', pe.data, self.global_step)

            if check_point > 0 and self.global_step % check_point == 0:
                self.save_model(save_dir, save_prefix)
            if self.global_step >= num_steps:
                exit_tag = True

        return max_metric_value, exit_tag, metric_save, patience

    def train(self, data):
        max_metric_value, patience, metric_save = 0., 0, 1e10
        step_pbar = tqdm(total=self.args.num_steps)
        exit_tag = False
        self.writer.add_scalar('train/g_lr', self.g_lr, self.global_step)
        self.writer.add_scalar('train/d_lr', self.d_lr, self.global_step)
        while not exit_tag:
            train_batches = data.gen_mini_batches('train', self.args.batch_size, shuffle=True)
            max_metric_value, exit_tag, metric_save, patience = self._train_epoch(train_batches, data,
                                                                                max_metric_value, metric_save,
                                                                                patience, step_pbar)

    def pretrain(self, data):
        max_metric_value, patience, metric_save = 0., 0, 1e10
        step_pbar = tqdm(total=self.args.num_steps)
        exit_tag = False
        evaluate = True
        num_steps = self.args.num_steps
        check_point, batch_size = self.args.check_point, self.args.batch_size
        save_dir, save_prefix = self.args.save_dir, self.args.algo
        self.writer.add_scalar('pretrain/g_lr', self.g_lr, self.global_step)
        self.writer.add_scalar('pretrain/d_lr', self.d_lr, self.global_step)
        while not exit_tag:
            train_batches = data.gen_mini_batches('train', self.args.batch_size, shuffle=True)
            for b_itx, batch in enumerate(train_batches):
                self.global_step += 1
                step_pbar.update(1)
                QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
                UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
                VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
                CLICKS_DISCRIM = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:, 1:])
                CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:, :-1])
                if self.use_cuda:
                    QIDS, UIDS, VIDS, CLICKS, CLICKS_DISCRIM = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda(), CLICKS_DISCRIM.cuda()

                self.policy.train()
                self.policy_optimizer.zero_grad()
                pred_logits, _, _, _ = self.policy(QIDS, UIDS, VIDS, CLICKS)
                loss, loss_list = self.compute_loss(pred_logits, batch['clicks'])
                loss.backward()
                self.policy_optimizer.step()
                self.writer.add_scalar('pretrain/g_loss', loss.data[0], self.global_step)

                self.policy.eval()
                actor_rnn_state = Variable(torch.zeros(1, QIDS.shape[0], self.gru_hidden_size))
                critic_rnn_state = Variable(torch.zeros(1, QIDS.shape[0], self.gru_hidden_size))
                CLICK_ = torch.zeros(QIDS.shape[0], 1, dtype=CLICKS.dtype)
                if self.use_cuda:
                    actor_rnn_state, critic_rnn_state, CLICK_ = actor_rnn_state.cuda(), critic_rnn_state.cuda(), CLICK_.cuda()
                click_list = []
                for i in range(self.max_d_num + 1):
                    logit, value, actor_rnn_state, critic_rnn_state = self.policy(QIDS[:, i:i + 1], UIDS[:, i:i + 1], VIDS[:, i:i + 1], CLICK_, actor_rnn_state=actor_rnn_state, critic_rnn_state=critic_rnn_state)
                    if i > 0:
                        CLICK_ = torch.distributions.Categorical(logit).sample()
                        click_list.append(CLICK_)

                CLICKS_ = torch.squeeze(torch.stack(click_list, dim=1))
                CLICKS_ = torch.cat((torch.zeros((CLICKS_.shape[0], 1), dtype=CLICKS_.dtype, device=self.device), CLICKS_), dim=1)

                self.discrim.train()
                self.discrim_optimizer.zero_grad()
                g_o, _ = self.discrim(QIDS, UIDS, VIDS, CLICKS_)
                e_o, _ = self.discrim(QIDS, UIDS, VIDS, CLICKS_DISCRIM)
                discrim_loss = self.discrim_criterion(g_o, torch.ones((QIDS.shape[0], g_o.shape[1]), device=self.device)) + \
                                self.discrim_criterion(e_o, torch.zeros((QIDS.shape[0], e_o.shape[1]), device=self.device))
                discrim_loss.backward()
                self.discrim_optimizer.step()
                self.writer.add_scalar('pretrain/d_loss', discrim_loss.data, self.global_step)

                if evaluate and self.global_step % self.eval_freq == 0:
                    if data.dev_set is not None:
                        dev_batches = data.gen_mini_batches('dev', batch_size, shuffle=False)
                        dev_loss, dev_perplexity, dev_perplexity_at_rank = self.evaluate(dev_batches, data)
                        torch.cuda.empty_cache()
                        #print('dev loss=%s' % dev_loss, 'dev ppl=%s' % dev_perplexity, 'dev ppl at rank=', dev_perplexity_at_rank)
                        
                        test_batches = data.gen_mini_batches('test', batch_size, shuffle=False)
                        test_loss, test_perplexity, test_perplexity_at_rank = self.evaluate(test_batches, data)
                        torch.cuda.empty_cache()
                        #print('test loss=%s' % test_loss, 'dev ppl=%s' % test_perplexity, 'dev ppl at rank=', test_perplexity_at_rank)

                        self.writer.add_scalar("dev/loss", dev_loss, self.global_step)
                        self.writer.add_scalar("dev/perplexity", dev_perplexity, self.global_step)
                        self.writer.add_scalar("test/loss", test_loss, self.global_step)
                        self.writer.add_scalar("test/perplexity", test_perplexity, self.global_step)

                        # Sequence dependent ranking task
                        label_batches = data.gen_mini_batches('label', 1, shuffle=False)
                        ndcg_version1, ndcg_version2 = self.ndcg_cheat(label_batches, data)
                        torch.cuda.empty_cache()
                        for trunc_level in self.trunc_levels:
                            self.writer.add_scalar("NDCG_version1/{}".format(trunc_level), ndcg_version1[trunc_level], self.global_step)
                            self.writer.add_scalar("NDCG_version2/{}".format(trunc_level), ndcg_version2[trunc_level], self.global_step)

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
            total_loss += loss.data[0] * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

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

    def ndcg(self, label_batches, data, result_dir=None, result_prefix=None, stop=-1):
        trunc_levels = [1, 3, 5, 10]
        ndcg_version1, ndcg_version2 = {}, {}
        useless_session, cnt_version1, cnt_version2 = {}, {}, {}
        for k in trunc_levels:
            ndcg_version1[k] = 0.0
            ndcg_version2[k] = 0.0
            useless_session[k] = 0
            cnt_version1[k] = 0
            cnt_version2[k] = 0
        with torch.no_grad():
            for b_itx, batch in enumerate(label_batches):
                if b_itx == stop:
                    break
                
                QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
                UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
                VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
                CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:, :-1])
                true_relevances = batch['relevances'][0]
                if self.use_cuda:
                    QIDS, UIDS, VIDS, CLICKS = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda()

                self.policy.eval()
                pred_logits, _, _, _ = self.policy(QIDS, UIDS, VIDS, CLICKS)
                pred_logits = pred_logits[:, :, 1:].squeeze(2)
                relevances = pred_logits.data.cpu().numpy().reshape(-1).tolist()
                pred_rels = {}
                for idx, relevance in enumerate(relevances):
                    pred_rels[idx] = relevance
                
                for k in trunc_levels:
                    #print('\n{}: {}'.format('trunc_level', k))
                    ideal_ranking_relevances = sorted(true_relevances, reverse=True)[:k]
                    ranking = sorted([idx for idx in pred_rels], key = lambda idx : pred_rels[idx], reverse=True)
                    ranking_relevances = [true_relevances[idx] for idx in ranking[:k]]
                    dcg = self.dcg(ranking_relevances)
                    idcg = self.dcg(ideal_ranking_relevances)
                    if dcg > idcg:
                        pprint.pprint(ranking_relevances)
                        pprint.pprint(ideal_ranking_relevances)
                        pprint.pprint(dcg)
                        pprint.pprint(idcg)
                        assert 0
                    ndcg = dcg / idcg if idcg > 0 else 1.0
                    if idcg == 0:
                        useless_session[k] += 1
                        cnt_version2[k] += 1
                        ndcg_version2[k] += ndcg
                    else:
                        ndcg = dcg / idcg
                        cnt_version1[k] += 1
                        cnt_version2[k] += 1
                        ndcg_version1[k] += ndcg
                        ndcg_version2[k] += ndcg

            for k in trunc_levels:
                assert cnt_version1[k] + useless_session[k] == 2000
                assert cnt_version2[k] == 2000
                ndcg_version1[k] /= cnt_version1[k]
                ndcg_version2[k] /= cnt_version2[k]
        return ndcg_version1, ndcg_version2

    def dcg(self, ranking_relevances):
        """
        Computes the DCG for a given ranking_relevances
        """
        return sum([(2 ** relevance - 1) / math.log(rank + 2, 2) for rank, relevance in enumerate(ranking_relevances)])

    def generate_synthetic_dataset(self, batch_type, dataset, file_path, file_name, synthetic_type='deterministic', shuffle_split=None, amplification=1):
        assert batch_type in ['train', 'dev', 'test'], 'unsupported batch_type: {}'.format(batch_type)
        assert synthetic_type in ['deterministic', 'stochastic'], 'unsupported synthetic_type: {}'.format(synthetic_type)
        if synthetic_type == 'deterministic' and shuffle_split is None and amplification > 1:
            print('this is a useless generative setting for synthetic dataset:')
            print('  - synthetic_type: {}'.format(synthetic_type))
            print('  - shuffle_split: {}'.format(str(shuffle_split)))
            print('  - amplification: {}'.format(amplification))
            return 
        np.random.seed(2333)
        torch.manual_seed(2333)

        check_path(file_path)
        data_path = os.path.join(file_path, file_name)
        file = open(data_path, 'w')
        self.logger.info('Generating synthetic dataset based on the {} set...'.format(batch_type))
        self.logger.info('  - The synthetic dataset will be expended by {} times'.format(amplification))
        self.logger.info('  - Click generative type {}'.format(synthetic_type))
        self.logger.info('  - Shuffle split: {}'.format(str(shuffle_split)))

        for amp_idx in range(amplification):
            self.logger.info('  - Generation at amplification {}'.format(amp_idx))
            eval_batches = dataset.gen_mini_batches(batch_type, self.args.batch_size, shuffle=False)
            for b_itx, batch in enumerate(eval_batches):
                #pprint.pprint(batch)
                if b_itx % 5000 == 0:
                    self.logger.info('    - Generating click sequence at step: {}.'.format(b_itx))

                # get the numpy version of input data
                QIDS_numpy = np.array(batch['qids'], dtype=np.int64)
                UIDS_numpy = np.array(batch['uids'], dtype=np.int64)
                VIDS_numpy = np.array(batch['vids'], dtype=np.int64)
                CLICKS_numpy = np.array(batch['clicks'], dtype=np.int64)

                # shuffle uids and vids according to shuffle_split
                if shuffle_split is not None:
                    self.logger.info('    - Start shuffling uids & vids...')
                    assert type(shuffle_split) == type([0]), 'type of shuffle_split should be a list, but got {}'.format(type(shuffle_split))
                    assert len(shuffle_split) > 1, 'shuffle_split should have at least 2 elements but got only {}'.format(len(shuffle_split))
                    shuffle_split.sort()
                    assert shuffle_split[0] >= 1 and shuffle_split[-1] <=11, 'all elements in shuffle_split should be in range of [1, 11], but got: {}'.format(shuffle_split)
                    for i in range(UIDS_numpy.shape[0]):
                        for split_idx in range(len(shuffle_split) - 1):
                            split_left = shuffle_split[split_idx]
                            split_right= shuffle_split[split_idx + 1]
                            shuffle_state = np.random.get_state()
                            np.random.shuffle(UIDS_numpy[i, split_left:split_right])
                            np.random.set_state(shuffle_state)
                            np.random.shuffle(VIDS_numpy[i, split_left:split_right])

                # get the tensor version of input data (maybe shuffled) from the numpy version
                QIDS = Variable(torch.from_numpy(QIDS_numpy))
                UIDS = Variable(torch.from_numpy(UIDS_numpy))
                VIDS = Variable(torch.from_numpy(VIDS_numpy))
                CLICKS = Variable(torch.from_numpy(CLICKS_numpy))
                if self.use_cuda:
                    QIDS, UIDS, VIDS, CLICKS = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda()

                # start predict the click info
                self.policy.eval()
                actor_rnn_state = Variable(torch.zeros(1, QIDS.shape[0], self.gru_hidden_size))
                critic_rnn_state = Variable(torch.zeros(1, QIDS.shape[0], self.gru_hidden_size))
                CLICK_ = torch.zeros(self.args.batch_size, 1, dtype=CLICKS.dtype)
                if self.use_cuda:
                    actor_rnn_state, critic_rnn_state, CLICK_ = actor_rnn_state.cuda(), critic_rnn_state.cuda(), CLICK_.cuda()
                click_list = []
                for i in range(self.max_d_num + 1):
                    logit, value, actor_rnn_state, critic_rnn_state = self.policy(QIDS[:, i:i+1], 
                                                                                    UIDS[:, i:i+1], 
                                                                                    VIDS[:, i:i+1], 
                                                                                    CLICK_, 
                                                                                    actor_rnn_state, 
                                                                                    critic_rnn_state)
                    if i > 0:
                        logit = logit[:, :, 1:].squeeze(2)
                        if synthetic_type == 'deterministic':
                            CLICK_ = (logit > 0.5).type(CLICKS.dtype)
                        elif synthetic_type == 'stochastic':
                            random_tmp = torch.rand(logit.shape)
                            if self.use_cuda:
                                random_tmp = random_tmp.cuda()
                            CLICK_ = (random_tmp <= logit).type(CLICKS.dtype)

                        click_list.append(CLICK_)
                CLICKS_ = torch.cat(click_list, dim=1).cpu().numpy().tolist()
                UIDS = UIDS.cpu().numpy().tolist()
                VIDS = VIDS.cpu().numpy().tolist()
                assert len(CLICKS_[0]) == 10
                for qids, uids, vids, clicks in zip(batch['qids'], UIDS, VIDS, CLICKS_):
                    qid = dataset.qid_query[qids[0]]
                    uids = [dataset.uid_url[uid] for uid in uids]
                    vids = [dataset.vid_vtype[vid] for vid in vids]
                    file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(0, qid, 0, 0, str(uids[1:]), str(vids[1:]), str(clicks)))
                #exit(0)
        self.logger.info('Finish synthetic dataset generation...')
        file.close()

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
        #print(state_dict[0].items())
        if self.args.data_parallel:
            state_dict = [{'module.{}'.format(k) if 'module' not in k else k:v for k, v in state_dict[i].items()} for i in [0, 1]]
        else:
            state_dict = [{k.replace('module.', ''):v for k, v in state_dict[i].items()} for i in [0, 1]]
        #print(state_dict[0].items())
        self.policy.load_state_dict(state_dict[0])
        self.discrim.load_state_dict(state_dict[1])
        self.logger.info('Model restored from {}, with prefix {} and global step {}.'.format(model_dir, model_prefix, global_step))
