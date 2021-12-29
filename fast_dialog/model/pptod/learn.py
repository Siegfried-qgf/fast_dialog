import os
import sys
import json

import random
import torch
from torch import nn
import torch.nn.functional as F
import operator
from operator import itemgetter
import progressbar
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.append('../../..')
from fast_dialog.evaluator.eval import MultiWozEvaluator

def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--data_path_prefix', type=str, help='The path where the data stores.')
    parser.add_argument('--dataset_name', type=str, choices=['MultiWOZ_2.0', 'MultiWOZ_2.1', 'MultiWOZ_2.1'])
    parser.add_argument('--shuffle_mode', type=str, default='shuffle_session_level', 
        help="shuffle_session_level or shuffle_turn_level, it controls how we shuffle the training data.")

    parser.add_argument('--use_db_as_input', type=str, default='True', 
        help="True or False, whether includes db result as part of the input when generating response.")

    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')

    parser.add_argument('--train_data_ratio', type=float, default=1.0, help='the ratio of training data used for training the model')

    # model configuration
    parser.add_argument('--model_name', type=str, help='t5-small, t5-base or t5-large')
    parser.add_argument('--pretrained_path', default=None, type=str, help='the path that stores pretrained checkpoint.')

    # training configuration
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--epoch_num", default=60, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help='Batch size for each gpu. 2080ti: 2; 3090: 4')  
    parser.add_argument("--batch_size_per_gpu_eval", type=int, default=4, help='Batch size for each gpu when evaluating. 2080ti: 64; 3090: 128;')  
    parser.add_argument("--number_of_gpu", type=int, default=8, help="Number of available GPUs.")  
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="gradient accumulation step.")
    parser.add_argument("--ckpt_save_path", type=str, help="directory to save the model parameters.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    return parser.parse_args()

def reduce_mean(number):
    results = [number for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(results, number)
    results = torch.Tensor(results)
    return results.mean()

def distributed_concat(result_list, num_total_examples):
    output_lists = [result_list[:] for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(output_lists, result_list)
    concat = []
    for single_gpu_output in output_lists:
        concat.extend(single_gpu_output)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]

def parse_dataset(args):
    if args.dataset_name == 'MultiWOZ_2.0':
        args.data_path_prefix = '../../data/dataset/multiwoz/MultiWOZ_2.0'
        if args.model_name == 't5-small':
            args.ckpt_save_path = './ckpt/MultiWOZ_2.0/small/full_training/'
        elif args.model_name == 't5-base':
            args.ckpt_save_path = './ckpt/MultiWOZ_2.0/base/full_training/'
        elif args.model_name == 't5-large':
            args.ckpt_save_path = './ckpt/MultiWOZ_2.0/large/full_training/'
    elif args.dataset_name == 'MultiWOZ_2.1':
        args.data_path_prefix = '../../data/dataset/multiwoz/MultiWOZ_2.1'
        if args.model_name == 't5-small':
            args.ckpt_save_path = './ckpt/MultiWOZ_2.1/small/full_training/'
        elif args.model_name == 't5-base':
            args.ckpt_save_path = './ckpt/MultiWOZ_2.1/base/full_training/'
        elif args.model_name == 't5-large':
            args.ckpt_save_path = './ckpt/MultiWOZ_2.1/large/full_training/'
    elif args.dataset_name == 'MultiWOZ_2.2':
        args.data_path_prefix = '../../data/dataset/multiwoz/MultiWOZ_2.2'
        if args.model_name == 't5-small':
            args.ckpt_save_path = './ckpt/MultiWOZ_2.2/small/full_training/'
        elif args.model_name == 't5-base':
            args.ckpt_save_path = './ckpt/MultiWOZ_2.2/base/full_training/'
        elif args.model_name == 't5-large':
            args.ckpt_save_path = './ckpt/MultiWOZ_2.2/large/full_training/'
    else:
        raise Exception('暂时还没这个数据集')

def get_optimizers(model, args):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    from transformers.optimization import Adafactor
    optimizer = Adafactor(
        optimizer_grouped_parameters,
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
        )
    scheduler = None
    return optimizer, scheduler

import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False

    args = parse_config()
    parse_dataset(args)

    if args.pretrained_path is None:
        args.pretrained_path = args.model_name
    if args.batch_size_per_gpu_eval is None:
        args.batch_size_per_gpu_eval = args.batch_size_per_gpu

    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
            torch.cuda.set_device(args.local_rank)
            device = torch.device('cuda', args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
        else:
            print ('Using single GPU training.')
            device = torch.device('cuda')
    else:
        pass

    if args.local_rank not in [-1, 0]: # block other processes
        torch.distributed.barrier()

    assert args.model_name.startswith('t5')
    from transformers import T5Tokenizer
    print ('Loading Pretrained Tokenizer...')
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path, local_files_only=True)

    if args.use_db_as_input == 'True':
        use_db_as_input = True
    elif args.use_db_as_input == 'False':
        use_db_as_input = False
    else:
        raise Exception('Wrong Use DB Mode!!!')

    if args.add_special_decoder_token == 'True':
        add_special_decoder_token = True
    elif args.add_special_decoder_token == 'False':
        add_special_decoder_token = False
    else:
        raise Exception('Wrong Add Special Token Mode!!!')

    print ('Start loading data...')
    from dataclass import SequentialDistributedSampler, MultiWozData, collate_fn, collate_fn_eval
    from fast_dialog.config.config_pptod import Config
    cfg = Config(args.data_path_prefix)

    MultiWozData.pad_token_id = tokenizer.convert_tokens_to_ids(['<_PAD_>'])[0]
    data = MultiWozData(args.model_name, tokenizer, cfg, args.data_path_prefix, shuffle_mode=args.shuffle_mode, 
        data_mode='train', use_db_as_input=use_db_as_input, add_special_decoder_token=add_special_decoder_token, 
        train_data_ratio=args.train_data_ratio)
    print ('Data loaded')
    evaluator = MultiWozEvaluator(data.reader, cfg)

    print ('Start loading model...')
    from modelling.T5Model import T5Gen_Model
    model = T5Gen_Model(args.pretrained_path, data.tokenizer, data.special_token_list, dropout=args.dropout, 
        add_special_decoder_token=add_special_decoder_token, is_training=True)
    print ('Model loaded')

    if args.local_rank == 0:
        torch.distributed.barrier()

    if cuda_available:
        model = model.to(device)
        if multi_gpu_training:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        pass

    optimizer, _ = get_optimizers(model, args)
    optimizer.zero_grad()

    min_dev_loss = 1e10
    max_dev_score, max_dev_str = 0., ''

    train_sampler = DistributedSampler(data.train_dataset_for_dataloader)
    train_dataloader = DataLoader(data.train_dataset_for_dataloader, sampler=train_sampler, batch_size=args.batch_size_per_gpu, collate_fn=collate_fn)
    # train_iterator = data.build_iterator(batch_size=args.number_of_gpu * args.batch_size_per_gpu, mode='train')
    for epoch in range(args.epoch_num):
        model.train()
        # --- training --- #
        if args.local_rank == 0:
            print ('-----------------------------------------')
            print ('Start training at epoch %d' % epoch)
        epoch_step, train_loss = 0, 0.
        train_dataloader.sampler.set_epoch(epoch)
        for train_batch in tqdm(train_dataloader):
            # one_train_input_batch, one_train_output_batch = train_batch
            # if len(one_train_input_batch) == 0 or len(one_train_output_batch) == 0: break
            # train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels = \
            # data.parse_batch_tensor(train_batch)

            train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels = train_batch

            if cuda_available:
                train_batch_src_tensor = train_batch_src_tensor.to(device)
                train_batch_src_mask = train_batch_src_mask.to(device)
                train_batch_input = train_batch_input.to(device)
                train_batch_labels = train_batch_labels.to(device)
            loss = model(train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels)
            loss = loss.mean()
            loss /= args.gradient_accumulation_steps
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            epoch_step += 1

            if (epoch_step+1) % args.gradient_accumulation_steps == 0 or (epoch_step + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()

        train_loss = train_loss / len(train_dataloader)
        reduce_mean(train_loss)
        if args.local_rank == 0:
            print ('At epoch %d, total update steps is %d, the total training loss is %5f' % (epoch, epoch_step, train_loss))
            print ('++++++++++++++++++++++++++++++++++++++++++')
        # **********************************************************************
        # for few-shot learning, we let the model to first train for several epochs
        if args.train_data_ratio <= 0.1:
            if args.pretrained_path == 'None':
                if epoch < 10: # first train 10 epoches
                    continue
            else:
                if epoch < 3: # first train 10 epoches
                    continue
        elif args.train_data_ratio == 0.2:
            if epoch < 3: # first train 5 epoches
                continue
        else:
            pass
        # **********************************************************************
        # --- evaluation --- #
        from inference_utlis import batch_generate
        if args.local_rank == 0:
            print ('Start evaluation at epoch %d' % epoch)
        model.eval()
        with torch.no_grad():
            ref_bs, ref_act, ref_db = False, False, False # we only consider e2e evaluation
            input_contain_db=use_db_as_input
            # dev_batch_list = data.build_all_evaluation_batch_list(ref_bs, ref_act, ref_db, input_contain_db, 
                # eva_batch_size=args.number_of_gpu * args.batch_size_per_gpu, eva_mode='dev')
            dev_dataset = data.build_all_evaluation_batch_list(ref_bs, ref_act, ref_db, input_contain_db, 
                eva_batch_size=args.number_of_gpu * args.batch_size_per_gpu, eva_mode='dev', for_dataloader=True)
            eval_sampler = SequentialDistributedSampler(dataset=dev_dataset, batch_size=args.batch_size_per_gpu_eval)
            eval_dataloader = DataLoader(dev_dataset, sampler=eval_sampler, batch_size=args.batch_size_per_gpu_eval, collate_fn=collate_fn_eval)
            all_dev_result = []
            for one_inference_batch in tqdm(eval_dataloader):
                dev_batch_parse_dict = batch_generate(model, one_inference_batch, ref_bs, ref_act, ref_db, 
                    input_contain_db, data, MultiWozData, args)
                for item in dev_batch_parse_dict:
                    all_dev_result.append(item)

            torch.distributed.barrier()

            all_dev_result = distributed_concat(all_dev_result, len(dev_dataset))

            if args.local_rank != 0:
                torch.distributed.barrier()
            else:
                dev_bleu, dev_success, dev_match = evaluator.validation_metric(all_dev_result)
                dev_score = 0.5 * (dev_success + dev_match) + dev_bleu

                print ('Inform: %2.2f  Success: %2.2f  BLEU: %2.2f    Score: %.2f' % (dev_match, dev_success, dev_bleu, dev_score))
                one_dev_str = 'dev_e2e_evaluation_inform_{}_success_{}_bleu_{}_combine_score_{}'.format(round(dev_match, 2),
                    round(dev_success,2), round(dev_bleu,2), round(dev_score,2))
                if dev_score > max_dev_score:
                    max_dev_str = one_dev_str
                    max_dev_score = dev_score
                    print ('Saving Model...')
                    model_save_path = args.ckpt_save_path + '/epoch_' + str(epoch) + '_' + one_dev_str
                    # model_save_path = args.ckpt_save_path + '/epoch_' + str(epoch) + '_best_ckpt'

                    import os
                    if os.path.exists(model_save_path):
                        pass
                    else: # recursively construct directory
                        os.makedirs(model_save_path, exist_ok=True)

                    if cuda_available and torch.cuda.device_count() > 1:
                        model.module.save_model(model_save_path)
                    else:
                        model.save_model(model_save_path)

                    pkl_save_path = model_save_path + '/' + one_dev_str + '.json'
                    with open(pkl_save_path, 'w') as outfile:
                        json.dump(all_dev_result, outfile, indent=4)
                    print ('Validation result saved.')
                    # --------------------------------------------------------------------------------------------- #
                    # removing extra checkpoints...
                    # only save 1 checkpoints
                    import os
                    from operator import itemgetter
                    fileData = {}
                    test_output_dir = args.ckpt_save_path
                    for fname in os.listdir(test_output_dir):
                        if fname.startswith('epoch'):
                            fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                        else:
                            pass
                    sortedFiles = sorted(fileData.items(), key=itemgetter(1))
                    max_save_num = 1
                    if len(sortedFiles) < max_save_num:
                        pass
                    else:
                        delete = len(sortedFiles) - max_save_num
                        for x in range(0, delete):
                            one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                            print (one_folder_name)
                            os.system('rm -r ' + one_folder_name)
                    print ('-----------------------------------')
                    # --------------------------------------------------------------------------------------------- #

                print ('Current Result: ' + one_dev_str)
                print ('Best Result: ' + max_dev_str)
                print ('dev evaluation finished.')

                torch.distributed.barrier()

        if args.local_rank == 0:
            print ('-----------------------------------------')