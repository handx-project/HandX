import os
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import json
from accelerate.utils import tqdm

import options.option_transformer as option_trans
import utils.utils_model as utils_model
from models.lit_llama.model_hf import LLaMAHF, LLaMAHFConfig
from transformers import T5EncoderModel, T5Tokenizer, T5TokenizerFast
from accelerate import Accelerator
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from dataset import dataset_TM_train_motionmillion
from accelerate.utils import TorchDynamoPlugin
def build_motion_tokens_with_text_offset(
    raw_m_tokens,
    raw_m_tokens_len,
    text_tokens_len,
    max_motion_length,
    mot_pad_idx,
    mot_end_idx,
    device,
):
    """
    Replicates the original logic:
      final_seq = [PAD x text_len] + m_tokens + [END] + [PAD ...]  (length = max_motion_length)

    Arguments:
      raw_m_tokens:    (B, L_raw_max) long, padded with mot_pad_idx
      raw_m_tokens_len:(B,) lengths before that padding
      text_tokens_len: (B,) number of text tokens (or 1 if pooled)
    """
    B = raw_m_tokens.size(0)
    final_tokens = torch.full(
        (B, max_motion_length),
        fill_value=mot_pad_idx,
        dtype=torch.long,
        device=device,
    )

    for i in range(B):
        tlen = int(text_tokens_len[i].item())
        mlen = int(raw_m_tokens_len[i].item())

        # where motion tokens can start
        start = tlen
        # leave space for END token
        max_motion_space = max_motion_length - start - 1
        if max_motion_space <= 0:
            # no room; just put END at last position
            final_tokens[i, -1] = mot_end_idx
            continue

        use_len = min(mlen, max_motion_space)
        if use_len > 0:
            final_tokens[i, start:start + use_len] = raw_m_tokens[i, :use_len].to(device)

        end_pos = start + use_len
        if end_pos <= max_motion_length:
            final_tokens[i, end_pos] = mot_end_idx

    return final_tokens

def encode_text_batch(
    captions,
    text_encode,
    text_sum_way,
    clip_model=None,
    hf_tokenizer=None,
    hf_model=None,
    device="cuda",
    max_text_length=320,
):
    """
    Returns:
      feat_clip_text: (B, T_text, D) in bfloat16
      y_mask:        (B, T_text)    int/bool
      text_tokens_len: (B,)         long
    """

    if text_encode == "clip":
        # CLIP: outputs a single pooled embedding per text
        with torch.no_grad():
            text_tokens = clip.tokenize(captions, truncate=True).to(device, non_blocking=True)
            feats = clip_model.encode_text(text_tokens).to(torch.bfloat16)  # (B, D)

        # treat as a single "token"
        feat_clip_text = feats.unsqueeze(1)                        # (B, 1, D)
        y_mask = torch.ones((feats.size(0), 1), device=device, dtype=torch.long)
        text_tokens_len = torch.ones((feats.size(0),), device=device, dtype=torch.long)

        return feat_clip_text, y_mask, text_tokens_len

    elif text_encode in ["flan-t5-xl", "flan-t5-xxl"]:
        # HuggingFace encoder-decoder / encoder model
        assert hf_tokenizer is not None and hf_model is not None, \
            "hf_tokenizer and hf_model must be provided for T5."

        with torch.no_grad():
            enc = hf_tokenizer(
                list(captions),
                padding=True,
                truncation=True,
                max_length=max_text_length,
                return_tensors="pt",
            )
            input_ids = enc.input_ids.to(device, non_blocking=True)
            attn_mask = enc.attention_mask.to(device, non_blocking=True)

            outputs = hf_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=False,
            )
            hidden = outputs.last_hidden_state.to(torch.bfloat16)  # (B, T, D)

        y_mask = attn_mask  # (B, T)
        text_tokens_len = y_mask.sum(dim=1)  # (B,)

        # reduce according to text_sum_way; keep sequence dim = 1 for compatibility
        if text_sum_way == "cls":
            # take first token
            pooled = hidden[:, 0, :]                      # (B, D)
            feat_clip_text = pooled.unsqueeze(1)          # (B, 1, D)
            text_tokens_len = torch.ones_like(text_tokens_len)  # treat as 1

        elif text_sum_way == "mean":
            # mask-aware mean
            lengths = text_tokens_len.clamp(min=1).unsqueeze(-1)  # (B, 1)
            pooled = (hidden * y_mask.unsqueeze(-1)).sum(dim=1) / lengths
            feat_clip_text = pooled.unsqueeze(1)          # (B, 1, D)
            text_tokens_len = torch.ones_like(text_tokens_len)

        elif text_sum_way == "sum":
            pooled = (hidden * y_mask.unsqueeze(-1)).sum(dim=1)   # (B, D)
            feat_clip_text = pooled.unsqueeze(1)                  # (B, 1, D)
            text_tokens_len = torch.ones_like(text_tokens_len)

        else:
            # keep full sequence
            feat_clip_text = hidden

        return feat_clip_text, y_mask, text_tokens_len

    else:
        raise ValueError(f"Unknown text encoder: {text_encode}")

# 自定义 warm-up + cosine decay scheduler
class WarmupCosineDecayScheduler:
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=0, resume_trans=None):
        self.optimizer = optimizer
        self.warmup_iters =  warmup_iters 
        self.total_iters = total_iters 
        self.min_lr = min_lr
        self.resume_trans = resume_trans
        
        # if self.resume_trans is None:
        # use LambdaLR to warm up the learning rate linearly
        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
        
        # use CosineAnnealingLR to decay the learning rate
        self.cosine_scheduler = CosineAnnealingLR(optimizer, 
                                                  T_max=total_iters - warmup_iters, 
                                                  eta_min=min_lr)
            
    def warmup_lambda(self, current_iter):
        # if in warm-up period, the learning rate increases linearly
        if current_iter < self.warmup_iters:
            return float(current_iter) / float(max(1, self.warmup_iters))
        # after warm-up period, lambda = 1 (no more learning rate modification by LambdaLR)
        return 1.0

    def step(self, current_iter):
        
        # if in warm-up period, call warmup_scheduler to update the learning rate
        if current_iter < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            # otherwise, use CosineAnnealingLR to update the learning rate
            self.cosine_scheduler.step()
    
    def state_dict(self):
        return {
            'warmup_scheduler' : self.warmup_scheduler.state_dict(),
            'cosine_scheduler' : self.cosine_scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler'])


# custom warm-up + constant scheduler
class WarmupConstantScheduler:
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=0, resume_trans=None):
        self.optimizer = optimizer
        self.warmup_iters = 12000 # warmup_iters
        self.total_iters = total_iters
        self.min_lr = min_lr
        self.resume_trans = resume_trans
        
        # if self.resume_trans is None:
        # use LambdaLR to warm up the learning rate linearly
        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
            
    def warmup_lambda(self, current_iter):
        # if in warm-up period, the learning rate increases linearly
        if current_iter < self.warmup_iters:
            return float(current_iter) / float(max(1, self.warmup_iters))
        # after warm-up period, lambda = 1
        return 1.0

    def step(self, current_iter):
        
        # if in warm-up period, call warmup_scheduler to update the learning rate
        if current_iter < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            pass
    
    def state_dict(self):
        return {
            'warmup_scheduler' : self.warmup_scheduler.state_dict()
        }
        
    def load_state_dict(self, state_dict):
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
 

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train_one_iter(feat_clip_text, m_tokens, m_tokens_len, y_mask, trans_encoder):
    
    m_tokens, m_tokens_len = m_tokens.to(comp_device), m_tokens_len.to(comp_device)
    bs = m_tokens.shape[0]

    target = m_tokens
    
    target = target.to(comp_device)
    
    input_index = target

    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(input_index.shape,
                                                        device=input_index.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape,
                                                        device=input_index.device))
    mask = mask.round().to(dtype=torch.int64)
    r_indices = torch.randint_like(input_index, args.nb_code)
    a_indices = mask*input_index+(1-mask)*r_indices

    cls_pred = trans_encoder(a_indices, feat_clip_text, y_mask)
    
    cls_pred = cls_pred.contiguous()
    

    return cls_pred, target


if __name__ == '__main__':

    ##### ---- Exp dirs ---- #####
    args = option_trans.get_args_parser()
    torch.manual_seed(args.seed)
    if args.debug:
        args.exp_name = 'debug'
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')

    os.makedirs(args.out_dir, exist_ok = True)

    if args.debug:
        args.print_iter = 1
    dynamo_plugin = TorchDynamoPlugin(
        backend="inductor",      # good default
        mode="default",
        fullgraph=False,
        # dynamic=True,
        # mode="default",          # "reduce-overhead" or "max-autotune" are other options
        # fullgraph=False,          # try compiling whole graph when possibleaccelerator = Accelerator(log_with="wandb" )
        # dynamic=False            # set True if your shapes vary a lot
    )
    # accelerate
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps,log_with="wandb")#,dynamo_plugin=dynamo_plugin)
    NAME = f'{args.exp_name}'
    accelerator.init_trackers(
            project_name="AR_Training3", 
            # config=opt,
            init_kwargs={"wandb": {"entity": "","name":NAME
                                    ,"resume":'allow',"id":NAME}}
            )
    
    comp_device = accelerator.device

    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    from utils.word_vectorizer import WordVectorizer

    ##### ---- Network ---- #####
    if args.text_encode == 'clip':
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=comp_device, jit=False)  # Must set jit=False for training
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
    elif args.text_encode == 'flan-t5-xl': #https://huggingface.co/google-t5/t5-base
        tokenizer = T5TokenizerFast.from_pretrained("t5-base")  # tokenizer NOT in prepare()
        # text_encoder =  export_and_get_onnx_model("t5-base")

        text_encoder = T5EncoderModel.from_pretrained("t5-base")#,attn_implementation="flash_attention_2")
        
        text_encoder.set_attn_implementation("sdpa")

        text_encoder.eval()
        # clip_model = (tokenizer, text_encoder)
        # clip_model[1].eval()
        for p in text_encoder.parameters():
            p.requires_grad = False
        

        
        args.clip_dim = 768
        logger.info(f'Flan-t5-xl loaded')
    elif args.text_encode == 'flan-t5-xxl':
        tokenizer = T5Tokenizer.from_pretrained('checkpoints/models--google--flan-t5-xxl/snapshots/ae7c9136adc7555eeccc78cdd960dfd60fb346ce', local_files_only=True)
        text_encoder = T5EncoderModel.from_pretrained('checkpoints/models--google--flan-t5-xxl/snapshots/ae7c9136adc7555eeccc78cdd960dfd60fb346ce', local_files_only=True).to(device=comp_device)
        text_encoder.eval()
        
        clip_model = (tokenizer, text_encoder)
        clip_model[1].eval()
        for p in clip_model[1].parameters():
            p.requires_grad = False
        args.clip_dim = 4096
        logger.info(f'Flan-t5-xxl loaded')
    else:
        raise ValueError(f'Unknown text encoder: {args.text_encode}')



    # net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
    #                     args.nb_code,
    #                     args.code_dim,
    #                     args.output_emb_width,
    #                     args.down_t,
    #                     args.stride_t,
    #                     args.width,
    #                     args.depth,
    #                     args.dilation_growth_rate,
    #                     args.vq_act,
    #                     args.vq_norm,
    #                     args.kernel_size,
    #                     args.use_patcher,
    #                     args.patch_size,
    #                     args.patch_method,
    #                     args.use_attn)
 
    mapping = {256:240,512:512,1024:1000,2048:1920,4096:4375,16384:15360,65536:64000}


    args.nb_code = mapping[args.nb_code]#net.vqvae.quantizer.codebook_size
    config = LLaMAHFConfig.from_name(args.pretrained_llama)
    config.block_size = args.block_size
    config.vocab_size = args.nb_code + 2
    config.clip_dim = args.clip_dim
    
    # if args.use_moe:
    #     config.n_experts = args.n_experts
    #     config.top_k = args.top_k
    #     config.norm_topk_prob = args.norm_topk_prob

    config.tie_weights = args.tie_weights
    print(config)
    trans_encoder = LLaMAHF(config) # , args.use_qkNorm, args.use_moe)

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info(f'Trans encoder total parameters: {total_params:,}')

    print ('loading checkpoint from {}'.format(args.resume_pth))
    
    # ckpt = torch.load(args.resume_pth, map_location='cpu')['net']
    # ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    # net.load_state_dict(ckpt, strict=True)
    # net.eval()
    # net.to(comp_device)

    nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
    
    save_path1 = os.path.join(args.out_dir, f'net_last.pth')
    if os.path.isfile(save_path1):
        args.resume_trans = save_path1
    else:
        args.resume_trans = None
    text_encoder = torch.compile(text_encoder)

    trans_encoder.lm_head = torch.compile(trans_encoder.lm_head)
    trans_encoder.transformer.wte = torch.compile(trans_encoder.transformer.wte)
    trans_encoder.transformer.ln_f = torch.compile(trans_encoder.transformer.ln_f)
    for i, block in enumerate(trans_encoder.transformer.h):
        trans_encoder.transformer.h[i].rms_1 = torch.compile(block.rms_1)
        trans_encoder.transformer.h[i].rms_2 = torch.compile(block.rms_2)
        trans_encoder.transformer.h[i].mlp = torch.compile(block.mlp)
        trans_encoder.transformer.h[i].attn.c_attn = torch.compile(block.attn.c_attn)
        trans_encoder.transformer.h[i].attn.c_proj = torch.compile(block.attn.c_proj)
     
    if args.resume_trans is not None:
        print ('loading transformer checkpoint from {}'.format(args.resume_trans))
        state_dict = torch.load(args.resume_trans, map_location='cpu')
        ckpt = state_dict['trans']
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        trans_encoder.load_state_dict(ckpt, strict=True)
        nb_iter = state_dict['nb_iter']
        print(f'loading transformer checkpoint from {args.resume_trans}, nb_iter: {nb_iter}')
        nb_iter = nb_iter + 1
    else:
        nb_iter = 0
        
    trans_encoder.train()
    trans_encoder.to(comp_device)
    

    if args.mixed_precision == 'fp16':
        trans_encoder = trans_encoder.half()
    elif args.mixed_precision == 'bf16':
        trans_encoder = trans_encoder.bfloat16()

    ##### ---- Optimizer & Scheduler ---- #####
    if args.mixed_precision == 'bf16':
        eps = 1e-06
    else:
        eps = 1e-08

    optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer, eps)
    if args.lr_scheduler_type == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
    elif args.lr_scheduler_type == 'CosineDecayScheduler':
        # leawrning rate warm up and then cosine decay # stupid
        print(3*args.total_iter//100//args.gradient_accumulation_steps,'IT')
        scheduler = WarmupCosineDecayScheduler(optimizer, 10*args.total_iter//100//args.gradient_accumulation_steps, args.total_iter//args.gradient_accumulation_steps, resume_trans=args.resume_trans)
    elif args.lr_scheduler_type == 'ConstantScheduler':
        scheduler = WarmupConstantScheduler(optimizer, 10*args.total_iter//100//args.gradient_accumulation_steps, args.total_iter//args.gradient_accumulation_steps, resume_trans=args.resume_trans)
    else:
        raise ValueError(f'Unknown learning rate scheduler: {args.lr_scheduler}')
    
    if args.resume_trans is not None:
        # unwrapped_optimizer = accelerator.unwrap_model(optimizer)
        # unwrapped_scheduler = accelerator.unwrap_model(scheduler)
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['scheduler'])

    right_num = 0
    nb_sample_train = 0
    
    ##### ---- get code ---- #####
    args.vq_dir = os.path.join("./dataset/HandX", f'{args.vq_name}')
    args.prob_dir = os.path.join("./dataset/HandX", f'{args.vq_name}' + '_prob.npy')

    print("Start Training!")
    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=args.nb_code+1)#,label_smoothing=0.05)
    train_loader = dataset_TM_train_motionmillion.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name, args.train_split,  args.text_encode, args.text_sum_way, motion_type=args.motion_type, text_type=args.text_type, version=args.version, unit_length=2**args.down_t, debug=args.debug, num_workers=args.num_workers)

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    text_encoder, trans_encoder, optimizer, train_loader = accelerator.prepare( text_encoder, trans_encoder, optimizer, train_loader)
    clip_model = (tokenizer,text_encoder)
    
    train_loader_iter = cycle(train_loader)
    if accelerator.is_main_process:
        progress_bar = tqdm(range(nb_iter, args.total_iter + 1))
    start_iter = nb_iter
    ##### ---- Training ---- #####
    for nb_iter in (range(start_iter, args.total_iter + 1)):
    # while nb_iter <= args.total_iter:
        if accelerator.is_main_process:
            progress_bar.update(1)
        if nb_iter % 3000 ==0 and accelerator.is_main_process:
            progress_bar.set_postfix(step=nb_iter)
        batch = next(train_loader_iter)
        
        # self.mot_pad_idx = codebook_size + 1    
        # with torch.no_grad():
        #     cap_inputs = clip_model[0](caption, padding=True, truncation=True, return_tensors="pt")
        #     y_mask = cap_inputs.attention_mask.to(device=comp_device) # 1,9
        #     print(y_mask.shape)
        #     feat_clip_text = clip_model[1](
        #         input_ids=cap_inputs.input_ids.to(comp_device), 
        #         attention_mask=cap_inputs.attention_mask.to(comp_device), output_hidden_states=False
        #     ).last_hidden_state

        #     cap_inputs = clip_model[0]('I am stupid pig', padding=True, truncation=True, return_tensors="pt")
        #     y_mask = cap_inputs.attention_mask.to(device=comp_device) # 1,9
        #     print(y_mask.shape)
        # print(type(caption),type(m_tokens_len),m_tokens_raw)
        # continue

        with accelerator.accumulate(trans_encoder):
            # forward pass and loss calculation
            # caption, m_tokens_raw,m_tokens_len = batch
            with torch.no_grad():
                caption, m_tokens_raw,m_tokens_len = batch
                feat_clip_text, y_mask, text_tokens_len = encode_text_batch(caption,args.text_encode,args.text_sum_way,clip_model=None,hf_tokenizer=clip_model[0],hf_model=clip_model[1],device=comp_device,max_text_length=320)
                m_tokens = build_motion_tokens_with_text_offset(m_tokens_raw,m_tokens_len,text_tokens_len,351,args.nb_code+1,args.nb_code,comp_device) # self.mot_end_idx = codebook_size          # end token id
            # clip_text, m_tokens, m_tokens_len, feat_clip_text, y_mask, text_tokens_len = batch
            if len(y_mask.shape) == 1:
                y_mask = y_mask.unsqueeze(1)
                feat_clip_text = feat_clip_text.unsqueeze(1)
                
            cls_pred, target = train_one_iter(feat_clip_text, m_tokens, m_tokens_len, y_mask, trans_encoder)
            bs = target.shape[0]  

            loss_cls = 0.0
            
            cls_pred = cls_pred[..., :-1, :].contiguous()
            target = target[..., 1:].contiguous().to(torch.int64)

            loss_cls = loss_ce(cls_pred.view(-1, cls_pred.shape[-1]), target.view(-1))

            probs = torch.softmax(cls_pred.float(), dim=-1)
            if args.if_maxtest:
                _, cls_pred_index = torch.max(probs, dim=-1)
            else:
                dist = Categorical(probs)
                cls_pred_index = dist.sample()
            token_mask = (target != args.nb_code+1)
            right_num += ((cls_pred_index == target) & token_mask).sum().item()
            nb_sample_train += token_mask.sum().item()

            optimizer.zero_grad()
            accelerator.backward(loss_cls)

            # only on the last gradient accumulation step, execute the optimizer step
            if accelerator.sync_gradients:
                optimizer.step()
                if args.lr_scheduler_type == 'CosineDecayScheduler' or args.lr_scheduler_type == 'ConstantScheduler':
                    scheduler.step(nb_iter//args.gradient_accumulation_steps)
                else:
                    scheduler.step()

        avg_loss_cls = avg_loss_cls + loss_cls.item()
        
        if accelerator.is_main_process:
            lr = optimizer.param_groups[0]['lr']
            dict2={'lr':lr}
        # for tag, value in logs.items():
        #     dict2['Train/%s'%tag] = value / log_every
            # self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
            # mean_loss[tag] = value / self.opt.log_every
            accelerator.log(dict2, step=nb_iter)

            # writer.add_scalar('./LR/train', lr, nb_iter//args.gradient_accumulation_steps)

        nb_iter += 1
        
        if (nb_iter-1) % args.gradient_accumulation_steps != 0:
            continue
        
        actual_nb_iter = (nb_iter-1)//args.gradient_accumulation_steps + 1
        if actual_nb_iter % args.print_iter ==  0 :
            if accelerator.is_main_process: 
                avg_loss_cls = avg_loss_cls / args.print_iter
                avg_acc = right_num * 100 / nb_sample_train
                dict2 = {}
                dict2['Train/Loss_cls'] = avg_loss_cls
                dict2['Train/Acc'] = avg_acc
                accelerator.log(dict2, step=actual_nb_iter)
                # writer.add_scalar('./Loss/train', avg_loss_cls, actual_nb_iter)
                # writer.add_scalar('./ACC/train', avg_acc, actual_nb_iter)
                msg = f"Train. Iter {actual_nb_iter} : LR. {lr:.6f}, Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
                logger.info(msg)
            avg_loss_cls = 0.
            right_num = 0
            nb_sample_train = 0
        
        accelerator.wait_for_everyone()
        if actual_nb_iter % args.save_iter == 0 and accelerator.is_main_process:
            save_dict = {
                'trans' : trans_encoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'nb_iter' : nb_iter,
                'actual_nb_iter' : actual_nb_iter
            }
            torch.save(save_dict, os.path.join(args.out_dir, f'net_{actual_nb_iter}.pth'))

        if actual_nb_iter % args.save_iter_last == 0 and accelerator.is_main_process:
            save_dict = {
                'trans' : trans_encoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'nb_iter' : nb_iter,
                'actual_nb_iter' : actual_nb_iter
            }
            torch.save(save_dict, os.path.join(args.out_dir, f'net_last.pth'))
            