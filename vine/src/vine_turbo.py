import os
import sys, gc
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from huggingface_hub import PyTorchModelHubMixin
from vine.src.stega_encoder_decoder import ConditionAdaptor
from vine.src.model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd, download_url


class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.mode() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


def initialize_unet(rank, return_lora_module_names=False):
    unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
    unet.requires_grad_(False)
    unet.train()
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n: continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break
    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder, lora_alpha=rank)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder, lora_alpha=rank)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_modules_others, lora_alpha=rank)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet


def initialize_unet_no_lora(path="stabilityai/sd-turbo"):
    unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet")
    unet.requires_grad_(True)
    unet.train()
    return unet
    

def initialize_vae(rank=4, return_lora_module_names=False):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
    vae.requires_grad_(False)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()
    # add the skip connection convs
    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    l_vae_target_modules = ["conv1","conv2","conv_in", "conv_shortcut",
        "conv", "conv_out", "skip_conv_1", "skip_conv_2", "skip_conv_3", 
        "skip_conv_4", "to_k", "to_q", "to_v", "to_out.0",
    ]
    vae_lora_config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules)
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae
    
    
def initialize_vae_no_lora(path="stabilityai/sd-turbo"):
    vae = AutoencoderKL.from_pretrained(path, subfolder="vae")
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()
    # add the skip connection convs
    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True).cuda().requires_grad_(True)
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True).cuda().requires_grad_(True)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True).cuda().requires_grad_(True)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True).cuda().requires_grad_(True)
    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1

    return vae


class VINE_Turbo(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, ckpt_path=None, device='cuda'):
        super().__init__()
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", use_fast=False,)
        text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder")
        text_encoder.requires_grad_(False)
        text_encoder.to(device)

        fixed_a2b_tokens = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
        self.fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.unsqueeze(0).to(device))[0].detach()
        del text_encoder, tokenizer, fixed_a2b_tokens  # free up some memory
        gc.collect()
        torch.cuda.empty_cache()

        self.sec_encoder = ConditionAdaptor()
        self.unet = initialize_unet_no_lora()
        self.vae_a2b = initialize_vae_no_lora()
        self.vae_enc = VAE_encode(self.vae_a2b)
        self.vae_dec = VAE_decode(self.vae_a2b)
        self.sched = make_1step_sched(device)
        self.timesteps = torch.tensor([self.sched.config.num_train_timesteps - 1] * 1, device=device).long()
        
        if ckpt_path is not None:
            self.load_ckpt_from_state_dict(ckpt_path, device)
            
    def load_ckpt_from_state_dict(self, ckpt_path, device):
        self.sec_encoder.load_state_dict(torch.load(os.path.join(ckpt_path, 'ConditionAdaptor.pth')))
        self.sec_encoder.to(device)
        self.sec_encoder.eval()

        self.unet.load_state_dict(torch.load(os.path.join(ckpt_path, 'UNet2DConditionModel.pth')))
        self.unet.to(device)
        self.unet.requires_grad_(False)
        self.unet.eval()
        
        self.vae_a2b.load_state_dict(torch.load(os.path.join(ckpt_path, 'vae.pth')))
        self.vae_a2b.to(device)
        self.vae_a2b.requires_grad_(False)
        self.vae_a2b.eval()

    @staticmethod
    def get_traininable_params(unet=None, vae_a2b=None, vae_b2a=None):
        # add all unet parameters
        params_gen = []
        if unet is not None:
            params_gen = params_gen + list(unet.conv_in.parameters())
            unet.conv_in.requires_grad_(True)
            unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
            for n,p in unet.named_parameters():
                # if "lora" in n and "default" in n:
                #     assert p.requires_grad
                if p.requires_grad:
                    params_gen.append(p)
        
        # add all vae_a2b parameters
        if vae_a2b is not None:
            for n,p in vae_a2b.named_parameters():
                # if "lora" in n and "vae_skip" in n:
                #     assert p.requires_grad
                if p.requires_grad:
                    params_gen.append(p)
            params_gen = params_gen + list(vae_a2b.decoder.skip_conv_1.parameters())
            params_gen = params_gen + list(vae_a2b.decoder.skip_conv_2.parameters())
            params_gen = params_gen + list(vae_a2b.decoder.skip_conv_3.parameters())
            params_gen = params_gen + list(vae_a2b.decoder.skip_conv_4.parameters())

        # add all vae_b2a parameters
        if vae_b2a is not None:
            for n,p in vae_b2a.named_parameters():
                if "lora" in n and "vae_skip" in n:
                    assert p.requires_grad
                    params_gen.append(p)
            params_gen = params_gen + list(vae_b2a.decoder.skip_conv_1.parameters())
            params_gen = params_gen + list(vae_b2a.decoder.skip_conv_2.parameters())
            params_gen = params_gen + list(vae_b2a.decoder.skip_conv_3.parameters())
            params_gen = params_gen + list(vae_b2a.decoder.skip_conv_4.parameters())
            
        return params_gen

    def forward(self, x, secret=None, timesteps=None):
        if timesteps == None:
            timesteps = self.timesteps
        B = x.shape[0]
        x_sec = self.sec_encoder(secret, x)
        x_enc = self.vae_enc(x_sec, direction="a2b").to(x.dtype)
        model_pred = self.unet(x_enc, timesteps, encoder_hidden_states=self.fixed_a2b_emb_base,).sample.to(x.dtype)
        x_out = torch.stack([self.sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
        x_out_decoded = self.vae_dec(x_out, direction="a2b").to(x.dtype)
        return x_out_decoded