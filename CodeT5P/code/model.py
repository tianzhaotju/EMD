import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss\


class CodeT5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.encoder.n_embd*2, 1024)
        self.out_proj = nn.Linear(1024, 2)

    def forward(self, features, **kwargs):
        x = features
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = CodeT5ClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, input_ids=None, labels=None, output_attentions=False):
        input_ids = input_ids.view(-1, self.args.code_length)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True,
                               output_attentions=output_attentions)
        eos_mask = input_ids.eq(self.tokenizer.eos_token_id)
        hidden_states = outputs['encoder_last_hidden_state']
        sequence_outputs = torch.cat((hidden_states[0][eos_mask[0],:].view(1, -1, hidden_states.size(-1))[:, -1, :], hidden_states[1][eos_mask[1],:].view(1, -1, hidden_states.size(-1))[:, -1, :]), dim=0)
        logits = self.classifier(sequence_outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            if output_attentions:
                return loss, prob, outputs.encoder_attentions
            else:
                return loss, prob, sequence_outputs
        else:
            return prob
    
