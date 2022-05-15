from torch import nn
import torch
import math



class Normal_SelfAttention(nn.Module):


    '''
    input:  (batch,n,input_size(default:1))
    output: (batch,n,all_head_size(default:100))

    '''

    def __init__(self, num_attention_heads, input_size, hidden_size,qkdim):
        super(Normal_SelfAttention, self).__init__()
        if ((hidden_size % num_attention_heads) !=0) or ((qkdim % num_attention_heads) != 0):
            raise ValueError(
                "The hidden/qkdim size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size
        
        self.qkdim=qkdim 
        self.query = nn.Linear(input_size, self.qkdim).cuda()
        self.key = nn.Linear(input_size, self.qkdim).cuda()
        self.value = nn.Linear(input_size, self.all_head_size).cuda()


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_for_qk(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, int(self.qkdim/self.num_attention_heads))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores_for_qk(mixed_query_layer)
        key_layer = self.transpose_for_scores_for_qk(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # Only keep the attention value of n=0 and subsequent calculations to save gpu memory
        attention_probs=attention_probs[:,:,0,:].view(attention_probs.shape[0],attention_probs.shape[1],-1,attention_probs.shape[3])
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        #attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
  
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        #hidden_states = self.dense(context_layer)
        #hidden_states = self.out_dropout(hidden_states)
        #hidden_states = self.LayerNorm(hidden_states + input_tensor)
     
        #return hidden_states
        return context_layer

if __name__ == '__main__':
    test_input=torch.randn(8,1000,1).cuda()
    att=Normal_SelfAttention(num_attention_heads=1, input_size=1, hidden_size=4700,qkdim=10)
    print(att(test_input).shape)