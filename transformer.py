import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self, dimension=768, n_classes=5):
        super(Classifier, self).__init__()
        self.n_classes = n_classes
        self.dimension = dimension
        self.classifier = nn.Linear(dimension, n_classes)

    def forward(self, x):
        x = self.classifier(x)
        return x

class BoxRegress(nn.Module):
    def __init__(self, dimension=768):
        super(BoxRegress, self).__init__()
        self.dimension = dimension
        self.boxes = MultilayerPerceptron(feature=[dimension, dimension, 4])

    def forward(self, x):
        x = self.boxes(x)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, embedding_size=768, patch_count = 197):
        super(PositionEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, patch_count, embedding_size), requires_grad=True) # (1, N, D)
        self.embedding_size = embedding_size

    def forward(self, x):
        return x + self.pos_embedding

class ClassEmbedding(nn.Module):
    def __init__(self, embedding_size=768):
        super(ClassEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, embedding_size), requires_grad=True)  # (1, 1, D)

    def forward(self, x):
        y = self.cls_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((y, x), dim=1)
        return x

class InputEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embedding_size=768):
        super(InputEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_count_one_side = image_size//patch_size
        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=embedding_size, kernel_size=self.patch_size, stride=self.patch_size)
        # self.cls_embedding = ClassEmbedding(embedding_size=embedding_size)
        # self.pos_embedding = PositionEmbedding(embedding_size=embedding_size, patch_count=self.patch_count_one_side*self.patch_count_one_side + 1)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_count_one_side*self.patch_count_one_side + 1, embedding_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))

    def forward(self, x):
        x = self.patch_embedding(x) # x.shape->(B,N,P,P)
        x = x.flatten(2).transpose(1, 2) # x.shape->(B,N,P2)->(B,P2,N)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embedding + x
        return x

class Attention(nn.Module):
    def __init__(self, input_size, dimension=768):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.extract_qkv = nn.Linear(self.input_size, self.input_size*3)
        self.scale = dimension**-0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x.shape->(B,N,D)
        B, N, D = x.shape
        x = x.transpose(1, 2) # x.shape->(B,D,N)
        qkv = self.extract_qkv(x)# x.shape->(B,D*3,N)
        qkv = qkv.reshape(B, 3, N, D).permute(1, 0, 2, 3)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        atte = Q@K.transpose(-2, -1)  # Q->(B,N,D) @ K.T->(B,D,N) ->(B,N,N)
        atte = self.softmax(atte*self.scale)
        x = atte@V  # atte->(B,N,N) @ V(B,N,D) x->(B,N,D)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dimension=768, heads=8):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.scale = (dimension//heads)**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dimension = dimension
        self.extract_qkv = nn.Linear(self.dimension, self.dimension * 3, bias=False)
        self.lin = nn.Linear(self.dimension, self.dimension)


    def forward(self, x):
        B, N, D = x.shape
        # x = x.transpose(1, 2)  # x.shape->(B,D,N)
        qkv = self.extract_qkv(x)  # x.shape->(B,D*3,N)
        qkv = qkv.reshape(B, N, 3, self.heads, D//self.heads).permute(2, 0, 3, 1, 4) # (3, B, H, N, D/H)
        Q, K, V = qkv[0], qkv[1], qkv[2] # Q K V->(B, H, N, D/H)
        atte = torch.matmul(Q, K.transpose(-2, -1)) # Q->(B, H, N, D/H) @ K.T(B, H, D/H, N) = atte->(B, H, N, N)
        atte = self.softmax(atte*self.scale)
        x = torch.matmul(atte,V) # atte->(B, H, N, N)@V->(B, H, N, D/H) = x->(B, H, N, D/H)
        x = x.reshape(B, N, D)
        # x = x.reshape(B, N, D)
        x = self.lin(x)
        return x

class MultilayerPerceptron(nn.Module):
    def __init__(self, feature=None, activation=None, drop_rate=0.2):
        super(MultilayerPerceptron, self).__init__()
        self.feature =[768, 768, 768] if feature is None else feature
        self.layers = nn.Sequential()
        self.activation = nn.GELU() if activation is None else activation
        for index in range(len(self.feature)-1):
            self.layers.append(nn.Linear(self.feature[index], self.feature[index+1]))
            self.layers.append(self.activation)
            self.layers.append(nn.Dropout(drop_rate))
        self.norm = nn.LayerNorm(self.feature[0])
    def forward(self, x):
        x = self.norm(x)
        x = self.layers(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, dimension=768, heads=8, drop_rate=0.2, activation=None):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(dimension, 8)
        self.MLP = MultilayerPerceptron(drop_rate=drop_rate, feature=[dimension, dimension*4, dimension], activation=activation)
        # self.feed_forward = FeedForward(dim=dimension, hidden_dim=dimension*4)
        self.norm1 = nn.LayerNorm(dimension)
        # self.dropout1 = nn.Dropout(drop_rate)
        self.norm2 = nn.LayerNorm(dimension)
        self.norm3 = nn.LayerNorm(dimension)

    def forward(self, x):
        x = self.attention(self.norm1(x)) + x
        x = self.MLP(x) + x
        return self.norm3(x)

class Decoder(nn.Module):
    def __init__(self, dimension=768, layers=8):
        super(Decoder, self).__init__()
        self.dimension = dimension
        self.layers = layers
        self.blocks = nn.Sequential()
        for index in range(layers):
            self.blocks.append(EncoderBlock(dimension=dimension))

    def forward(self, x):
        x = self.blocks(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, 
                 input_size=224,
                 hidden_size= 768,
                  num_layers=8, 
                  dimension=768, 
                  layer_count=4, 
                  patch_size=16):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_embedding = InputEmbedding(image_size=input_size, 
                                              patch_size=patch_size, 
                                              embedding_size=dimension)
        self.encoder = Decoder(dimension, layer_count)
        self.classifier = Classifier()
        self.softmax = nn.Softmax(dim=-1)
        self.to_latent = nn.Identity()
    

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.encoder(x)
        x = x[:,0]
        x = self.to_latent(x)
        x = self.classifier(x)
        # x = self.softmax(x)
        return x