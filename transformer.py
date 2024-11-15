import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, dimension=768, n_classes=4):
        super(Classifier, self).__init__()
        self.n_classes = n_classes
        self.dimension = dimension
        self.classifier = MultilayerPerceptron(feature=[dimension, dimension, n_classes])

    def forward(self, x):
        x = self.classifier(x)
        return x[:,0]


class BoxRegress(nn.Module):
    def __init__(self, dimension=768):
        super(BoxRegress, self).__init__()
        self.dimension = dimension
        self.boxes = MultilayerPerceptron(feature=[dimension, dimension, 4])

    def forward(self, x):
        x = self.boxes(x)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, embedding_size=768, patch_count = 64):
        super(PositionEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, patch_count, embedding_size), requires_grad=True) # (1, N, D)
        self.embedding_size = embedding_size

    def forward(self, x):
        return x + self.pos_embedding

class ClassEmbedding(nn.Module):
    def __init__(self, embedding_size=768):
        super(ClassEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, embedding_size), requires_grad=True)  # (1, 1, D)

    def forward(self, x):
        y = self.cls_embedding.expand(x.size(0), 1, self.embedding_size)
        x = torch.cat((y, x), dim=1)
        return x

class InputEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embedding_size=768):
        super(InputEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_count_one_side = image_size//patch_size
        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=embedding_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_embedding = ClassEmbedding(embedding_size=embedding_size)
        self.pos_embedding = PositionEmbedding(embedding_size=embedding_size, patch_count=self.patch_count_one_side*self.patch_count_one_side + 1)

    def forward(self, x):
        x = self.patch_embedding(x) # x.shape->(B,N,P,P)
        x = x.flatten(2).transpose(1, 2) # x.shape->(B,N,P2)->(B,P2,N)
        x = self.cls_embedding(x)
        x = self.pos_embedding(x)
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
        self.scale = dimension**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dimension = dimension
        self.extract_qkv = nn.Linear(self.dimension, self.dimension * 3)


    def forward(self, x):
        B, N, D = x.shape
        # x = x.transpose(1, 2)  # x.shape->(B,D,N)
        qkv = self.extract_qkv(x)  # x.shape->(B,D*3,N)
        qkv = qkv.reshape(B, N, 3, self.heads, D//self.heads).permute(2, 3, 0, 1, 4) # (3, H, B, N, D/H)
        Q, K, V = qkv[0], qkv[1], qkv[2] # Q K V->(H, B, N, D/H)
        atte = Q@K.transpose(-2, -1) # Q->(H, B, N, D/H) @ K.T(H, B, D/H, N) = atte->(H, B, N, N)
        atte = self.softmax(atte*self.scale)
        x = atte@V # atte->(H, B, N, N)@V->(H, B, N, D/H) = x->(H, B, N, D/H)
        x = x.permute(2, 3, 0, 1).reshape(B, N, D)
        return x

class MultilayerPerceptron(nn.Module):
    def __init__(self, feature=None, activation=None, drop_rate=0.2):
        super(MultilayerPerceptron, self).__init__()
        self.feature =[768, 768, 768] if feature is None else feature
        self.layers = nn.Sequential()
        self.activation = nn.ReLU() if activation is None else activation
        for index in range(len(self.feature)-1):

            self.layers.append(nn.Linear(self.feature[index], self.feature[index+1]))
            self.layers.append(self.activation)
            self.layers.append(nn.Dropout(drop_rate))

    def forward(self, x):
        x = self.layers(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, dimension=768, heads=8, drop_rate=0.2, activation=None):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(dimension=dimension, heads=heads)
        self.MLP = MultilayerPerceptron(drop_rate=drop_rate, feature=[dimension, dimension, dimension], activation=activation)
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)

    def forward(self, x):
        y = self.attention(x)
        y = self.norm1(y) + x
        y1 = self.MLP(y)
        y1 = self.norm2(y1) + y
        return y1

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
    def __init__(self, input_size=224, hidden_size= 768, num_layers=8, dimension=768, layer_count=8, patch_size=16):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_embedding = InputEmbedding(image_size=input_size, patch_size=patch_size, embedding_size=dimension)
        self.encoder = Decoder(dimension, layer_count)
        self.classifier = Classifier()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.input_embedding(x)
        x = self.encoder(x)
        x = self.classifier(x)
        x = self.softmax(x)
        return x