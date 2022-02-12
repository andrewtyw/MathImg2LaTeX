from torch import Tensor
import torch.nn as nn
from .positional_encoding import PositionalEncoding1D, PositionalEncoding2D
import torchvision
from torchvision import transforms
from .my_data_loader import get_loader
from data_preprocess.my_build_vocab import Vocabulary
import torch
import math
import pickle
from typing import Union


class ResNet_Transformer(nn.Module):
    def __init__(
            self,
            args,
            d_model: int,
            dim_feedforward: int,
            nhead: int,
            dropout: float,
            num_decoder_layers: int,
            max_output_len: int,
            sos_index: int,
            eos_index: int,
            pad_index: int,
            num_classes: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_output_len = max_output_len + 2
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.cuda_index = args.cuda_index

        #ResNet encoder
        # resnet50 = torchvision.models.resnet50(pretrained=False)
        # modules=list(resnet50.children())[:-4]
        # self.backbone = nn.Sequential(*modules)
        # self.bottleneck = nn.Conv2d(512, self.d_model, 1)
        # Encoder

        resnet = torchvision.models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        self.bottleneck = nn.Conv2d(256, self.d_model, 1)
        
        self.image_positional_encoder = PositionalEncoding2D(self.d_model)

        # Decoder
        self.embedding = nn.Embedding(num_classes, self.d_model)
        self.y_mask = generate_square_subsequent_mask(self.max_output_len)
        self.word_positional_encoder = PositionalEncoding1D(self.d_model, max_len=self.max_output_len)
        transformer_decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_decoder_layers)
        self.fc = nn.Linear(self.d_model, num_classes)

        # It is empirically important to initialize weights properly
        if self.training:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)

        nn.init.kaiming_normal_(
            self.bottleneck.weight.data,
            a=0,
            mode="fan_out",
            nonlinearity="relu",
        )
        if self.bottleneck.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.bottleneck.weight.data)
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(self.bottleneck.bias, -bound, bound)    

    def encode(self, x: Tensor) -> Tensor:
        """Encode inputs.

        Args:
            x: (B, C, _H, _W) 64，3，256，256

        Returns:
            (Sx, B, E)
        """
        # Resnet expects 3 channels but training images are in gray scale
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x)  # (B, RESNET_DIM, H, W); H = _H // 32, W = _W // 32  =>64,256,16,16
        x = self.bottleneck(x)  # (B, E, H, W) (64,128,16,16)
        x = self.image_positional_encoder(x)  # (B, E, H, W)
        x = x.flatten(start_dim=2)  # (B, E, H * W)
        x = x.permute(2, 0, 1)  # (Sx, B, E); Sx = H * W
        return x

    def decode(self, y: Tensor, encoded_x: Tensor, length=None) -> Tensor:
        """Decode encoded inputs with teacher-forcing.

        Args:
            encoded_x: (Sx, B, E)
            y: (B, Sy) with elements in (0, num_classes - 1)

        Returns:
            (Sy, B, num_classes) logits
        """
        y = y.permute(1, 0)  # (Sy, B)
        y = self.embedding(y) * math.sqrt(self.d_model)  # (Sy, B, E)
        y = self.word_positional_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(encoded_x)  # (Sy, Sy)
        if length is not None:
            padding_mask = generate_padding_mask(length)[:, :y_mask.shape[1]]
            padding_mask = padding_mask.to(torch.device("cuda:{}".format(self.cuda_index) if \
                torch.cuda.is_available() else "cpu"))  # !!!!!!!!改index
        else:
            padding_mask = None
        output = self.transformer_decoder(y, encoded_x, tgt_mask=y_mask,
                                          tgt_key_padding_mask=padding_mask)  # (Sy, B, E)
        output = self.fc(output)  # (Sy, B, num_classes)
        return output

    def forward(self, x: Tensor, y: Tensor,length=None) -> Tensor:
        """Forward pass.

        Args:
            x: (B, _E, _H, _W)
            y: (B, Sy) with elements in (0, num_classes - 1)

        Returns:
            (B, num_classes, Sy) logits
        """
        encoded_x = self.encode(x)  # in: (bs,n_channel,w,h) out:(h*w,bs,E)
        output = self.decode(y, encoded_x,length)  # (Sy, B, num_classes)
        output = output.permute(1, 2, 0)  # (B, num_classes, Sy)
        return output

    def predict(self, x: Tensor) -> Tensor:
        """Make predctions at inference time.

            Args:
                x: (B, C, H, W). Input images.

            Returns:
                (B, max_output_len) with elements in (0, num_classes - 1).
            """
        B = x.shape[0]  # batch size
        S = self.max_output_len  # 可能的输出的LaTeX公式长度

        encoded_x = self.encode(x)  # (Sx, B, E)  对图像编码

        output_indices = torch.full((B, S), self.pad_index).type_as(x).long()
        output_indices[:, 0] = self.sos_index

        has_ended = torch.full((B,), False).bool()
        # print("output_indices 0")
        # print(output_indices)

        for Sy in range(1, S):
            y = output_indices[:, :Sy]  # (B, Sy)
            logits = self.decode(y, encoded_x)  # (Sy, B, num_classes)
            # Select the token with the highest conditional probability
            output = torch.argmax(logits, dim=-1)  # (Sy, B)
            output_indices[:, Sy] = output[-1:]  # Set the last output token

            # print("output_indices", Sy)
            # print(output_indices)

            # Early stopping of prediction loop to speed up prediction
            has_ended |= (output_indices[:, Sy] == self.eos_index).type_as(has_ended)
            if torch.all(has_ended):
                break

        # Set all tokens after end token to be padding
        eos_positions = find_first(output_indices, self.eos_index)
        for i in range(B):
            j = int(eos_positions[i].item()) + 1
            output_indices[i, j:] = self.pad_index
        # print("final output_indices")
        # print(output_indices)
        # print(output_indices.shape)
        return output_indices


def generate_square_subsequent_mask(size: int) -> Tensor:
    """Generate a triangular (size, size) mask."""
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def generate_padding_mask(length) -> Tensor:
    """
    给定一个batch数据的length， 返回padding_mask.
    返回的Tensor为True的地方表示是padding(0)

    length: list
    return: torch.Tensor

    Usage:
        >>> generate_padding_mask([3,1,2])
        tensor([[False, False, False],
                [False,  True,  True],
                [False, False,  True]])
    """
    bs = len(length)
    max_len = max(length)
    mask = torch.full((bs, max_len), 1)
    for i, l in enumerate(length):
        mask[i, :l] = 0
    mask = mask.bool()

    return mask


def find_first(x: Tensor, element: Union[int, float], dim: int = 1) -> Tensor:
    """Find the first occurence of element in x along a given dimension.

    Args:
        x: The input tensor to be searched.
        element: The number to look for.
        dim: The dimension to reduce.

    Returns:
        Indices of the first occurence of the element in x. If not found, return the
        length of x along dim.

    Usage:
        >>> first_element(Tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
        tensor([2, 1, 3])

    Reference:
        https://discuss.pytorch.org/t/first-nonzero-index/24769/9

        I fixed an edge case where the element we are looking for is at index 0. The
        original algorithm will return the length of x instead of 0.
    """
    mask = x == element
    found, indices = ((mask.cumsum(dim) == 1) & mask).max(dim)
    indices[(~found) & (indices == 0)] = x.shape[dim]
    return indices


if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 有人知道为什么要强硬变成3通道吗？
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])  # 修改的位置

    labels_lst_file_path = "../data/im2latex_formulas.norm.lst"
    train_images_lst_file_path = "../data/im2latex_train_filter.lst"
    images_dir_path = "../data/math_formula_images_grey_no_chinese_resized/"
    train_batch_size = 2
    vocab_pkl_file_path = "../data/vocab.pkl"

    train_data_loader = get_loader(labels_lst_file_path=labels_lst_file_path,
                                   images_lst_file_path=train_images_lst_file_path,
                                   images_dir_path=images_dir_path, batch_size=train_batch_size,
                                   vocab_pkl_file_path=vocab_pkl_file_path, transform=transform)

    with open(vocab_pkl_file_path, 'rb') as f:
        vocab = pickle.load(f)

    num_classes = len(vocab.word2idx)

    model = ResNet_Transformer(d_model=128, dim_feedforward=256,
                               nhead=4, dropout=0.3, num_decoder_layers=3,
                               max_output_len=150, sos_index=vocab('<start>'),
                               eos_index=vocab('<end>'), pad_index=vocab('<pad>'), num_classes=num_classes).to(device)

    for i, (images, labels, length) in enumerate(train_data_loader):
        if (i >= 1):
            break
        images = images.to(device)
        # labels = labels.to(device)
        # features = model.encode(images)
        output = model.forward(images, labels)
        pred = model.predict(images)
        # print(output.shape)
        # print(output)

    pred = pred.cpu().numpy()
    sample_labels = []
    for label_id in pred[0]:
        print(label_id)
        label = vocab.idx2word[label_id]
        sample_labels.append(label)
        if (label == '<end>'):
            break
    sentence = ' '.join(sample_labels)
    print(sentence)

'''
something to fix:
1, maybe we can try to convert our dataset to coco form but it maybe troublesome
2, the sorting according to file name is kind of strange

something question:
1, d_model means the depth of the conv2d?  tianyin: represents the dimension of word-embedding
'''
