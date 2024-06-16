import torch
from torch import nn
class LSTMModel(nn.Module):
    def __init__(self, window_size, input_size,
                 hidden_dim, pred_len, num_layers, batch_size, device) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.input_size = input_size
        self.device = device
        self.lstm_encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers,
                                    batch_first=True).to(self.device)
        self.lstm_decoder = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers,
                                    batch_first=True).to(self.device)
        self.relu = nn.GELU()
        self.fc = nn.Linear(hidden_dim, input_size)
    def forward(self, src):
        src = torch.unsqueeze(src, -1)#展平
        _, decoder_hidden = self.lstm_encoder(src)
        cur_batch = src.shape[0]
        decoder_input = torch.zeros(cur_batch, 1, self.input_size).to(self.device)
        outputs = torch.zeros(self.pred_len, cur_batch, self.input_size).to(self.device)
        for t in range(self.pred_len):
            decoder_output, decoder_hidden = self.lstm_decoder(decoder_input, decoder_hidden)
            decoder_output = self.relu(decoder_output)
            decoder_input = self.fc(decoder_output)
            outputs[t] = torch.squeeze(decoder_input, dim=-2)
        return outputs
if __name__ == '__main__':#模型调用示例
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cpu")
    feature = 2#每个时间戳特征数
    timestep = 3#时间步长
    batch_size = 1#批次
    inputseq = torch.randn(timestep, feature).to(device)  # 模拟输入,生成batch批次的，序列长度为timestep，序列每个时间戳特征数为feature的随机序列
    hidden_dim = 5#lstm隐藏层
    num_layers = 1 #lstm层数
    window_size = 5
    input_size = 2#输入序列每个时间戳的特征数
    pred_len = 1#重构序列的个数

    model = LSTMModel(window_size, input_size, hidden_dim, pred_len, num_layers, batch_size, device)
    output = model(inputseq)#模型输出与输入相同，用于重构x'误差的计算
    ou = 1
    "output.size()=[1,3,2],inputseq.size=[3,2],故output.squezze"
