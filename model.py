"""SimpleLSTM extracted from the original project notebook; architecture unchanged."""
import torch
from torch import nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层，简化模型只使用LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # LSTM层前向传播
        # 初始隐状态和细胞状态为None，表示全零初始状态
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取LSTM输出的最后一个时间步
        lstm_out = lstm_out[:, -1, :]
        
        # 全连接层
        output = self.fc(lstm_out)
        return output
