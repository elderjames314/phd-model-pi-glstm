import torch
import torch.nn as nn


class BaseLSTM(nn.Module):

    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)


class PIGLSTMCell(nn.Module):

    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.weight_ih = nn.Linear(input_dim, 4 * hidden_dim)
        self.weight_hh = nn.Linear(hidden_dim, 4 * hidden_dim)
        # Fix: Input dimension is input_dim + hidden_dim (4 + 64 = 68)
        self.guard_layer = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, states):
        h_prev, c_prev = states
        gates = self.weight_ih(x) + self.weight_hh(h_prev)
        i, f, g, o = gates.chunk(4, 1)

        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        # Pass concatenated [x, h_next] (shape 68) to guard_layer
        xh = torch.cat([x, h_next], dim=-1)
        g_t = torch.sigmoid(self.guard_layer(xh))
        h_guarded = h_next * g_t

        return h_guarded, c_next


class PIGLSTM(nn.Module):

    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = PIGLSTMCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(
            batch_size, self.hidden_dim, device=x.device, dtype=x.dtype
        )
        c_t = torch.zeros(
            batch_size, self.hidden_dim, device=x.device, dtype=x.dtype
        )

        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))

        return self.fc(h_t)