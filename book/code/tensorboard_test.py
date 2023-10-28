"""Show model graph in tensorboard."""
import model as m
import torch
import training as t
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    dataloaders, data = t.create_dataloaders(
        batch_size=20,
        use_edge_features=True,
    )

    model = m.EdgeAttentionGCN(
        num_features=7,
        conv_dims=[256, 128, 54],
        activation=torch.nn.ReLU(),
        dropout=0.5,
    )

    writer = SummaryWriter("tb/EdgeAttention")

    # dataiter = iter(dataloaders[0])
    graph = data[0]["adj"]
    features = data[0]["node_features"]
    class_y = data[0]["class_y"]
    edge_features = data[0]["edge_features"]
    writer.add_graph(
        model,
        (
            features.unsqueeze(0),
            graph.unsqueeze(0),
            edge_features.unsqueeze(0),
        ),
    )
    writer.close()
