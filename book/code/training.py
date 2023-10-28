"""Training loop and evaluation functions for the MUTAG dataset."""

import time
from os import environ

import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from utils import LOG as logger

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WANDB_MODE = "disabled" if not WANDB_AVAILABLE else "online"
# WANDB_MODE = "disabled"
USE_DETERMINISTIC_ALGORITHMS = True
SEED = 42
environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False

torch.manual_seed(SEED)
np.random.seed(SEED)

##################
#  DATA LOADING  #
##################

torch.use_deterministic_algorithms(
    USE_DETERMINISTIC_ALGORITHMS, warn_only=True
)


class MutagDataset(torch.utils.data.Dataset):
    """Custom Dataset for the MUTAG dataset."""

    def __init__(self, data):
        """Creates a dataset from the MUTAG dataset."""
        self.adj = data["adj"]
        self.node_features = data["node_features"]
        self.edge_features = data["edge_features"]
        self.class_y = data["class_y"]
        self.num_nodes = data["num_nodes"]

    def __getitem__(self, idx):
        """Returns the idx-th sample from the dataset."""
        return {
            "adj": self.adj[idx],
            "class_y": self.class_y[idx],
            "num_nodes": self.num_nodes[idx],
            "node_features": self.node_features[idx],
            "edge_features": self.edge_features[idx],
        }

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.adj)


def pad_adjacency_matrices(adj, max_num_nodes):
    """Expands all adjacency matrices to a constant size for all graphs."""
    expanded_adj = np.zeros((len(adj), max_num_nodes, max_num_nodes))
    for i, a in enumerate(adj):
        expanded_adj[i, : a.shape[0], : a.shape[1]] = a
    return expanded_adj


def pad_edge_features_matrix(edge_features, max_num_nodes):
    """Expands all edge features to a constant size for all graphs."""
    expanded_edge_features = np.zeros(
        (len(edge_features), max_num_nodes, max_num_nodes, 4)
    )
    for i, f in enumerate(edge_features):
        expanded_edge_features[i, : f.shape[0], : f.shape[1], :] = f
    return expanded_edge_features


def pad_features(nodes_features, max_num_nodes, feat_dim=7):
    """Expands all node features to a constant size for all graphs."""
    expanded_features = np.zeros(
        (len(nodes_features), max_num_nodes, feat_dim)
    )
    for i, f in enumerate(nodes_features):
        expanded_features[i, : len(f), :] = f
    return expanded_features


def pad_edges(edges_feat, max_num_edges, feat_dim=4):
    """Expands all edge features to a constant size for all graphs."""
    logger.debug(f"edges_feat: {len(edges_feat)}")
    logger.debug(f"edges_feat[0]: {len(edges_feat[0])}")
    expanded_edges = np.zeros((len(edges_feat), max_num_edges, feat_dim))
    for i, f in enumerate(edges_feat):
        expanded_edges[i, : len(f), :] = f
    return expanded_edges


def create_edge_feat_matrix(adj, edge_feature, num_nodes, num_features=4):
    """Creates an array of shape num_nodes x num_nodes x num_edge_features.

    Uses the adjacency matrix to place the edge features in the correct place.
    """
    edge_feats_array = torch.zeros((num_nodes, num_nodes, num_features))
    for i, a in enumerate(adj):
        for j, e in enumerate(a):
            edge_feats_array[i, e[0], :] = edge_feature[i][j]


def create_dataset_dict(add_edge_features=False):
    """Create a dictionary containing the data."""
    data = utils.load_dataset_mutag("./data/mutag.pickle")["train"]

    max_nodes = np.max(np.array(data["num_nodes"]))
    logger.debug(f"max_nodes: {max_nodes}")
    max_edges = np.max(np.array([len(edge) for edge in data["edge_attr"]]))
    logger.debug(f"max_edges: {max_edges}")
    adjacencies = [
        utils.convert_edges_to_adjacency(edge) for edge in data["edge_index"]
    ]
    if add_edge_features:
        edge_attrs = [np.array(d) for d in data["edge_attr"]]
        edge_feats = [
            utils.convert_edges_to_adjacency(edge, edge_feat)
            for edge, edge_feat in zip(data["edge_index"], edge_attrs)
        ]
        expanded_edge_features = torch.Tensor(
            pad_edge_features_matrix(edge_feats, max_nodes)
        )
        logger.debug(f"expanded_edge_features: {expanded_edge_features.shape}")
    else:
        expanded_edge_features = torch.Tensor(
            pad_edges(data["edge_attr"], max_edges)
        )

    logger.debug(f"adjacencies: {len(adjacencies)}")
    expanded_adj = torch.Tensor(pad_adjacency_matrices(adjacencies, max_nodes))
    logger.debug(f"expanded_adj: {expanded_adj.shape}")
    expanded_node_features = torch.Tensor(
        pad_features(data["node_feat"], max_nodes)
    )
    logger.debug(f"expanded_node_features: {expanded_node_features.shape}")
    y = torch.Tensor(data["y"]).squeeze().float()
    logger.debug(f"y: {y.shape}")
    num_nodes = torch.Tensor(data["num_nodes"])
    logger.debug(f"num_nodes: {num_nodes.shape}")

    # consistency check
    for adj, feat in zip(
        expanded_adj,
        expanded_node_features,
    ):
        assert adj.shape[0] == feat.shape[0]
        assert adj.shape == (max_nodes, max_nodes)
        assert feat.shape[1] == 7
        # assert ed.shape[0] == max_nodes
        # assert ed.shape[2] == 4
    # if add_edge_features:
    #     edge_feats = create_edge_feat_matrix(expanded_adj, expanded_edge_features)
    #     # replace the 1s in the adjacency matrix with the argmax of the edge features
    #     adjacencies = np.argmax(edge_feats, axis=3)
    # else:
    adjacencies = expanded_adj

    return {
        "adj": adjacencies,
        "node_features": expanded_node_features,
        "edge_features": expanded_edge_features,
        "class_y": y,
        "num_nodes": num_nodes,
    }
    # data_dict["full_features"] = full_features
    # return data_dict


def create_dataloaders(
    batch_size=1, train_split=0.7, valid_split=0.15, use_edge_features=False
):
    """Create train/val/test DataLoaders for the MUTAG dataset.

    Note : train_split + valid_split should be less than 1. The remaining data will be used for testing.

    Args:
        batch_size (int): Batch size.
        train_split (float): Percentage of data to use for training.
        valid_split (float): Percentage of data to use for validation.
        use_edge_features (bool): Whether to use the full features or not. (Tries to incorporate edge features)

    Returns:
        tuple(torch.utils.data.DataLoader): Training, validation and test dataloaders.
        dict: Dictionary containing the data.
    """
    data = create_dataset_dict(add_edge_features=use_edge_features)
    dataset = MutagDataset(data)

    # Split the dataset into train/val/test sets
    test_split = 1 - train_split - valid_split
    if test_split < 0:
        raise ValueError(
            "The sum of train_split and valid_split should be less than 1."
        )

    train_id = int(len(dataset.adj) * train_split)
    valid_id = int(len(dataset.adj) * (train_split + valid_split))
    logger.info(f"Length of train set: {train_id}")
    logger.info(f"Length of validation set: {valid_id - train_id}")
    logger.info(f"Length of test set: {len(dataset.adj) - valid_id}")

    train_dataset = torch.utils.data.Subset(dataset, range(train_id))
    valid_dataset = torch.utils.data.Subset(dataset, range(train_id, valid_id))
    test_dataset = torch.utils.data.Subset(
        dataset, range(valid_id, len(dataset))
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return (train_dataloader, valid_dataloader, test_dataloader), dataset


##################
# EVAL FUNCTIONS #
##################


@torch.no_grad()
def get_train_accuracy(dataloader, model, device=DEFAULT_DEVICE):
    """Compute accuracy of the edge prediction on the training data.

    Hint: do not forget to mask the train edges.

    Args:
        dataloader (dict): Dictionary of tensors containing the data.
        model (nn.Module): Model to be tested.
        device (torch.device): Device to run the test on.

    Returns:
        tuple(float, float): Accuracy for positive edges, accuracy for negative edges (in %).
    """
    # Set the model to evaluation mode
    model.eval()

    preds = model(
        dataloader.dataset[:]["node_features"].to(device),
        dataloader.dataset[:]["adj"].to(device),
        dataloader.dataset[:]["edge_features"].to(device),
    ).cpu()
    ys = dataloader.dataset[:]["class_y"].cpu()

    preds = (torch.sigmoid(preds) > 0.9).float()

    score = (preds == ys).float().mean()
    return score * 100, preds, ys


@torch.no_grad()
def test(dataset, model, device=DEFAULT_DEVICE, return_preds=False):
    """Test the model's prediction on the positive and negative edges.

    Args:
        dataset (dict): Dataset containing the data.
        model (nn.Module): Model to be tested.
        device (torch.device): Device to run the test on.
        return_preds (bool): Whether to return the predictions and the ground truth.

    Returns:
        tuple(float, float): Area under ROC curve, and average precision metrics.
    """
    # Set the model to evaluation mode

    logger.debug(f"model device: {next(model.parameters()).device}")
    logger.debug(f"dataset.device: {dataset[:]['node_features'].device}")

    model.eval()
    preds = model(
        dataset[:]["node_features"].to(device),
        dataset[:]["adj"].to(device),
        dataset[:]["edge_features"].to(device),
    ).cpu()
    preds = torch.sigmoid(preds)
    ys = dataset[:]["class_y"].cpu()

    assert preds.shape == ys.shape

    if return_preds:
        return (
            roc_auc_score(ys, preds),
            average_precision_score(ys, preds),
            preds,
            ys,
        )

    return roc_auc_score(ys, preds), average_precision_score(ys, preds)


@torch.no_grad()
def plot_roc_curve(dataset, model, device=DEFAULT_DEVICE):
    """Plot the ROC curve given the positive and negative edges.

    Args:
        dataset (torch.utils.data.Dataset): Dataset containing the data.
        model (nn.Module): Model to be tested.
        device (torch.device): Device to run the test on.

    Returns:
        matplotlib.figure.Figure: The ROC curve figure.
    """
    # Set the model to evaluation mode
    model.eval()

    preds = model(
        dataset[:]["node_features"].to(device),
        dataset[:]["adj"].to(device),
        dataset[:]["edge_features"].to(device),
    ).cpu()
    preds = torch.sigmoid(preds)
    ys = dataset[:]["class_y"].cpu()

    # Compute the false positive rate and true positive rate
    fpr, tpr, _ = roc_curve(ys, preds)

    # Plot the ROC curve
    with plt.style.context("seaborn-v0_8-dark"):
        fig, ax = plt.subplots(1, 1, dpi=150, figsize=(4, 4))
        ax.plot(fpr, tpr, label="ROC curve\n(with threshold values)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        ax.grid()
        plt.show()
        return fig


def log_test_results(dataset, model, device=DEFAULT_DEVICE):
    """Sends the test results to wandb."""
    test_roc, test_ap, preds, ys = test(
        dataset, model, device, return_preds=True
    )
    print(f"Test ROC: {test_roc:.4f} - Test AP: {test_ap:.4f}")
    if WANDB_AVAILABLE:
        wandb.log({"test/test_roc": test_roc, "test/test_ap": test_ap})
        frp, tpr, _ = roc_curve(ys, preds)
        wandb.log(
            {
                "test/roc_curve": wandb.plot.line_series(
                    xs=frp,
                    ys=[tpr],
                    keys=["True Positive Rate"],
                    title="ROC Curve",
                    xname="False Positive Rate",
                )
            }
        )
        # wandb.log(
        #     {"conf_mat" : wandb.plot.confusion_matrix(
        #         preds=(preds > 0.5).float(),
        #         y_true=ys,
        #         class_names=["0", "1"],)
        #      })

        # cm = confusion_matrix(ys, preds > 0.5)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["neg", "pos"])
        # wandb.log({"conf_mat": disp.plot().figure_})


#################
# TRAINING LOOP #
#################


def train_loop(
    history,
    train_dataloader,
    val_dataloader,
    model,
    loss_fn,
    optimizer,
    epochs,
    device,
    use_scheduler=False,
    test_dataloader=None,
    use_edges=False,
):
    """Full training loop over the dataset for several epochs."""
    if WANDB_AVAILABLE:
        wandb.init(
            config={
                "epochs": epochs,
                "batch_size": train_dataloader.batch_size,
                "optimizer": optimizer.__class__.__name__,
                "lr": optimizer.defaults["lr"],
                "loss": loss_fn.__class__.__name__,
                "model": model,
                "pooling": model.pool_type,
                "norm": model.norm_type,
                "device": device,
                "seed": SEED,
                "train_amount": len(train_dataloader.dataset),
                "val_amount": len(val_dataloader.dataset),
                "test_amount": len(test_dataloader.dataset)
                if test_dataloader is not None
                else 0,
            },
            project="DL Biomed Homework 2 - v3",
            mode=WANDB_MODE,
        )
        wandb.watch(model, log_freq=100, log="all")

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.1,
            patience=100,
            verbose=True,
            cooldown=100,
        )
    for t in range(epochs):
        print(f"Epoch {t+1: 3d}/{epochs}:", end="")
        start_epoch = time.time()  # time the epoch
        epoch_loss = 0
        step = 0
        model.train()
        for batch in train_dataloader:
            step += 1
            adj = batch["adj"].to(device)
            features = batch["node_features"].to(device)
            y = batch["class_y"].to(device)

            logger.debug(f"adj: {adj.shape}")
            logger.debug(f"node_features: {features.shape}")
            # Training step
            optimizer.zero_grad()
            if not use_edges:
                out = model(features, adj)
            else:
                edge_features = batch["edge_features"].to(device)
                logger.debug(f"edge_features: {edge_features.shape}")
                logger.debug(f"features: {features.shape}")
                out = model(features, adj, edge_features)
            # out = torch.sigmoid(out)

            logger.debug(f"out: {out.shape}")
            logger.debug(f"y: {y.shape}")

            loss = loss_fn(
                out,
                y.float(),
            )

            loss.backward()
            optimizer.step()

            if WANDB_AVAILABLE:
                wandb.log({"train_loss": loss})

            train_loss = loss.detach().item()
            logger.debug(f"Batch loss: {train_loss:.4f}")
            epoch_loss += train_loss

        epoch_loss /= step
        print(f"Epoch loss: {epoch_loss:.4f}", end="")
        if WANDB_AVAILABLE:
            wandb.log({"epoch_loss": epoch_loss})

        # Training accuracy
        acc, _, _ = get_train_accuracy(train_dataloader, model)
        # acc, preds, ys = get_train_accuracy(train_dataloader, model, device)
        # return preds, ys
        print(f" - avg acc: {acc:.1f}%", end="")
        if WANDB_AVAILABLE:
            wandb.log({"train_acc": acc})

        # # Validation
        valid_roc, valid_ap = test(val_dataloader.dataset, model, device)
        print(f" - val-roc: {valid_roc:.4f} - val-ap: {valid_ap:.4f}", end="")
        if WANDB_AVAILABLE:
            wandb.log({"val/val_roc": valid_roc, "val/val_ap": valid_ap})

        if use_scheduler:
            scheduler.step(train_loss)

        # Log
        history["epoch"] += 1
        history["loss"].append(epoch_loss)
        history["acc"].append(acc)
        history["val-roc"].append(valid_roc)
        history["val-ap"].append(valid_ap)
        print(f" ({time.time() - start_epoch:.1f}s/epoch)")
    logger.info("Done!")
    if WANDB_AVAILABLE:
        log_test_results(test_dataloader.dataset, model, device)
        wandb.finish()

    return history
