"""Utilities for Homework 2."""

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

#################
# Data loading  #
#################
LOG = logging.getLogger(__name__)
# LOG.setLevel(logging.DEBUG)


def _print_gradient_hook(model):
    for _name, layer in model.named_modules():
        layer.register_backward_hook(
            lambda module, grad_input, grad_output: print(grad_output)
        )


def squeeze_matrix(matrix):
    """Returns an adjacency matrix where all empty rows and columns are removed."""
    # Remove empty rows
    row_sums = matrix.sum(axis=1)
    non_empty_rows = np.where(row_sums > 0)[0]
    matrix = matrix[non_empty_rows, :]

    # Remove empty columns
    col_sums = matrix.sum(axis=0)
    non_empty_cols = np.where(col_sums > 0)[0]
    matrix = matrix[:, non_empty_cols]

    return matrix


def one_hot_encode_y(y):
    """One-hot encodes the labels.

    Args:
        y (int): Label.

    Returns:
        np.ndarray: One-hot encoded label.
    """
    if y == 1:
        return np.array([1, 0])
    return np.array([0, 1])


def plot_history(history):
    """Plot the training history of a model."""
    with plt.style.context("seaborn-v0_8-dark"):
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        # fig.patch.set_color("#14181e")
        for ax in axs:
            ax.set_xlabel("Epoch")
            # ax.set_facecolor("#14181e")
        axs[0].plot(history["loss"])
        axs[0].set_title("Training loss")
        col = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
        axs[1].plot(history["acc"], c=col)
        axs[1].set_title("Training accuracy")
        axs[1].axvline(
            x=np.argmax(history["acc"]),
            c=col,
            ls="--",
            label=f"Max accuracy: {np.max(history['acc']):.2f} @ {np.argmax(history['acc'])}",
        )
        axs[1].legend(loc="lower right")
        col = plt.rcParams["axes.prop_cycle"].by_key()["color"][2]
        axs[2].plot(history["val-roc"], c=col)
        axs[2].axvline(
            x=np.argmax(history["val-roc"]),
            c=col,
            ls="--",
            label=f"Max ROC-AUC: {np.max(history['val-roc']):.2f} @ {np.argmax(history['val-roc'])}",
        )
        axs[2].set_title("Validation ROC-AUC")
        axs[2].legend(loc="lower right")
        col = plt.rcParams["axes.prop_cycle"].by_key()["color"][3]
        axs[3].plot(history["val-ap"], c=col)
        axs[3].axvline(
            x=np.argmax(history["val-ap"]),
            c=col,
            ls="--",
            label=f"Max AP: {np.max(history['val-ap']):.2f} @ {np.argmax(history['val-ap'])}",
        )
        axs[3].set_title("Validation Average Precision")
        axs[3].legend(loc="lower right")
        plt.show()


def convert_edges_to_adjacency(edges, edge_features=None):
    """Convert an edge list to an adjacency matrix.

    Args:
        edges (List[list(int)]): List of edges. First dimension contains the id of the connected node, second dimension contains the nodes it is connected to.
        edge_features (List[list(int)]): Optional list of edge features. First dimension contains the id of the connected node, second dimension contains the edge features.

    Returns:
        np.ndarray: Adjacency matrix of shape (num_nodes, num_nodes).
    """
    num_nodes = len(np.unique(edges[0]))
    adjacency = np.zeros((num_nodes, num_nodes))
    edge_features_matrix = np.zeros((num_nodes, num_nodes, 4))
    for i, node_id in enumerate(edges[0]):
        adjacency[node_id, edges[1][i]] = 1
        if edge_features is not None:
            edge_features_matrix[node_id, edges[1][i]] = edge_features[i]
    if edge_features is not None:
        return np.array(edge_features_matrix)

    return adjacency


def load_dataset_mutag(data_path):
    """Loads the dataset from a pickle file if it exists, or downloads the dataset from HF."""
    data_path = Path(data_path)
    data_exists = data_path.resolve().exists()
    if data_exists:
        with data_path.open("rb") as f:
            data = pickle.load(f)
    else:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data = load_dataset("graphs-datasets/MUTAG")
        with data_path.open("wb") as f:
            pickle.dump(data, f)
    return data


def find_mislabeled_molecules(preds, ys, threshold=0.5):
    """Returns the indices of the molecules that are mislabeled."""
    preds = np.array(preds)
    ys = np.array(ys)
    thresh_ys = np.where(preds > threshold, 1, 0)
    return np.where(thresh_ys != ys)[0]


def draw_molecules(
    adjacency_matrices,
    node_features,
    edge_features,
    class_y,
    preds=None,
    ids=None,
    n_rows=4,
    n_cols=5,
    figsize=(10, 10),
):
    """Draws the molecules in a 4x5 grid."""
    with plt.style.context("seaborn-v0_8-dark"):
        # plt.rcParams["savefig.facecolor"] = (0.2, 0.24, 0.3, 0.0)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=200)
        # fig.patch.set_alpha(0)
        # fig.patch.set_color("#14181e")
        # for ax in axes.flatten():
        # ax.patch.set_facecolor("#14181e")
        # ax.patch.set_alpha(0)
        colors = []
        widths = []
        for i in range(len(adjacency_matrices)):
            colors.append(
                np.array([np.argmax(node) for node in node_features[i]])
            )
            widths.append(
                np.array([np.argmax(node) for node in edge_features[i]]) + 0.75
            )
            graph = nx.from_numpy_array(adjacency_matrices[i])
            ax = axes[i // n_cols, i % n_cols]
            nx.draw_kamada_kawai(
                graph,
                ax=ax,
                node_size=30,
                node_color=colors[i],
                # edge_color="white",
                cmap="tab10",
                width=widths[i],
            )
            ax.set_axis_on()
            ax.tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            mutag = (
                "Class 1, probably mutagenic"
                if class_y[i][0] == 1
                else "Class 2, probably non-mutagenic"
            )
            if preds is not None:
                mutag += (
                    "\nPredicted class 1"
                    if preds[i] > 0.5
                    else "\nPredicted class 2"
                )
            molecule_id = ids[i] if ids is not None else i + 1
            ax.set_xlabel(
                f"Molecule {molecule_id}\n ({mutag})", fontdict={"fontsize": 8}
            )
        plt.tight_layout()
        plt.show()
        return colors, widths


def show_preds_distribution(preds, ys, threshold=0.5):
    """Shows the distribution of the predictions and labels."""
    with plt.style.context("seaborn-v0_8-dark"):
        fig, ax = plt.subplots(dpi=150, figsize=(4, 4))
        # ax.set_facecolor("#14181e")
        sns.histplot(ys, alpha=1, ax=ax, label="Labels", color="skyblue")
        sns.histplot(
            preds, alpha=0.6, ax=ax, label="Predictions", color="purple"
        )
        plt.legend()
        plt.show()
        print(
            "This plot shows the distribution of the labels and predictions;\n"
            + "predictions are overlayed on top of the labels,\n"
            +"showing whether they are missing or surnumerous."
        )
        print(
            f"The labels are {int(ys.sum())} positive and {int(len(ys) - ys.sum())} negative."
        )
        print(
            f"The predictions are {(preds > threshold).sum()} positive and {(preds < threshold).sum()} negative."
        )


def plot_confusion_matrix(preds, ys, threshold=0.5):
    """Plots the confusion matrix."""
    with plt.style.context("seaborn-v0_8-dark"):
        cm = confusion_matrix(ys, preds > threshold)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["neg", "pos"]
        )
        disp.plot(cmap="RdPu")
        plt.show()


def draw_molecule_from_dict(
    list_data_dict,
    preds=None,
    mol_ids=None,
    n_rows=4,
    n_cols=5,
    figsize=(15, 15),
):
    """Draws using a dict from MutagDataset class as data input."""
    adjs = []
    nodes = []
    edges = []
    classes = []
    ids = []
    for i, data_dict in enumerate(list_data_dict):
        adj = data_dict["adj"].cpu().int().numpy()
        adjs.append(squeeze_matrix(adj))
        node = data_dict["node_features"].cpu().int().numpy()
        nodes.append(squeeze_matrix(node))
        edge = np.array(data_dict["edge_features"])  # .cpu().int().numpy()
        # convert N x N x P edge feature matrix back to edge list
        if len(edge.shape) == 3:
            edge = np.array(np.where(edge == 1)).T
        edges.append(squeeze_matrix(edge))
        y = data_dict["class_y"].cpu().int().numpy()
        y = one_hot_encode_y(y)
        classes.append(y)
        if mol_ids is not None:
            ids.append(mol_ids[i])

    draw_molecules(
        adjs,
        nodes,
        edges,
        classes,
        ids=ids,
        n_rows=n_rows,
        n_cols=n_cols,
        preds=preds,
        figsize=figsize,
    )
