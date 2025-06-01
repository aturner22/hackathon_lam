# Third-party
import torch_geometric as pyg

# First-party
from neural_lam.utils import utils as project_utils
from neural_lam.interaction_net import InteractionNet
from neural_lam.models.base_graph_model import BaseGraphModel


class GraphCast(BaseGraphModel):
    """
    Full graph-based model that can be used with different
    (non-hierarchical) graphs. Mainly based on GraphCast, but the model from
    Keisler (2022) is almost identical.
    """

    def __init__(self, model_cfg, training_cfg, data_cfg): # Updated signature
        super().__init__(model_cfg, training_cfg, data_cfg) # Pass cfgs to parent

        assert (
            not self.hierarchical
        ), "GraphCast does not use a hierarchical mesh graph"

        # grid_dim from data + static + batch_static
        mesh_dim = self.mesh_static_features.shape[1]
        m2m_edges, m2m_dim = self.m2m_features.shape
        print(
            f"Edges in subgraphs: m2m={m2m_edges}, g2m={self.g2m_edges}, "
            f"m2g={self.m2g_edges}"
        )

        # Define sub-models
        # Feature embedders for mesh
        self.mesh_embedder = project_utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
        self.m2m_embedder = project_utils.make_mlp([m2m_dim] + self.mlp_blueprint_end)

        # GNNs
        # processor
        # Parameters from model_cfg
        processor_nets = [
            InteractionNet(
                self.m2m_edge_index,
                model_cfg.hidden_dim,
                hidden_layers=model_cfg.hidden_layers,
                aggr=model_cfg.mesh_aggr,
            )
            for _ in range(model_cfg.processor_layers) # This was ARModel's processor_layers
        ]
        self.processor = pyg.nn.Sequential(
            "mesh_rep, edge_rep",
            [
                (net, "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep")
                for net in processor_nets
            ],
        )

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        return self.mesh_static_features.shape[0], 0

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        Returns tensor of shape (N_mesh, d_h)
        """
        return self.mesh_embedder(self.mesh_static_features)  # (N_mesh, d_h)

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, N_mesh, d_h)
        Returns mesh_rep: (B, N_mesh, d_h)
        """
        # Embed m2m here first
        batch_size = mesh_rep.shape[0]
        m2m_emb = self.m2m_embedder(self.m2m_features)  # (M_mesh, d_h)
        m2m_emb_expanded = self.expand_to_batch(
            m2m_emb, batch_size
        )  # (B, M_mesh, d_h)

        mesh_rep, _ = self.processor(
            mesh_rep, m2m_emb_expanded
        )  # (B, N_mesh, d_h)
        return mesh_rep
