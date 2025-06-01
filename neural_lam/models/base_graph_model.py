# Third-party
import torch

# First-party
from neural_lam.utils import utils as project_utils
from neural_lam.interaction_net import InteractionNet, PropagationNet
from neural_lam.models.ar_model import ARModel


class BaseGraphModel(ARModel):
    """
    Base (abstract) class for graph-based models building on
    the encode-process-decode idea.
    """

    def __init__(self, model_cfg, training_cfg, data_cfg): # Updated signature
        super().__init__(model_cfg, training_cfg, data_cfg) # Pass cfgs to parent

        # Load graph with static features
        # NOTE: (IMPORTANT!) mesh nodes MUST have the first
        # num_mesh_nodes indices,
        # Assuming graph name is in model_cfg, e.g., model_cfg.graph_name
        self.hierarchical, graph_ldict = project_utils.load_graph(model_cfg.graph_name)
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Specify dimensions of data
        self.num_mesh_nodes, _ = self.get_num_mesh()
        print(
            f"Loaded graph with {self.num_grid_nodes + self.num_mesh_nodes} "
            f"nodes ({self.num_grid_nodes} grid, {self.num_mesh_nodes} mesh)"
        )

        # grid_dim from data + static
        self.g2m_edges, g2m_dim = self.g2m_features.shape
        self.m2g_edges, m2g_dim = self.m2g_features.shape

        # Define sub-models
        # Feature embedders for grid
        # Assuming hidden_dim and hidden_layers are in model_cfg
        self.mlp_blueprint_end = [model_cfg.hidden_dim] * (model_cfg.hidden_layers + 1)
        self.grid_embedder = project_utils.make_mlp(
            [self.grid_dim] + self.mlp_blueprint_end
        )
        self.g2m_embedder = project_utils.make_mlp([g2m_dim] + self.mlp_blueprint_end)
        self.m2g_embedder = project_utils.make_mlp([m2g_dim] + self.mlp_blueprint_end)

        # GNNs
        # Assuming vertical_propnets is in model_cfg
        gnn_class = PropagationNet if model_cfg.vertical_propnets else InteractionNet
        # encoder
        self.g2m_gnn = gnn_class(
            self.g2m_edge_index,
            model_cfg.hidden_dim,
            hidden_layers=model_cfg.hidden_layers,
            update_edges=False,
        )
        self.encoding_grid_mlp = project_utils.make_mlp(
            [model_cfg.hidden_dim] + self.mlp_blueprint_end
        )

        # decoder
        self.m2g_gnn = gnn_class(
            self.m2g_edge_index,
            model_cfg.hidden_dim,
            hidden_layers=model_cfg.hidden_layers,
            update_edges=False,
        )

        # Output mapping (hidden_dim -> output_dim)
        self.output_map = project_utils.make_mlp(
            [model_cfg.hidden_dim] * (model_cfg.hidden_layers + 1)
            + [self.grid_output_dim],
            layer_norm=False,
        )  # No layer norm on this one

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        raise NotImplementedError("get_num_mesh not implemented")

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        Returns tensor of shape (num_mesh_nodes, d_h)
        """
        raise NotImplementedError("embedd_mesh_nodes not implemented")

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, num_mesh_nodes, d_h)
        Returns mesh_rep: (B, num_mesh_nodes, d_h)
        """
        raise NotImplementedError("process_step not implemented")

    def predict_step(self, prev_state, prev_prev_state, forcing):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        """
        batch_size = prev_state.shape[0]

        # Create full grid node features of shape (B, num_grid_nodes, grid_dim)
        grid_features = torch.cat(
            (
                prev_state,
                prev_prev_state,
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ),
            dim=-1,
        )

        # Embed all features
        grid_emb = self.grid_embedder(grid_features)  # (B, num_grid_nodes, d_h)
        g2m_emb = self.g2m_embedder(self.g2m_features)  # (M_g2m, d_h)
        m2g_emb = self.m2g_embedder(self.m2g_features)  # (M_m2g, d_h)
        mesh_emb = self.embedd_mesh_nodes()

        # Map from grid to mesh
        mesh_emb_expanded = self.expand_to_batch(
            mesh_emb, batch_size
        )  # (B, num_mesh_nodes, d_h)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)

        # This also splits representation into grid and mesh
        mesh_rep = self.g2m_gnn(
            grid_emb, mesh_emb_expanded, g2m_emb_expanded
        )  # (B, num_mesh_nodes, d_h)
        # Also MLP with residual for grid representation
        grid_rep = grid_emb + self.encoding_grid_mlp(
            grid_emb
        )  # (B, num_grid_nodes, d_h)

        # Run processor step
        mesh_rep = self.process_step(mesh_rep)

        # Map back from mesh to grid
        m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep, grid_rep, m2g_emb_expanded
        )  # (B, num_grid_nodes, d_h)

        # Map to output dimension, only for grid
        net_output = self.output_map(
            grid_rep
        )  # (B, num_grid_nodes, d_grid_out)

        if self.output_std:
            pred_delta_mean, pred_std_raw = net_output.chunk(
                2, dim=-1
            )  # both (B, num_grid_nodes, d_f)
            # Note: The predicted std. is not scaled in any way here
            # linter for some reason does not think softplus is callable
            # pylint: disable-next=not-callable
            pred_std = torch.nn.functional.softplus(pred_std_raw)
        else:
            pred_delta_mean = net_output
            pred_std = None

        # Rescale with one-step difference statistics
        rescaled_delta_mean = (
            pred_delta_mean * self.step_diff_std + self.step_diff_mean
        )

        # Residual connection for full state
        return prev_state + rescaled_delta_mean, pred_std
