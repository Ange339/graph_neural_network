import torch


class LossFunction:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def reconstruction_loss(self, preds, edge_label):
        loss_type = self.cfg.get('reconstruction_loss', "binary")
        if loss_type == "binary":
            return self.binary_loss(preds, edge_label)
        elif loss_type == "bpr":
            return self.bpr_loss(preds, edge_label)
        elif loss_type == "bce":
            return self.bce_loss(preds, edge_label)
        else:
            raise ValueError(f"Unknown reconstruction loss type: {loss_type}")


    def kl_loss(self, mu, logvar, beta=1.0):
        """
        Compute the Kullback-Leibler divergence loss.
        It measures the difference between the learned latent distribution and the prior distribution.
        Formula: D_KL(q(z|x) || p(z)) = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        **** ADD BETA WEIGHTING ****
        """
        if beta == 0:
            return torch.tensor(0.0, device=mu.device)
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_div * beta


    def binary_loss(self, preds, edge_label):
        """
        Compute the binary reconstruction loss.
        It encourages the model to assign high scores to positive edges and low scores to negative edges.
        Formula: -log(pos_preds) - log(1 - neg_preds)
        """
        EPS = 1e-15

        pos_mask = edge_label == 1
        pos_preds = preds[pos_mask]
        neg_preds = preds[~pos_mask]

        pos_loss = -torch.log(pos_preds + EPS).mean()
        neg_loss = -torch.log(1 - neg_preds + EPS).mean()
        loss = pos_loss + neg_loss
        return loss


    def bpr_loss(self, preds, edge_label):
        """
        Compute the Bayesian Personalized Ranking (BPR) loss.
        It encourages the model to rank positive edges higher than negative edges.
        Formula: -log(sigmoid(pos_preds - neg_preds))
        """
        EPS = 1e-15

        pos_mask = edge_label == 1
        pos_preds = preds[pos_mask]
        neg_preds = preds[~pos_mask]

        # Create all pairwise combinations of positive and negative predictions
        pos_preds_expanded = pos_preds.view(-1, 1)  # Shape: (num_pos, 1)
        neg_preds_expanded = neg_preds.view(1, -1)  # Shape: (1, num_neg)

        # Compute the difference between positive and negative predictions
        diff = pos_preds_expanded - neg_preds_expanded  # Shape: (num_pos, num_neg)

        # Apply the BPR loss formula
        loss = -torch.log(torch.sigmoid(diff) + EPS).mean()
        return loss


    def bce_loss(self, preds, edge_label):

        loss_fn = torch.nn.BCEWithLogitsLoss()
        return loss_fn(preds, edge_label)

    