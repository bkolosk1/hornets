import itertools
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HorNetsArchitecture(nn.Module):
    def __init__(
        self,
        dim,
        outpt,
        num_rules=256,
        num_features=10,
        exp_param=4,
        feature_names=None,
        activation="polyclip",
        order=5,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.num_features = num_features
        self.num_tars = outpt
        self.feature_names = feature_names
        self.k = 2 * exp_param + 1
        self.num_rules = num_rules
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.comb_indices = []
        comb_possible = itertools.combinations(list(range(num_features)), order)
        for _ in range(num_rules):
            try:
                self.comb_indices.append(next(comb_possible))
            except StopIteration:
                break

        self.comb_space = torch.nn.Parameter(
            data=torch.randn(order, len(self.comb_indices)), requires_grad=True
        )
        self.out_linear = torch.nn.Linear(len(self.comb_indices), outpt)
        self.register_buffer("comb_scores", torch.zeros(len(self.comb_indices)))
        self.linW = torch.nn.Parameter(data=torch.ones(len(self.comb_indices)))
        self.linF = torch.nn.Parameter(data=torch.ones(num_features))
        self.activation = activation
        self.ract = torch.nn.ReLU()
        self.dp = torch.nn.Dropout(p=0.1)
        self.initHAttention = torch.nn.Parameter(data=torch.ones(num_features))
        self.initHAttention2 = torch.nn.Parameter(data=torch.ones(num_features))
        self.out_linear2 = torch.nn.Linear(num_features, outpt)

    def polyClip(self, x, hard=False):
        x = torch.clamp(torch.pow(x, self.k), -1, 1)
        if hard:
            x = torch.round(x)
        return x

    def get_route(self, x):
        sortex, indices = torch.sort(torch.unique(x), 0)
        if len(sortex) == 2 and sortex[0] == 0 and sortex[1] == 1:
            return 1
        return 0

    def _get_embedding(self, x, num_samples=None):
        if self.get_route(x) == 0:
            x = torch.nn.functional.normalize(x)
            x = x.view(-1, self.num_features)
            return self.polyClip(self.initHAttention) * x

        if num_samples is None:
            num_samples = len(self.comb_indices)

        comb_pred = torch.zeros(
            (x.shape[0], len(self.comb_indices)), device=x.device
        )
        cat_subspace = torch.randperm(
            len(self.comb_indices), device=x.device
        )[:num_samples]
        cat_mask = torch.zeros(len(self.comb_indices), device=x.device)
        cat_mask[cat_subspace] = 1

        for enx, combination in enumerate(self.comb_indices):
            if cat_mask[enx] == 1:
                comb_subspace = x[:, combination]
                params = self.comb_space[:, enx]
                if self.activation == "polyclip":
                    comb_pred[:, enx] = self.polyClip(
                        torch.matmul(comb_subspace, params)
                    )
                else:
                    comb_pred[:, enx] = self.ract(torch.matmul(comb_subspace, params))

        comb_pred = self.dp(comb_pred)
        return F.softmax(comb_pred, dim=1)

    def forward(self, x, num_samples=None):
        embedding = self._get_embedding(x, num_samples)

        if self.get_route(x) == 0:
            return self.out_linear2(embedding)

        self.comb_scores += torch.mean(embedding, axis=0).detach()
        out = self.out_linear(embedding)
        return F.log_softmax(out, dim=1)

    def embed(self, x, num_samples=None):
        return self._get_embedding(x, num_samples)

    def reset_comb_scores(self):
        self.comb_scores.zero_()

    def get_rules(self, top_k=3):
        scores = self.comb_scores.detach().cpu().numpy()
        sindices = np.argsort(scores)[::-1][:top_k]
        rules = []
        for j in sindices:
            features = [str(self.feature_names[x]) for x in list(self.comb_indices[j])]
            features = [x for x in features if "synth" not in x]
            score = float(scores[j])
            rules.append({"features": features, "score": score})
            print(f"Feature comb: {features}, score: {score}")
        return rules
