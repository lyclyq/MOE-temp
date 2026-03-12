from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_heads(dim: int, preferred: int) -> int:
    for h in [preferred, 12, 8, 6, 4, 3, 2, 1]:
        if dim % h == 0:
            return h
    return 1


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(dtype=x.dtype)
    s = (x * m).sum(dim=1)
    d = m.sum(dim=1).clamp_min(1.0)
    return s / d


def _masked_last(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Decode-only style pooling: take the last non-pad token state.
    idx = mask.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
    b = torch.arange(x.shape[0], device=x.device)
    return x[b, idx, :]


def _infer_hidden_dim(cfg: object, fallback: int) -> int:
    for key in ("hidden_size", "n_embd", "dim", "d_model"):
        v = getattr(cfg, key, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return int(fallback)


class TinyTransformerBackbone(nn.Module):
    """
    Lightweight transformer-style backbone with family presets:
    roberta / deberta / distilbert.
    """

    def __init__(
        self,
        *,
        family: str,
        vocab_size: int,
        hidden_dim: int,
        max_seq_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        fam = str(family).strip().lower()
        if fam not in {"roberta", "deberta", "distilbert"}:
            raise RuntimeError(f"unsupported backbone family={family!r}")

        if fam == "roberta":
            num_layers = 6
            num_heads = _pick_heads(hidden_dim, 8)
            ff_dim = hidden_dim * 4
        elif fam == "deberta":
            num_layers = 8
            num_heads = _pick_heads(hidden_dim, 8)
            ff_dim = hidden_dim * 4
        else:
            num_layers = 4
            num_heads = _pick_heads(hidden_dim, 8)
            ff_dim = hidden_dim * 4

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.in_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.hidden_dim = int(hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = int(input_ids.shape[0]), int(input_ids.shape[1])
        if seqlen > self.max_seq_len:
            raise RuntimeError(
                f"input sequence len {seqlen} > model.max_seq_len {self.max_seq_len}. "
                "Increase model.max_seq_len in config."
            )

        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.in_drop(x)

        key_padding_mask = attention_mask <= 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        pooled = _masked_mean(x, attention_mask)
        return pooled


class HFBackbone(nn.Module):
    """
    HuggingFace backbone families instantiated from configs (random init, no download).
    """

    def __init__(
        self,
        *,
        family: str,
        vocab_size: int,
        hidden_dim: int,
        max_seq_len: int,
        dropout: float,
        load_pretrained: bool = False,
        pretrained_name: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        fam = str(family).strip().lower()
        if fam not in {"roberta", "deberta", "distilbert", "gpt2"}:
            raise RuntimeError(f"unsupported backbone family={family!r}")
        self.family = fam
        try:
            from transformers import (
                DebertaV2Config,
                DebertaV2Model,
                DistilBertConfig,
                DistilBertModel,
                GPT2Config,
                GPT2Model,
                RobertaConfig,
                RobertaModel,
            )
        except Exception as e:
            raise RuntimeError(
                "transformers is required for model.backbone_backend=hf. "
                "Install with: pip install transformers"
            ) from e

        name_map = {
            "roberta": "roberta-base",
            "deberta": "microsoft/deberta-v3-base",
            "distilbert": "distilbert-base-uncased",
            "gpt2": "gpt2",
        }
        chosen_name = str(pretrained_name or name_map[fam])

        if bool(load_pretrained):
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(chosen_name, local_files_only=bool(local_files_only))
            self.hidden_dim = _infer_hidden_dim(getattr(self.model, "config", object()), int(hidden_dim))
            return

        heads = _pick_heads(hidden_dim, 8)
        ff_dim = hidden_dim * 4

        if fam == "roberta":
            cfg = RobertaConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_dim,
                num_hidden_layers=6,
                num_attention_heads=heads,
                intermediate_size=ff_dim,
                max_position_embeddings=max_seq_len + 2,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                type_vocab_size=1,
            )
            self.model = RobertaModel(cfg)
        elif fam == "deberta":
            cfg = DebertaV2Config(
                vocab_size=vocab_size,
                hidden_size=hidden_dim,
                num_hidden_layers=8,
                num_attention_heads=heads,
                intermediate_size=ff_dim,
                max_position_embeddings=max_seq_len + 2,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                type_vocab_size=0,
            )
            self.model = DebertaV2Model(cfg)
        elif fam == "distilbert":
            cfg = DistilBertConfig(
                vocab_size=vocab_size,
                max_position_embeddings=max_seq_len + 2,
                dim=hidden_dim,
                hidden_dim=ff_dim,
                n_layers=4,
                n_heads=heads,
                dropout=dropout,
                attention_dropout=dropout,
            )
            self.model = DistilBertModel(cfg)
        else:
            vocab_size = max(int(vocab_size), 50257)
            cfg = GPT2Config(
                vocab_size=vocab_size,
                n_positions=max_seq_len,
                n_ctx=max_seq_len,
                n_embd=hidden_dim,
                n_layer=6,
                n_head=_pick_heads(hidden_dim, 12),
                resid_pdrop=dropout,
                embd_pdrop=dropout,
                attn_pdrop=dropout,
            )
            self.model = GPT2Model(cfg)
        self.hidden_dim = _infer_hidden_dim(getattr(self.model, "config", object()), int(hidden_dim))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        if self.family == "gpt2":
            return _masked_last(last, attention_mask)
        return _masked_mean(last, attention_mask)


class FFNExpert(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, ffn_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class LoRAExpert(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise RuntimeError(f"lora rank must be > 0, got {rank}")
        self.drop = nn.Dropout(dropout)
        self.A = nn.Linear(hidden_dim, rank, bias=False)
        self.B = nn.Linear(rank, num_classes, bias=False)
        self.scale = float(alpha) / float(rank)

        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(self.drop(h))) * self.scale


class MoEClassifier(nn.Module):
    def __init__(self, model_cfg: Dict[str, object]):
        super().__init__()
        backbone = str(model_cfg.get("backbone", "roberta")).strip().lower()
        backend = str(model_cfg.get("backbone_backend", "tiny")).strip().lower()
        vocab_size = int(model_cfg.get("vocab_size", 30522))
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        max_seq_len = int(model_cfg.get("max_seq_len", 128))
        num_classes = int(model_cfg.get("num_classes", 2))
        num_experts = int(model_cfg.get("num_experts", 4))
        dropout = float(model_cfg.get("dropout", 0.1))

        if backbone == "gpt2" and backend != "hf":
            raise RuntimeError("gpt2 backbone requires model.backbone_backend=hf")

        if backend == "tiny":
            self.backbone = TinyTransformerBackbone(
                family=backbone,
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
        elif backend == "hf":
            hf_load_pretrained = bool(model_cfg.get("hf_load_pretrained", False))
            hf_pretrained_name = model_cfg.get("hf_pretrained_name")
            hf_local_files_only = bool(model_cfg.get("hf_local_files_only", False))
            self.backbone = HFBackbone(
                family=backbone,
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                max_seq_len=max_seq_len,
                dropout=dropout,
                load_pretrained=hf_load_pretrained,
                pretrained_name=(None if hf_pretrained_name is None else str(hf_pretrained_name)),
                local_files_only=hf_local_files_only,
            )
        else:
            raise RuntimeError(f"backbone_backend must be tiny/hf, got: {backend!r}")
        hidden_dim_eff = int(getattr(self.backbone, "hidden_dim", hidden_dim))
        self.router = nn.Linear(hidden_dim_eff, num_experts)
        self.routing_mode = str(model_cfg.get("routing_mode", "softmax")).strip().lower()
        self.top_k = int(model_cfg.get("top_k", 2))
        if self.routing_mode not in {"softmax", "topk"}:
            raise RuntimeError(f"routing_mode must be softmax/topk, got: {self.routing_mode!r}")
        if self.top_k <= 0:
            raise RuntimeError("top_k must be > 0")

        self.expert_type = str(model_cfg.get("expert_type", "ffn")).strip().lower()
        if self.expert_type not in {"ffn", "lora"}:
            raise RuntimeError(f"expert_type must be ffn/lora, got: {self.expert_type!r}")

        if self.expert_type == "ffn":
            ffn_hidden = int(model_cfg.get("ffn_hidden_dim", hidden_dim_eff * 4))
            self.base_head = None
            self.experts = nn.ModuleList(
                [FFNExpert(hidden_dim_eff, num_classes, ffn_hidden, dropout) for _ in range(num_experts)]
            )
        else:
            rank = int(model_cfg.get("lora_rank", 16))
            alpha = float(model_cfg.get("lora_alpha", 16.0))
            self.base_head = nn.Linear(hidden_dim_eff, num_classes)
            self.experts = nn.ModuleList(
                [LoRAExpert(hidden_dim_eff, num_classes, rank, alpha, dropout) for _ in range(num_experts)]
            )

    def route_probs_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        scores = self.router(h)
        if self.routing_mode == "softmax":
            return F.softmax(scores, dim=-1)

        k = min(int(self.top_k), int(scores.shape[-1]))
        top_vals, top_idx = torch.topk(scores, k=k, dim=-1)
        masked = torch.full_like(scores, float("-inf"))
        masked.scatter_(dim=-1, index=top_idx, src=top_vals)
        return F.softmax(masked, dim=-1)

    def _combine_logits(self, h: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        if self.expert_type == "ffn":
            logits_each = torch.stack([exp(h) for exp in self.experts], dim=1)
            return torch.einsum("bk,bkc->bc", probs, logits_each)

        assert self.base_head is not None
        base_logits = self.base_head(h)
        delta_each = torch.stack([exp(h) for exp in self.experts], dim=1)
        delta = torch.einsum("bk,bkc->bc", probs, delta_each)
        return base_logits + delta

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        probs = self.route_probs_from_hidden(h)
        logits = self._combine_logits(h, probs)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "router_probs": probs,
            "hidden": h,
        }
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out

    def router_params(self) -> List[torch.nn.Parameter]:
        return list(self.router.parameters())

    def task_params(self) -> List[torch.nn.Parameter]:
        # Task-loss channel updates all non-router params.
        router_set = {id(p) for p in self.router.parameters()}
        return [p for p in self.parameters() if id(p) not in router_set]

    def conflict_params(self) -> List[torch.nn.Parameter]:
        # Conflict summary is defined over expert branches.
        return [p for p in self.experts.parameters() if p.requires_grad]

    def trainable_params(self) -> List[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


# Backward-compatible alias used by early scaffold.
SimpleMoEClassifier = MoEClassifier
