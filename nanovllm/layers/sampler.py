import torch
from torch import nn


class Sampler(nn.Module):

    compile_methods = ("scale_logits", "sample_probs")

    def scale_logits(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        return logits.float() / temperatures.unsqueeze(dim=1)

    def sample_probs(self, probs: torch.Tensor) -> torch.Tensor:
        noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        return (probs / noise).argmax(dim=-1)

    def filter_logits(
        self,
        logits: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        min_ps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        vocab_size = sorted_logits.size(-1)

        # top_ks 规范化: 排除小于等于零的情况, 截断在 [1, vocab_size] 内
        token_ranks = torch.arange(vocab_size, device=logits.device).unsqueeze(0)
        top_ks = torch.where(top_ks <= 0, vocab_size, top_ks).clamp_(min=1, max=vocab_size)
        sorted_logits = sorted_logits.masked_fill(token_ranks >= top_ks.unsqueeze(1), -torch.inf)

        # top-p: 保留累计概率达到 top_p 的最小高概率前缀，屏蔽其余 token
        probs = torch.softmax(sorted_logits, dim=-1)
        top_ps = torch.where(top_ps >= 1.0, torch.full_like(top_ps, torch.inf), top_ps) # top_p 超过1则不屏蔽任何 token
        top_p_mask = probs.cumsum(dim=-1) > top_ps.unsqueeze(1)
        # mask 右移一位，保留第一个让累计概率超过 top_p 的 token, 每个序列至少留首 token 不被屏蔽
        top_p_mask[:, 1:] = top_p_mask[:, :-1].clone()
        top_p_mask[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(top_p_mask, -torch.inf)

        # min-p: 保留绝对概率大于 max_prob × min_p 的 token
        probs = torch.softmax(sorted_logits, dim=-1) # 做完 top_p 后 probs 需要更新
        min_p_thresholds = probs[:, :1] * min_ps.unsqueeze(1) # 保留概率 >= max_prob * min_p
        sorted_logits = sorted_logits.masked_fill(probs < min_p_thresholds, -torch.inf)
        return sorted_logits, sorted_indices

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        min_ps: torch.Tensor,
        needs_filter: bool = True,
    ):
        logits = self.scale_logits(logits, temperatures)
        sorted_indices: torch.Tensor | None = None
        if needs_filter:
            logits, sorted_indices = self.filter_logits(logits, top_ps, top_ks, min_ps)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = self.sample_probs(probs)
        if sorted_indices is not None:
            sample_tokens = sorted_indices.gather(1, sample_tokens.unsqueeze(1)).squeeze(1)
        return sample_tokens
