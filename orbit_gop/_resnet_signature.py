from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F
EPSILON=1e-12

def stable_margin(raw_output: Tensor, classes: Tensor) -> Tensor:
    mask = F.one_hot(classes, num_classes=raw_output.shape[1]).bool()
    competitors = raw_output.masked_fill(mask, -torch.inf)
    return raw_output.gather(1, classes[:, None]).squeeze(1) - torch.logsumexp(competitors, dim=1)



def _forward_view(model, normalized_view: Tensor, fixed_classes: Tensor | None):
    captured: dict[str, Tensor] = {}

    def hook(_module, _inputs, output):
        captured["layer3"] = output

    handle = model.layer3.register_forward_hook(hook)
    try:
        output = model(normalized_view)
    finally:
        handle.remove()
    layer3 = captured["layer3"]
    classes = output["probability"].argmax(dim=1) if fixed_classes is None else fixed_classes
    margin = stable_margin(output["raw_output"], classes)
    gradient = torch.autograd.grad(
        outputs=margin.sum(), inputs=layer3, create_graph=False, retain_graph=False
    )[0]
    energy = gradient.square().mean(dim=(-1, -2))
    trace = energy.sum(dim=1)
    signature = (energy + EPSILON) / (trace[:, None] + 256.0 * EPSILON)
    epsilon_ratio = (256.0 * EPSILON) / (trace + 256.0 * EPSILON)
    return {
        "raw": output["raw_output"].detach(),
        "probability": output["probability"].detach(),
        "uncertainty": output["uncertainty"].detach(),
        "strength": output["strength"].detach(),
        "pred": output["probability"].argmax(dim=1).detach(),
        "margin": margin.detach(),
        "layer3_gap": layer3.detach().mean(dim=(-1, -2)),
        "gradient": gradient.detach(),
        "gradient_norm": gradient.detach().flatten(1).norm(dim=1),
        "trace": trace.detach(),
        "signature": signature.detach(),
        "epsilon_ratio": epsilon_ratio.detach(),
        "evidence": output["evidence"].detach(),
        "alpha": output["alpha"].detach(),
    }, classes.detach()

