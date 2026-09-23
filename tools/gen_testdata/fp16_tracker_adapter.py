"""Use the NumPy graph NN inside the existing CUDA DPVO tracker.

No checkpoint is loaded. ONNX initializers are the sole NN weight source.
CUDA DPVO still owns patch selection, correlation, geometry and BA.
"""
from __future__ import annotations


def make_tracker_network(reference):
    # Lazy imports keep graph-only tests usable on machines without CUDA DPVO.
    import torch
    from dpvo.net import Patchifier
    from dpvo import fastba

    def numpy(tensor):
        return tensor.detach().cpu().contiguous().numpy()

    class FeatureOutput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = None

        def forward(self, images):
            return self.value

    class GraphPatchifier(Patchifier):
        def __init__(self):
            # Reuse only the patch selection/geometry code, not random encoders.
            torch.nn.Module.__init__(self)
            self.patch_size = 3
            self.fnet = FeatureOutput()
            self.inet = FeatureOutput()

        def forward(self, images, *args, **kwargs):
            # Parent divides by four and samples in float32, then these three
            # feature buffers are rounded back to half at the storage boundary.
            fmap, imap = reference.feature(numpy(images.float()))
            self.fnet.value = torch.from_numpy(fmap).to(images.device).float()
            self.inet.value = torch.from_numpy(imap).to(images.device).float()
            try:
                with torch.cuda.amp.autocast(enabled=False):
                    out = list(super().forward(images.float(), *args, **kwargs))
                out[:3] = [v.half() for v in out[:3]]
                return tuple(out)
            finally:
                self.fnet.value = self.inet.value = None

    class GraphUpdate(torch.nn.Module):
        def forward(self, net, ctx, corr, flow, ii, jj, kk):
            if flow is not None:
                raise ValueError("Deployed update graph does not accept flow")
            ix, jx = fastba.neighbors(kk, jj)
            result = reference.update(*(numpy(v) for v in (net, ctx, corr, ii, jj, kk, ix, jx)))
            net_out, delta, weight = [torch.from_numpy(v).to(net.device) for v in result]
            return net_out, (delta, weight, None)

    class GraphNetwork(torch.nn.Module):
        DIM, RES, P = 384, 4, 3

        def __init__(self):
            super().__init__()
            self.patchify = GraphPatchifier()
            self.update = GraphUpdate()

    return GraphNetwork().eval()
