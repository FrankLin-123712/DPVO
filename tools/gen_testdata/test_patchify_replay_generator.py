#!/usr/bin/env python3
"""Test capture ownership/schema/hook cleanup without importing torch or CUDA."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from generate_patchify_replay_testdata import PatchifyRecorder, ReplayWriter, parse_args, snapshot


class Tensor:
    def __init__(self, array): self.array = array
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.array


class Encoder:
    def __init__(self, channels):
        self.hooks, self.calls = [], 0
        self.array = np.arange(channels*4*6, dtype=np.float32).reshape(1,1,channels,4,6) / 8
    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))
    def __call__(self, images):
        self.calls += 1
        output = Tensor(self.array)
        for hook in self.hooks: hook(self, (images,), output)
        return output


class Patchifier:
    patch_size = 3
    def __init__(self):
        self.fnet, self.inet = Encoder(128), Encoder(384)
        self.calls, self.fail = 0, False
    def forward(self, images, patches_per_image=2, disps=None, centroid_sel_strat="RANDOM", return_color=True):
        self.calls += 1
        fmap = self.fnet(images).array / 4
        imap = self.inet(images).array / 4
        if self.fail: raise RuntimeError("injected encoder failure")
        centers = [(1,1),(4,2)]
        gmap = np.stack([fmap[0,0,:,y-1:y+2,x-1:x+2] for x,y in centers])[None]
        ctx = np.stack([imap[0,0,:,y:y+1,x:x+1] for x,y in centers])[None]
        patches = np.array([[[[x+dx if c == 0 else y+dy if c == 1 else 1 for dx in (-1,0,1)]
                              for dy in (-1,0,1)] for c in range(3)] for x,y in centers], dtype=np.float32)[None]
        colors = np.stack([images.array[0,0,:,4*y+2,4*x+2] for x,y in centers])[None]
        return tuple(Tensor(a) for a in (fmap,gmap,ctx,patches,np.zeros(2,dtype=np.int64),colors))


class CaptureTests(unittest.TestCase):
    def test_actual_call_is_observed_once_and_snapshots_are_owned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patchifier = Patchifier()
            tracker = SimpleNamespace(network=SimpleNamespace(patchify=patchifier), n=0, counter=0)
            writer = ReplayWriter(root)
            recorder = PatchifyRecorder(tracker,writer)
            recorder.start_frame(3,Path("frame.png"))
            image = Tensor(np.zeros((1,1,3,16,24),dtype=np.float32))
            with recorder.installed():
                outputs = patchifier.forward(image,patches_per_image=2,return_color=True)
                outputs[3].array[:,:,2] = 999  # tracker mutates the returned tensor
                patchifier.fnet.array.fill(-999)
                image.array.fill(-999)
            self.assertEqual((patchifier.calls,patchifier.fnet.calls,patchifier.inet.calls),(1,1,1))
            self.assertNotIn("forward",patchifier.__dict__)
            self.assertEqual(patchifier.fnet.hooks,[])
            self.assertEqual(patchifier.inet.hooks,[])
            case = root / "case_0000"
            grid = np.fromfile(case / "golden_patches.bin",dtype="<f4").reshape(2,3,3,3)
            self.assertTrue(np.all(grid[:,2] == 1))
            self.assertTrue(np.all(np.fromfile(case / "image.bin",dtype="<f4") == 0))
            self.assertGreaterEqual(np.fromfile(case / "raw_fmap.bin",dtype="<f4").min(),0)
            meta = json.loads((case / "metadata.json").read_text())
            self.assertEqual(meta["input_frame_index"],3)
            self.assertEqual(meta["tensor_shapes"]["golden_imap"],[2,384,1,1])
            self.assertEqual((root / "cases.txt").read_text(),"case_0000\n")

    def test_failure_restores_original_instance_forward_and_hooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            patchifier = Patchifier()
            patchifier.fail = True
            original = patchifier.forward
            patchifier.forward = original
            tracker = SimpleNamespace(network=SimpleNamespace(patchify=patchifier),n=0,counter=0)
            recorder = PatchifyRecorder(tracker,ReplayWriter(Path(tmp)))
            with self.assertRaisesRegex(RuntimeError,"injected"):
                with recorder.installed():
                    patchifier.forward(Tensor(np.zeros((1,1,3,16,24),dtype=np.float32)),2,return_color=True)
            self.assertIs(patchifier.forward,original)
            self.assertEqual(patchifier.fnet.hooks,[])
            self.assertEqual(patchifier.inet.hooks,[])
            self.assertFalse((Path(tmp) / "cases.txt").exists())

    def test_fp16_and_nonfinite_are_rejected(self):
        for array in (np.zeros(1,dtype=np.float16),np.array([np.nan],dtype=np.float32)):
            with self.assertRaises(ValueError): snapshot(Tensor(array),"float32")

    def test_defaults_enforce_fp32_and_expected_sequence(self):
        args = parse_args([])
        self.assertFalse(args.mixed_precision)
        self.assertEqual((args.frame_count,args.patches_per_frame,args.max_long_edge),(16,16,752))
        self.assertIn("patchify_replay",str(args.output_root))


if __name__ == "__main__":
    unittest.main()
