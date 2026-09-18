"""CPU-only tests of capture boundaries, ownership, stage labels, and cleanup."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np
from generate_update_replay_testdata import UpdateRecorder, ReplayWriter, parse_args


class Tensor:
    def __init__(self, value): self.array = np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.array
    def __getitem__(self, item): return Tensor(self.array[item])
    def __lt__(self, value): return self.array < value


def zeros(*shape): return Tensor(np.zeros(shape, dtype=np.float32))


class Network:
    def __init__(self): self.calls = 0
    def forward(self, net, inp, corr, flow, ii, jj, kk):
        self.calls += 1
        net.array += 1
        return net, (zeros(1,2,2), Tensor(np.ones((1,2,2),np.float32)), None)


class Tracker:
    n=2; m=2; M=1; N=8; ht=16; wd=16; pmem=3; mem=4; counter=7
    is_initialized=False
    def __init__(self, module):
        self.module=module
        self.network=SimpleNamespace(update=Network())
        self.cfg=SimpleNamespace(OPTIMIZATION_WINDOW=7,BA_ITERATIONS=2,REMOVAL_WINDOW=16,PATCH_LIFETIME=11)
        idx=Tensor(np.array([0,1],dtype=np.int64))
        self.ix=idx
        self.pg=SimpleNamespace(poses_=zeros(8,7),patches_=zeros(8,1,3,3,3),intrinsics_=zeros(8,4),
                                points_=zeros(8,3),net=zeros(1,2,384),ii=idx,jj=idx,kk=idx)
        self.ran_global_ba=np.zeros(8,bool)
        self.gmap_=zeros(3,1,128,3,3); self.imap_=zeros(3,1,384)
        self.pyramid=(zeros(1,4,128,4,4),zeros(1,4,128,1,1))
        self.fail=False; self.ba_fail=False
    def corr(self, coords, indicies=None): return zeros(1,2,882)
    def update(self):
        corr=self.corr(zeros(1,2,2,3,3))
        self.pg.net,(delta,weight,_)=self.network.update.forward(
            self.pg.net,zeros(1,2,384),corr,None,self.pg.ii,self.pg.jj,self.pg.kk)
        if self.fail: raise RuntimeError("injected update failure")
        self.pg.target=delta; self.pg.weight=weight
        try: self.module.fastba.BA(self)
        except RuntimeError: pass  # Python tracker also catches BA errors
        self.pg.points_.array[:2]=3
    def motion_probe(self):
        self.network.update.forward(zeros(1,2,384),zeros(1,2,384),zeros(1,2,882),
                                    None,self.pg.ii,self.pg.jj,self.pg.kk)


class CaptureTests(unittest.TestCase):
    def context(self, root, include=False):
        module=ModuleType('dpvo.dpvo')
        def ba(tracker):
            if tracker.ba_fail: raise RuntimeError('injected BA failure')
            tracker.pg.poses_.array[:2] += 2
            tracker.pg.patches_.array[:2] += 4
        module.fastba=SimpleNamespace(BA=ba)
        parent=ModuleType('dpvo'); parent.dpvo=module
        tracker=Tracker(module)
        recorder=UpdateRecorder(tracker,ReplayWriter(root),include)
        recorder.start_frame(7,Path('frame.png'))
        return tracker,recorder,patch.dict(sys.modules,{'dpvo':parent,'dpvo.dpvo':module})

    def test_capture_original_once_owns_pre_and_post_state_and_labels_termination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); t,r,modules=self.context(root)
            with modules,r.installed():
                t.motion_probe()  # excluded by default
                t.update()
                r.terminating=True; r.next_iteration=0
                t.update()
                t.pg.poses_.array.fill(-99)
                t.pg.net.array.fill(-99)
            self.assertEqual(t.network.update.calls,3)
            self.assertEqual(len(r.writer.records),2)
            self.assertNotIn('forward',t.network.update.__dict__)
            self.assertNotIn('update',t.__dict__)
            a=root/'case_0000'
            self.assertTrue(np.all(np.fromfile(a/'poses.bin',dtype='<f4')==0))
            self.assertTrue(np.all(np.fromfile(a/'golden_poses.bin',dtype='<f4')==2))
            self.assertTrue(np.all(np.fromfile(a/'net.bin',dtype='<f4')==0))
            self.assertTrue(np.all(np.fromfile(a/'golden_net.bin',dtype='<f4')==1))
            self.assertEqual([x['stage'] for x in r.writer.records],['initialization','terminate'])
            self.assertTrue(all(x['iteration_supported'] for x in r.writer.records))
            self.assertEqual(json.loads((a/'metadata.json').read_text())['tensor_shapes']['patches'],[2,1,3,3,3])
            self.assertEqual(len(list((root/'features').glob('*.bin'))),3)  # identical features dedup

    def test_probe_is_network_only_and_ba_failure_is_not_claimed_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            t,r,modules=self.context(Path(tmp),True)
            with modules,r.installed():
                t.motion_probe()
                t.ba_fail=True; t.update()
            self.assertEqual(r.writer.records[0]['stage'],'motion_probe')
            self.assertFalse(r.writer.records[0]['iteration_supported'])
            self.assertTrue(r.writer.records[1]['ba_failed'])
            self.assertFalse(r.writer.records[1]['iteration_supported'])

    def test_failure_restores_instance_methods_and_does_not_index_partial_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            t,r,modules=self.context(Path(tmp))
            original=t.update; t.update=original
            original_ba=t.module.fastba.BA
            t.fail=True
            with self.assertRaisesRegex(RuntimeError,'injected update'),modules,r.installed(): t.update()
            self.assertIs(t.update,original)
            self.assertIs(t.module.fastba.BA,original_ba)
            self.assertNotIn('forward',t.network.update.__dict__)
            self.assertEqual(r.writer.records,[])
            self.assertIsNone(r.current)

    def test_defaults_and_explicit_fp32(self):
        a=parse_args([])
        self.assertFalse(a.mixed_precision)
        self.assertFalse(a.include_motion_probes)
        self.assertFalse(a.skip_terminate_updates)
        self.assertEqual(a.patches_per_frame,16)


if __name__=='__main__': unittest.main()
