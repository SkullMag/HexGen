"""Offline checks of topology construction, failed allocations, inventory and ECDFs."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
from provision import build,provision,validate
from resources import capture,write_inventory
from plot_ecdf import ecdf
from export_results import export
from resources import digest

ROOT=Path(__file__).resolve().parent


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):self.plan=json.loads((ROOT/'five-node-plan.json').read_text())

    def test_duplicate_hosts_rejected(self):
        self.plan['nodes'][1]['host']=self.plan['nodes'][0]['host']
        with self.assertRaises(ValueError):validate(self.plan)

    def test_single_site_sts_rejected(self):
        self.plan['networks'][0]['type']='L2STS'
        with self.assertRaises(ValueError):validate(self.plan)

    def test_address_outside_subnet_rejected(self):
        self.plan['networks'][0]['members'][0]['address']='192.0.2.1'
        with self.assertRaises(ValueError):validate(self.plan)

    def test_real_fablib_offline_topology(self):
        from fabrictestbed_extensions.fablib.fablib import FablibManager
        fab=FablibManager(offline=True,validate_config=False,no_ssh=True,auto_token_refresh=False,
                          project_id='00000000-0000-0000-0000-000000000000',bastion_username='offline')
        sl,ports=build(fab,validate(self.plan),'offline-test')
        self.assertEqual(len(sl.get_nodes()),5)
        self.assertEqual(len(sl.get_networks()),2)
        self.assertEqual(len(ports),4)
        models=[c.get_model() for c in sl.get_components()]
        self.assertEqual(models.count('GPU_TeslaT4'),4)
        self.assertEqual(models.count('GPU_RTX6000'),2)
        self.assertEqual(models.count('GPU_A30'),2)
        self.assertTrue(all(str(n.get_type())=='L2Bridge' for n in sl.get_networks()))

    def test_failed_wait_keeps_allocation_id_and_report(self):
        fab=Mock();fab.get_slices.return_value=[]
        sl=Mock();sl.submit.return_value='slice-123';sl.get_slice_id.return_value='slice-123'
        sl.wait.side_effect=TimeoutError('not ready')
        with tempfile.TemporaryDirectory() as directory:
            args=argparse.Namespace(out=directory,name='test',plan=ROOT/'five-node-plan.json',
                                    lease_hours=24,timeout=1,host_info=False)
            with patch('provision.build',return_value=(sl,{})),patch('provision.capture',return_value={'slices':[]}):
                with self.assertRaises(TimeoutError):provision(fab,args,self.plan)
            state=json.loads((Path(directory)/'provision-state.json').read_text())
            self.assertEqual(state['slice_id'],'slice-123');self.assertEqual(state['status'],'incomplete')
            self.assertTrue((Path(directory)/'resources.txt').exists());sl.delete.assert_not_called()

    def test_existing_name_never_modified(self):
        fab=Mock();old=Mock();old.get_name.return_value='test';fab.get_slices.return_value=[old]
        with tempfile.TemporaryDirectory() as directory:
            args=argparse.Namespace(out=directory,name='test')
            with self.assertRaises(ValueError):provision(fab,args,self.plan)
        fab.new_slice.assert_not_called();old.submit.assert_not_called()

    def test_refresh_failure_still_writes_inventory(self):
        sl=Mock();sl.update.side_effect=RuntimeError('API down');sl.toDict.return_value={'name':'test'}
        sl.get_nodes.side_effect=RuntimeError();sl.get_networks.side_effect=RuntimeError()
        data=capture([sl],guest=False)
        self.assertEqual(data['slices'][0]['refresh_error']['unavailable'],'RuntimeError')
        with tempfile.TemporaryDirectory() as directory:
            write_inventory(directory,data)
            self.assertIn('refresh_error', (Path(directory)/'resources.txt').read_text())

    def test_ecdf_ties_failures_and_raw_seconds(self):
        records=[{'success':True,'latency_seconds':2},{'success':True,'latency_seconds':1},
                 {'success':True,'latency_seconds':2},{'success':False,'latency_seconds':.1}]
        x,y=ecdf(records);np.testing.assert_array_equal(x,[1.,2.]);np.testing.assert_array_equal(y,[.25,.75])

    def test_ecdf_invalid_success_rejected(self):
        for value in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):ecdf([{'success':True,'latency_seconds':value}])
        with self.assertRaises(ValueError):ecdf([])

    def test_ecdf_all_failures_stays_zero(self):
        x,y=ecdf([{'success':False,'latency_seconds':None}])
        self.assertEqual(len(x),0);self.assertEqual(len(y),0)

    def test_published_results_bound_to_inventory(self):
        runs=list((ROOT/'results/2026-10-13').glob('*/manifest.json'))
        self.assertEqual(len(runs),6)
        for manifest in runs:
            run=manifest.parent;inventory=json.loads((run/'resources.json').read_text())
            self.assertEqual(inventory['source_kind'],'historical_recorded_evidence')
            self.assertEqual(sum(len(s['nodes']) for s in inventory['slices']),5)
            for arm in ('homogeneous','heterogeneous'):
                result=run/(arm+'.json');records=json.loads(result.read_text())['requests']
                self.assertEqual(len(records),100)
                self.assertTrue(all(set(r)=={'success','latency_seconds'} for r in records))
                self.assertTrue(all(r['success'] for r in records))
                self.assertEqual(inventory['result_files'][arm]['sha256'],digest(result))
                x,y=ecdf(records);self.assertEqual(y[-1],1.)

    def test_export_strips_prompts_and_controller_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';run=source/'example';run.mkdir(parents=True)
            (run/'manifest.json').write_text(json.dumps({'rate':.5,'output_tokens':128,
                                                       'secret_controller_path':'SENSITIVE'}))
            for arm in ('homogeneous','heterogeneous'):
                (run/(arm+'.json')).write_text(json.dumps({'requests':[
                    {'success':True,'latency_seconds':2.,'output_text':'SENSITIVE',
                     'prompt':'SENSITIVE'}]}))
            recorded={'schema_version':1,'assembled_utc':'recorded','source_kind':'historical_recorded_evidence',
                      'note':'historical','not_recorded':[],
                      'sources':{'reservation':{'path':'/SENSITIVE/reservation.json','sha256':'recorded'}},
                      'runtime_provenance':'SENSITIVE','slices':[{'attributes':{'name':'test'},'nodes':[
                          {'attributes_recorded_by_fablib':{'name':'node','host':'physical-host',
                                                           'management_ip':'SENSITIVE'},
                           'requested_resources_not_new_measurements':{'count':2},
                           'post_run_runtime':{'gpu_runtime':'Tesla T4','resident_worker_processes':'SENSITIVE',
                                               'native_peers':'SENSITIVE'}}]}]}
            (run/'resources.json').write_text(json.dumps(recorded))
            dest=Path(directory)/'export';export(source,dest)
            published=''.join(p.read_text() for p in dest.rglob('*') if p.is_file())
            self.assertNotIn('SENSITIVE',published)
            self.assertIn('physical-host',published);self.assertIn('Tesla T4',published)
            with self.assertRaises(FileExistsError):export(source,dest)


if __name__=='__main__':unittest.main()
