"""Seven-method collection and cross-system rejection."""
import json,tempfile,unittest
from pathlib import Path
from tools.summarize_continuous_runs import collect_campaign,METHODS

class CollectionTests(unittest.TestCase):
    def fixture(self,root,T,system,methods):
        run=root/f'T{T}/run';(run/'models').mkdir(parents=True)
        settings=dict(T=T,seed=101,N_obs=0,noise_sigma=.005,window=3.5,eval_systems=1,min_duration_separation=.01)
        cfg=dict(settings=settings,observation_kind='endpoint_rocof',rocof_sample_dt=.025,
                 physical_config={'system':{'name':system}})
        rows=[dict(method=m,evaluation_seed=1001,system=0,true_MK=[1.]*10,
            terminal_spce_nats=1.,duration_sequence_s=[.3+.2*k for k in range(T)]) for m in methods]
        contents={'run_config.json':cfg,'completion.json':dict(objective='eig',smoke_only=False,methods=methods,evaluation_seeds=[1001],records=len(rows)),
            'rollouts.json':rows,'summary.json':[dict(method=m,mean_spce_nats=1.) for m in methods],
            'models/rl_sboed_training.json':[{'redq':{'target_entropy':-1.}}]}
        for name,value in contents.items():(run/name).write_text(json.dumps(value))
        (run/'exit_code').write_text('0')
    def test_ieee14_seven_methods(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);methods=METHODS+('moe_sboed',)
            self.fixture(root,3,'ieee14',methods)
            collect_campaign(root,horizons=(3,),methods=methods)
            report=(root/'combined_results/RESULTS.md').read_text()
            self.assertIn('IEEE14',report);self.assertIn('moe_sboed',report)
    def test_mixed_systems_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.fixture(root,3,'ieee9',METHODS);self.fixture(root,4,'ieee14',METHODS)
            with self.assertRaises(AssertionError):collect_campaign(root,horizons=(3,4))
if __name__=='__main__':unittest.main()
