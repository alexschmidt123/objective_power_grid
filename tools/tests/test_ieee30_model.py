"""Validate the paper-based specification and reject unsupported execution."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from src.config import load_config_for_run
from src.domains.swing.design import build_simulator
from src.domains.swing.simulator import ieee30_physical_input_map,generate_ieee30_coupling_matrix
ROOT=Path(__file__).resolve().parents[2]
class Tests(unittest.TestCase):
 def setUp(self):self.cfg=load_config_for_run('configs/ieee30_mocu.yaml',ROOT);self.d=self.cfg.raw
 def test_published_machine_and_governor_parameters(self):
  machines=self.d['dynamic_machines']
  self.assertEqual(self.cfg.N,6)
  self.assertEqual([m['physical_bus'] for m in machines],[1,2,5,8,11,13])
  self.assertEqual([m['terminal_bus'] for m in machines],list(range(31,37)))
  self.assertEqual([m['H_s'] for m in machines],[4.13,5.078,1.52,1.52,1.2,1.2])
  self.assertEqual([m['rated_power_mva'] for m in machines],[270,51.2,40,40,25,25])
  self.assertEqual([m['physical_bus'] for m in machines if m['governor']],[1,2])
  self.assertEqual([m['governor']['R_pu'] for m in machines[:2]],[.0185,.1523])
  self.assertTrue(all(m['model']=='GENROU' and m['exciter']['model']=='IEEET1' for m in machines))
 def test_eight_latents_and_machine_base_conversion(self):
  self.assertEqual(self.d['uncertainty']['latent_dimension'],8)
  self.assertEqual(self.d['uncertainty']['latent_names'],['H_1','H_2','H_5','H_8','H_11','H_13','R_1','R_2'])
  expected=[2*m['H_s']*m['rated_power_mva']/100/(2*np.pi*50) for m in self.d['dynamic_machines']]
  np.testing.assert_allclose(self.d['derived_parameter_conventions']['M_nominal_nodes'],expected,rtol=1e-14)
  self.assertEqual(self.d['derived_parameter_conventions']['K_static_nominal_nodes'][2:],[0,0,0,0])
 def test_full_model_cannot_silently_run_legacy_solver(self):
  with self.assertRaisesRegex(NotImplementedError,'GENROU'):build_simulator(self.cfg)
  self.assertFalse(self.d['data']['generate_if_missing'])
  self.assertFalse(self.d['implementation']['current_backend_compatible'])
 def test_legacy_topology_mapping_remains_available(self):
  mapping=ieee30_physical_input_map();B=generate_ieee30_coupling_matrix()
  self.assertEqual(mapping.shape,(30,6));self.assertEqual(B.shape,(6,6))
  np.testing.assert_allclose(mapping.sum(1),1,atol=1e-12)
  np.testing.assert_allclose(mapping[np.array([1,2,5,8,11,13])-1],np.eye(6),atol=1e-12)
if __name__=='__main__':unittest.main(verbosity=2)
