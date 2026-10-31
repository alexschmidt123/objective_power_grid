import unittest
from src.research.training import training_contrasts

class Tests(unittest.TestCase):
    def test_predeclared_boundaries_and_legacy_preservation(self):
        s=dict(contrasts=16384,spce_curriculum=3)
        self.assertEqual([training_contrasts(s,x) for x in [0,.199,.2,.499,.5,1]],[1024,1024,4096,4096,16384,16384])
        self.assertEqual(training_contrasts(s|dict(spce_curriculum=1),0),16384)
        self.assertEqual(training_contrasts(s|dict(spce_curriculum=2),0),128)
        self.assertEqual(training_contrasts(s|dict(contrasts=8),.8),8)

    def test_schedule_respects_hard_forward_budget(self):
        s=dict(contrasts=16384,spce_curriculum=3)
        budget=10000000;spent=0;counts={}
        while True:
            contrasts=training_contrasts(s,spent/budget)
            batch=min(4,(budget-spent)//(contrasts+1))
            if batch<1:break
            spent+=batch*(contrasts+1)
            counts[contrasts]=counts.get(contrasts,0)+batch
        self.assertLessEqual(spent,budget)
        self.assertLess(budget-spent,16385)
        self.assertEqual(set(counts),{1024,4096,16384})

if __name__=='__main__':unittest.main()
