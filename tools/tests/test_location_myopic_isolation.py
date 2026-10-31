import unittest
import torch
from unittest.mock import patch
from src.research.benchmarks import LocationFinding
from src.research.location_posterior import LocationSMC


class MyopicIsolationTests(unittest.TestCase):
    def benchmark(self):
        return LocationFinding({'location_finding':{'design_bounds':[-4,4]},
            'research':{'paper_protocol':{'corrected_methods':True}}})

    def test_other_episode_observations_do_not_change_posterior(self):
        b=self.benchmark();streams=[]
        for other in [0.,20.]:
            s=LocationSMC(b,2,256,torch.Generator().manual_seed(192817),moves=2)
            for y in [1.,1.5]:
                s.update(torch.zeros(2,2,dtype=torch.float64),
                         torch.tensor([[y],[other]],dtype=torch.float64))
            streams.append(s)
        torch.testing.assert_close(streams[0].theta[0],streams[1].theta[0],rtol=0,atol=0)
        torch.testing.assert_close(streams[0].scale[0],streams[1].scale[0],rtol=0,atol=0)



if __name__=='__main__':unittest.main()
