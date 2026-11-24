"""Common streamed sPCE EIG evaluation with the actual SIR ODE likelihood."""
import math,torch
from src.research.engine import replay_likelihood

def contrast_scores(benchmark,result,bank,chunk=128,device=None):
 device=device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
 with torch.device(device),torch.no_grad():
  truth=result['truth'].to(device);actions=[a.to(device) for a in result['actions']];observations=[y.to(device) for y in result['observations']]
  _,_,true=replay_likelihood(benchmark,truth[:,None,:],actions,observations)
  true=torch.stack([v[:,0] for v in true],1)
  sums=torch.full_like(true,-torch.inf)
  for start in range(0,len(bank),chunk):
   contrast=bank[start:start+chunk].to(device)[None].expand(len(truth),-1,-1)
   _,_,values=replay_likelihood(benchmark,contrast,actions,observations)
   scores=torch.stack(values,2)
   sums=torch.logaddexp(sums,torch.logsumexp(scores,1))
  info=true-torch.logaddexp(true,sums)+math.log(len(bank)+1)
  upper=true[:,-1]-sums[:,-1]+math.log(len(bank))
 return info.cpu(),upper.cpu()
