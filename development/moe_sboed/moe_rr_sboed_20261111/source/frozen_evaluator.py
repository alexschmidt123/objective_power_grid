"""Memory-bounded Case 1a sPCE against a shared bank, as upstream asses()."""
import math
import torch


def contrast_scores(benchmark, result, bank, chunk=16384, device=None):
    """Exact log-sum-exp reduction over every contrast; no subsampling."""
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    actions = torch.stack(result['actions'],1).to(device)
    observations = torch.stack(result['observations'],1).to(device)
    truth = result['truth'].to(device)
    b,t,_ = actions.shape
    sigma = benchmark.sigma
    constant = math.log(sigma*math.sqrt(2*math.pi))
    true_cumulative = torch.zeros(b,dtype=torch.float64,device=device)
    true_stages = []
    for stage in range(t):
        distances = (truth.reshape(b,2,2)-actions[:,stage,None]).square().sum(-1)
        mean = (.1+(1e-4+distances).reciprocal().sum(-1)).log()
        true_cumulative += -.5*((observations[:,stage,0]-mean)/sigma).square()-constant
        true_stages.append(true_cumulative.clone())
    true_stages = torch.stack(true_stages,1)
    contrast_sums = torch.full((b,t),-torch.inf,dtype=torch.float64,device=device)
    for start in range(0,len(bank),chunk):
        sources = bank[start:start+chunk].to(device).reshape(1,-1,2,2)
        cumulative = torch.zeros(b,len(sources[0]),dtype=torch.float64,device=device)
        for stage in range(t):
            distances = (sources-actions[:,stage,None,None]).square().sum(-1)
            mean = (.1+(1e-4+distances).reciprocal().sum(-1)).log()
            cumulative += -.5*((observations[:,stage,0,None]-mean)/sigma).square()-constant
            contrast_sums[:,stage] = torch.logaddexp(contrast_sums[:,stage], torch.logsumexp(cumulative,1))
    info = true_stages-torch.logaddexp(true_stages,contrast_sums)+math.log(len(bank)+1)
    upper = true_stages[:,-1]-contrast_sums[:,-1]+math.log(len(bank))
    benchmark.calls += b*t*len(bank)
    return info.cpu(), upper.cpu()
