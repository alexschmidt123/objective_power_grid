import math,torch
from hec_policy import mean_signal,advance
from soft_policy import prefix_inputs
@torch.no_grad()
def native_logpost(a,ds,ys,theta,count):
 out=theta.new_empty(len(theta))
 for n in count.unique().tolist():
  idx=(count==n).nonzero().flatten()
  with torch.device(theta.device):out[idx]=a.post_approx.prior.log_poi_prior(theta[idx]) if n==0 else a.post_approx.log_poi_post(n-1,ds[idx,:n],ys[idx,:n],theta[idx])
 return out
@torch.no_grad()
def labels(a,reference,model,x,info,generator,replicates=4):
 w=info['geometry_attention'];mu=info['geometry_mu'];sd=info['geometry_sd'];mean=(w[...,None]*mu).sum(1);cov=torch.einsum('bm,bmi,bmj->bij',w,mu,mu)-mean[:,:,None]*mean[:,None,:]+torch.diag_embed((w[...,None]*sd.square()).sum(1));var=cov.diagonal(dim1=-2,dim2=-1).clamp_min(1e-6);cdf=(w[...,None]*(.5*(1+torch.erf((mean[:,None]-mu)/sd/math.sqrt(2))))).sum(1);ij=torch.triu_indices(4,4,1,device=x.device);corr=cov[:,ij[0],ij[1]]/(var[:,ij[0]]*var[:,ij[1]]).sqrt();geo=torch.cat([cdf,torch.log1p(var),corr],-1)
 F=info['F'];eig,vec=torch.linalg.eigh(F.double());v=info['direction'];sens=torch.cat([torch.log1p(eig.clamp_min(0)).float(),vec[:,:,0].square().float(),torch.log1p(torch.einsum('bi,bij,bj->b',v,F,v).clamp_min(0))[:,None]],-1)
 c2=info['commands'][:,4:10];direction=c2[:,:2]+x.new_tensor([1.,0.]);direction=direction/(direction.square().sum(-1,keepdim=True)+1e-6).sqrt();rot=torch.stack([direction[:,0]-direction[:,1],direction[:,0]+direction[:,1]],-1)/math.sqrt(2);a0=info['probes'][:,0];rotated=(a0[:,None]+x.new_tensor([0.,1.,-1.])[None,:,None]*rot[:,None]).clamp(-4,4);pred=mean_signal(info['nodes'],rotated);wp=info['pair_plus'];wm=info['pair_minus'];mp=(wp[:,None]*pred).sum(-1);mm=(wm[:,None]*pred).sum(-1);vp=(wp[:,None]*(pred-mp[:,:,None]).square()).sum(-1);vm=(wm[:,None]*(pred-mm[:,:,None]).square()).sum(-1);dis=torch.log1p((vp+vm+(mp-mm).square())/.5)
 b=len(x);r=replicates;ix=torch.multinomial(info['geometry_weights'],r,replacement=True,generator=generator);row=torch.arange(b,device=x.device)[:,None];th=mu[row,ix]+sd[row,ix]*torch.randn(b,r,4,device=x.device,generator=generator);theta=th[:,None].expand(-1,3,-1,-1).reshape(-1,4);xx=x[:,None,None].expand(-1,3,r,-1).reshape(-1,117);stage=xx[:,:30].argmax(-1);ds=x.new_zeros(len(xx),30,2);ys=x.new_zeros(len(xx),30,1);ds[:,:29]=xx[:,30:88].reshape(-1,29,2);ys[:,:29,0]=xx[:,88:];actions=info['probe4'][:,:,None].expand(-1,-1,r,-1).reshape(-1,2);noise1=torch.randn(b,r,device=x.device,generator=generator)[:,None].expand(-1,3,-1).reshape(-1);noise2=torch.randn(b,r,device=x.device,generator=generator)[:,None].expand(-1,3,-1).reshape(-1);ids=torch.arange(len(xx),device=x.device);y1=mean_signal(theta[:,None],actions[:,None])[:,0,0]+.5*noise1;ds[ids,stage]=actions;ys[ids,stage,0]=y1;count1=stage+1;lp1=native_logpost(a,ds,ys,theta,count1);r2=x.new_zeros(len(xx));tail=x.new_zeros(len(xx));alive=(count1<30).nonzero().flatten();calls=len(xx)
 if len(alive):
  nx=prefix_inputs(ds[alive],ys[alive],count1[alive]);next_action=reference(nx);next_y=mean_signal(theta[alive,None],next_action[:,None])[:,0,0]+.5*noise2[alive];ds[alive,count1[alive]]=next_action;ys[alive,count1[alive],0]=next_y;count2=count1[alive]+1;lp2=native_logpost(a,ds[alive],ys[alive],theta[alive],count2);r2[alive]=lp2-lp1[alive];calls+=len(alive);more=(count2<30).nonzero().flatten()
  if len(more):
   idx=alive[more];nx2=prefix_inputs(ds[idx],ys[idx],count2[more]);na=reference(nx2);tail[idx]=a.critic_net(torch.cat([nx2,na],-1)).squeeze(-1)
 second=r2.reshape(b,3,r).mean(-1)/5;future=tail.reshape(b,3,r).mean(-1)/10;seq=torch.cat([second-second[:,:1],future-future[:,:1]],-1);return [geo,dis,sens,seq],dict(predictive_observations=calls)

def supervised(model,x,commands,targets,scale):
 _,info=model(x,command_override=commands,collect=True);features=info['responses'];pred=[model.response_readouts[0](features[:,1,12:]),model.response_readouts[1](features[:,3,12:])];loss=[((pred[0]-targets[1])/scale[1]).square().mean(),((pred[1]-targets[3])/scale[3]).square().mean()];return sum(loss),loss
