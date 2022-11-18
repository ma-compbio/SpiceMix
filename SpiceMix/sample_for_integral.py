import torch


def project2simplex(x, dim):
	"""
	# https://math.stackexchange.com/questions/2402504/orthogonal-projection-onto-the-unit-simplex
	find a scalar u such that || (x-u)_+ ||_1 = 1
	"""
	x.sub_(x.sum(dim=dim, keepdim=True).sub_(1), alpha=1/x.shape[dim])
	u = x.max(dim=dim, keepdim=True)[0].div_(2)
	g_prev, g = None, None
	for i in range(x.shape[dim]):
		t = x.sub(u)
		f = t.clip_(min=0).sum(dim, keepdim=True).sub_(1)
		g = t.gt_(0).sum(dim, keepdim=True)
		if i > 0 and g_prev.eq_(g).all():
			break
		u.addcdiv_(f, g)
		g_prev = g
	x.sub_(u).clip_(min=0)
	assert x.sum(dim=dim).sub_(1).abs_().max() < 1e-4, x.sum(dim=dim).sub_(1).abs_().max()
	return x


def integrate_of_exponential_over_simplex(eta, eps=1e-15):
	assert torch.isfinite(eta).all()
	K = eta.shape[-1]
	A = torch.empty_like(eta)
	signs = torch.empty_like(A)
	for k in range(K):
		t = eta - eta[..., [k]]
		# assert torch.isfinite(t).all()
		t[..., k] = 1
		tsign = t.sign()
		signs[..., k] = tsign.prod(-1)
		t = t.abs().add(eps).log()
		# assert torch.isfinite(t).all()
		t[..., k] = eta[..., k]
		A[..., k] = t.sum(-1).neg()
	assert torch.isfinite(A).all()
	# -- signed logsumexp
	o = A.max(-1, keepdim=True)[0]
	ret = A.sub(o).exp()
	assert torch.isfinite(ret).all()
	ret = ret.mul(signs).sum(-1)
	ret = ret.clip(min=eps)
	assert (ret > 0).all(), ret.min().item()
	ret = ret.log().add(o.squeeze(-1))
	return ret
