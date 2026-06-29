import torch
from flash_attn.cute import flash_attn_func
print("torch", torch.__version__)
B,S,Hn,D=2,512,16,128
q=torch.randn(B,S,Hn,D,device="cuda",dtype=torch.bfloat16,requires_grad=True)
k=torch.randn(B,S,Hn,D,device="cuda",dtype=torch.bfloat16,requires_grad=True)
v=torch.randn(B,S,Hn,D,device="cuda",dtype=torch.bfloat16,requires_grad=True)
out=flash_attn_func(q,k,v,causal=True)
if isinstance(out,tuple): out=out[0]
try:
    out.float().pow(2).mean().backward()
    torch.cuda.synchronize()
    print("FA4 FWD+BWD OK  dq.mean", q.grad.float().mean().item())
except Exception as e:
    import traceback; print("FA4 BWD FAILED:", type(e).__name__, str(e)[:120]); 
