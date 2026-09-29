import torch
import torch.nn as nn


# fix seed so that random initialization always performs the same 
torch.manual_seed(13)


# create the model N as described in the question
N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))

# random input
x = torch.rand((1,10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data
x.requires_grad_() # this is required so we can compute the gradient w.r.t x

t = 0 # target class

epsReal = 0.5  #depending on your data this might be large or small
eps = epsReal - 1e-7 # small constant to offset floating-point erros

# The network N classfies x as belonging to class 2
original_class = N(x).argmax(dim=1).item()  # TO LEARN: make sure you understand this expression
print("Original Class: ", original_class)
assert(original_class == 2)

# compute gradient
# note that CrossEntropyLoss() combines the cross-entropy loss and an implicit softmax function
L = nn.CrossEntropyLoss()
loss = L(N(x), torch.tensor([t], dtype=torch.long)) # TO LEARN: make sure you understand this line
loss.backward()

# your code here
# adv_x should be computed from x according to the fgsm-style perturbation such that the new class of xBar is the target class t above
# hint: you can compute the gradient of the loss w.r.t to x as x.grad
adv_x = x - eps * x.grad.sign()

new_class = N(adv_x).argmax(dim=1).item()
print("New Class: ", new_class)
assert(new_class == t)
# it is not enough that adv_x is classified as t. We also need to make sure it is 'close' to the original x. 
print(torch.norm((x-adv_x),  p=float('inf')).data)
assert( torch.norm((x-adv_x), p=float('inf')) <= epsReal)

t = 1
x.grad.zero_()
N.zero_grad()
loss = L(N(x), torch.tensor([t], dtype=torch.long))
loss.backward()
adv_x_t1 = x - eps * x.grad.sign()
new_class_t1 = N(adv_x_t1).argmax(dim=1).item()
print("Target 1 FGSM Class:", new_class_t1)
print("Target 1 L_inf:", torch.norm(x - adv_x_t1, p=float("inf")).item())
assert torch.norm(x - adv_x_t1, p=float("inf")) <= epsReal

# Another method: Random-restart targeted PGD
pgd_target = torch.tensor([1], dtype=torch.long)
pgd_epsilon = eps
pgd_step_size = 0.005
pgd_steps = 500
pgd_restarts = 64
generator = torch.Generator().manual_seed(521)
adv_x_pgd = None

for restart in range(pgd_restarts):
    random_delta = torch.empty_like(x).uniform_(
        -pgd_epsilon,
        pgd_epsilon,
        generator=generator,
    )
    candidate = (x.detach()+random_delta).detach()

    for step in range(pgd_steps):
        candidate.requires_grad_()
        pgd_loss = L(N(candidate), pgd_target)
        pgd_gradient = torch.autograd.grad(pgd_loss, candidate)[0]

        candidate = candidate.detach() - pgd_step_size * pgd_gradient.sign()
        delta = (candidate-x.detach()).clamp(
            -pgd_epsilon,
            pgd_epsilon,
            )
        candidate = (x.detach()+delta).detach()

        candidate_class = N(candidate).argmax(dim=1).item()
        if candidate_class == 1:
            adv_x_pgd = candidate
            print("PGD succeeded on restart:", restart+1)
            break
    if adv_x_pgd is not None:
        break

assert adv_x_pgd is not None, "PGD didn't find a target1 exp"
pgd_class = N(adv_x_pgd).argmax(dim=1).item()
pgd_linf = torch.norm(
    x.detach()-adv_x_pgd,
    p=float("inf"),
).item()
pgd_l2 = torch.norm(x.detach() - adv_x_pgd, p=2).item()
print("Target1 PGD Class:", pgd_class)
print("Target1 PGD L_inf:", pgd_linf)
print("Target1 PGD L2:", pgd_l2)
assert pgd_class == 1
assert pgd_linf <= epsReal