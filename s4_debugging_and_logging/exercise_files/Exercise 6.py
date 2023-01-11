### Exercise 6
print("Exercise 6")
import torch
import torchvision.models as models
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)
with profile(activities=[ProfilerActivity.CPU], on_trace_ready=tensorboard_trace_handler("log/resnet18")) as prof:
    for i in range(10):
        model(inputs)
        prof.step()