import torch
from torch import nn


def make_randomized_layernorm(noise_rampup_steps=10000):
    """
    Return a RandomizedLayerNorm class.
    it is customized by the specifed noise_rampup_steps.
    Source: https://github.com/bolianchen/pytorch_depth_from_videos_in_the_wild
    """

    class RandomizedLayerNorm(nn.Module):
        def __init__(self, num_features, affine=True):
            super(RandomizedLayerNorm, self).__init__()
            self.beta = torch.nn.Parameter(torch.zeros(num_features), requires_grad=affine)
            self.gamma = torch.nn.Parameter(torch.ones(num_features), requires_grad=affine)

            # the difference between 1.0 and the next smallest
            # machine representable float
            self.epsilon = torch.finfo(torch.float32).eps
            self.step = 0

        def _truncated_normal(self, shape=(), mean=0.0, stddev=1.0):
            """
            # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/20
            refactor heiner's solution.
            TODO: check the correctness of the function: remap uniform to standard normal.
            """
            uniform = torch.rand(shape, device=self.beta.device)

            def parameterized_truncated_normal(uniform, mean, stddev, lower_level=-2, upper_level=2):
                # stardard normal
                normal = torch.distributions.normal.Normal(0, 1)

                lower_normal_cdf = normal.cdf(torch.tensor(lower_level))
                upper_normal_cdf = normal.cdf(torch.tensor(upper_level))

                p = lower_normal_cdf + (upper_normal_cdf - lower_normal_cdf) * uniform

                # clamp the values out of the range to the edge values
                v = torch.clamp(2 * p - 1, -1.0 + self.epsilon, 1.0 - self.epsilon)
                x = mean + stddev * torch.sqrt(torch.tensor(2.0)) * torch.erfinv(v)

                return x

            return parameterized_truncated_normal(uniform, mean, stddev)

        def forward(self, x):
            mean = x.mean((2, 3), keepdim=True)
            variance = torch.square(x - mean).mean((2, 3), keepdim=True)
            if noise_rampup_steps <= 0:
                stddev = 0.5
            else:
                stddev = 0.5 * pow(min(self.step / noise_rampup_steps, 1.0), 2)
            if self.training:
                mean = torch.mul(
                    mean,
                    1.0 + self._truncated_normal(mean.shape, stddev=stddev)
                    # 1.0 + init.trunc_normal_(torch.zeros_like(mean), std=stddev)
                )
                variance = torch.mul(
                    variance,
                    1.0 + self._truncated_normal(variance.shape, stddev=stddev)
                    # 1.0 + init.trunc_normal_(torch.zeros_like(variance), std=stddev)
                )
            outputs = self.gamma.view(1, -1, 1, 1) * torch.div(x - mean, torch.sqrt(variance) + 1e-3) + self.beta.view(
                1, -1, 1, 1
            )
            self.step += 1
            return outputs

    return RandomizedLayerNorm
