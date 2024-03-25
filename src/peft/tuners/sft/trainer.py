import logging
import math

import numpy as np
import torch

from transformers import (
    TrainerCallback,
)
from accelerate.optimizer import AcceleratedOptimizer

from .layer import Linear
from .optimizer import SftAdamW, SftSM3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def update_optimizer(
    optimizer,
    param,
    changing_indices,
    init_momenta={},
):
    """
    Updates optimizer state for a PEFT parameter tensor after dropping and regrowth.

    Args:
      - optimizer: the optimizer to update.
      - param: the parameter tensor.
      - changing_indices: the indices in the optimizer state that need to be updated.
      - init_momenta: dict mapping state keys to seed values. If not supplied, values
          are seeded to zero.
    """
    if isinstance(optimizer, AcceleratedOptimizer):
        optimizer = optimizer.optimizer

    optimizer_state = optimizer.state[param]
    for optim_aux in ['age', 'exp_avg', 'exp_avg_sq']:
        optimizer_params = optimizer_state[optim_aux]
        init = init_momenta.get(optim_aux, None)
        if init is not None:
            if isinstance(init, torch.Tensor):
                init = init.to(dtype=optimizer_params.dtype)
            optimizer_params[changing_indices] = init
        else:
            optimizer_params[changing_indices] = 0.0

def optimizer_resize_param(optimizer, param, changing_indices, init_momenta, staying, new_param, weight_change=0):

    if weight_change == 0:
        update_optimizer(optimizer, param, changing_indices, init_momenta) 
    else:
        if isinstance(optimizer, AcceleratedOptimizer):
            optimizer = optimizer.optimizer

        optimizer_state = optimizer.state[param]
        if weight_change > 0:
            for optim_aux in ['age', 'exp_avg', 'exp_avg_sq']:
                optimizer_params = optimizer_state[optim_aux]
                optimizer_params = torch.cat((optimizer_params,torch.zeros((weight_change), dtype=optimizer_params.dtype, device=optimizer_params.device)))
                init = init_momenta.get(optim_aux, None)
                if init is not None:
                    if isinstance(init, torch.Tensor):
                        init = init.to(dtype=optimizer_params.dtype)
                    optimizer_params[changing_indices] = init
                else:
                    optimizer_params[changing_indices] = 0.0

            del optimizer.state[param]
            for idx, group in enumerate(optimizer.param_groups):
                if len(group["params"]) == 1 and group["params"][0] is param:
                    del optimizer.param_groups[idx]
                    break
            optimizer.add_param_group({"params": new_param})
            logger.info(f"Num param_groups: {len(optimizer.param_groups)}")
        else:
            for optim_aux in ['age', 'exp_avg', 'exp_avg_sq']:
                optimizer_params = optimizer_state[optim_aux]
                new_size = optimizer_params.shape[0] + weight_change
                optimizer_params = optimizer_params[staying]
                num_staying = optimizer_params.shape[0]
                optimizer_params = torch.cat((optimizer_params, torch.zeros((new_size - num_staying), dtype=optimizer_params.dtype, device=optimizer_params.device)))
                init = init_momenta.get(optim_aux, None)
                if init is not None:
                    if isinstance(init, torch.Tensor):
                        init = init.to(dtype=optimizer_params.dtype)
                    optimizer_params[num_staying:] = init
                else:
                    optimizer_params[num_staying:] = 0.0
            del optimizer.state[param]
            for idx, group in enumerate(optimizer.param_groups):
                if len(group["params"]) == 1 and group["params"][0] is param:
                    del optimizer.param_groups[idx]
                    break
            optimizer.add_param_group({"params": new_param})
            logger.info(f"Num param_groups: {len(optimizer.param_groups)}")




def sample(probs_1: torch.Tensor, probs_2: torch.Tensor, n: int) -> torch.Tensor:
    """This function samples pairs from the joint probability distribution induced by the probabilities in probs_1 and probs_2 and returns the indices of the sampled pairs"""
    # Sample n indices without replacement from probs_1 and probs_2
    indices_1 = torch.multinomial(probs_1, num_samples=n, replacement=True)
    indices_2 = torch.multinomial(probs_2, num_samples=n, replacement=True)
    # Combine the indices into pairs and return as a tensor
    return indices_1, indices_2


class SftSelector:
    """
    Implements SFT tunable parameter reselection. Simply construct the SftSelector and call
    .step() after each training update step.
    """

    def __init__(
        self,
        model,
        optimizer,
        sft_config,
        total_update_steps,
        grad_accumulation_steps,
        completed_steps=0, # number of already completed steps if resuming from saved ckpt.
    ):
        self.model = model
        self.optimizer = optimizer
        self.sft_config = sft_config
        self.total_update_steps = total_update_steps
        self.grad_accumulation_steps = grad_accumulation_steps
        self.completed_steps = completed_steps
        self.begin_selection_phase()

    def step(self):
        self.completed_steps += 1
        
        if (self.completed_steps + 1) % self.sft_config.reselection_steps == 0:
            self.begin_selection_phase()
        
        if (
            self.completed_steps % self.sft_config.reselection_steps ==
            self.sft_config.selection_accumulation_steps
        ):
            self.end_selection_phase()

    def begin_selection_phase(self):
        if self.sft_config.selection_algorithm == "sm3":
            return

        logger.info('Beginning selection phase')
        self.reselection_scores = {}
        self.scaler_row = {}
        self.nsamples = {}

        if self.sft_config.selection_algorithm == "gse" and self.sft_config.selection_level == "global":
            num_deltas = 0
            weights = []
            for n, m in self.model.named_modules():
                if (
                    isinstance(m, Linear) and
                    m.active_adapter is not None and
                    m.active_adapter in m.sft_delta
                ):
                    num_deltas += 1
                    weights.append(m.sft_delta[m.active_adapter].indices.shape[0])

            num_connections = sum(weights)
            self.num_delta_weights =  num_connections
            num_to_sample = num_connections * self.sft_config.subset_fraction
            batch_size = int(1e5)
            layer_nums = torch.zeros(num_deltas)
            src = torch.ones((int(batch_size)))
            weights = torch.tensor(weights, dtype=torch.float32)
            for idx in range(0, num_to_sample, batch_size):
                num = min(batch_size,  num_to_sample-idx)
                index = torch.multinomial(weights, num, replacement=True)
                layer_nums = layer_nums.scatter_reduce(0, index, src, reduce="sum")
            logger.info(f"Sampled candidate distribution over the deltas: {layer_nums}")

        # Apply hooks to gather gradients for growth selection
        layer_idx = 0
        for n, m in self.model.named_modules():
            if (
                isinstance(m, Linear) and
                m.active_adapter is not None and
                m.active_adapter in m.sft_delta
            ):
                if self.sft_config.selection_algorithm == "rigl":
                    m.apply_hook(self.gradient_accumulation_hook(n))
                if self.sft_config.selection_algorithm == "gse":
                    p = m.weight
                    probs_out = p.new_ones(p.size(0), dtype=torch.float32)
                    probs_in = p.new_ones(math.prod(p.shape[1:]), dtype=torch.float32)
                    if self.sft_config.selection_level == "global":
                        num = int(layer_nums[layer_idx].item())
                        layer_idx += 1
                    else:
                        num = int(m.sft_delta[m.active_adapter].indices.shape[0] * self.sft_config.subset_fraction)

                    to_nodes, from_nodes = sample(probs_out, probs_in, num)

                    grad_mask = torch.zeros(
                        m.sft_delta[m.active_adapter].shape, 
                        dtype=torch.bool, 
                        device=m.sft_delta[m.active_adapter].indices.device,
                    )
                    grad_mask[to_nodes, from_nodes] = True

                    # remove already active indeces from candidate list
                    is_current = torch.zeros(
                        [m.sft_delta[m.active_adapter].dense_numel],
                        dtype=torch.bool,
                        device=m.sft_delta[m.active_adapter].indices.device,
                    )
                    is_current[m.sft_delta[m.active_adapter].indices] = True

                    grad_mask = grad_mask.view(-1)

                    grad_mask = torch.logical_and(grad_mask, ~is_current)
                    candidate_indices = torch.arange(grad_mask.shape[0], device=grad_mask.device)[grad_mask] 

                    m.apply_hook(self.gradient_accumulation_hook(n, candidate_indices))
                del layer_nums
                    
                if self.sft_config.drop_algorithm == "wanda":
                    m.apply_pre_forward_hook(self.get_wanda_hook(n))


    def end_selection_phase(self):
        logger.info('Ending selection phase')
        if self.completed_steps > self.total_update_steps:
            return

        if self.sft_config.selection_algorithm != "sm3":
            # Remove hooks
            for n, m in self.model.named_modules():
                if (
                    isinstance(m, Linear) and
                    m.active_adapter is not None and
                    m.active_adapter in m.sft_delta
                ):
                    m.apply_hook(None)
                    if self.sft_config.drop_algorithm != "magnitude":
                        m.apply_pre_forward_hook(None)

        # Replace all parameters if it's the first reselection step, linear
        # decay from initial rate otherwise.
        if self.sft_config.reselection_rate_policy == "linear":
            if self.completed_steps == self.sft_config.selection_accumulation_steps:
                p = 1
            else:
                p = self.sft_config.initial_reselection_rate * (
                    1 - self.completed_steps / self.total_update_steps
                )
        elif self.sft_config.reselection_rate_policy == "cosine":
            p = self.sft_config.initial_reselection_rate * (
                1 + math.cos(math.pi * self.completed_steps / self.total_update_steps)
            ) / 2
        else:
            raise ValueError(f'Unsupported reselection rate policy {self.sft_config.reselection_rate_policy}')
        self.select(p)
        self.reselection_scores = {}
        self.scaler_row = {}
        self.nsamples = {}
        logger.info(f"Number of optim param_groups: {len(self.optimizer.param_groups)}")

    def select(self, p):
        
        # select drop algorithm
        if self.sft_config.drop_algorithm == "magnitude":
            drop_fn = self.drop_magnitude
        elif self.sft_config.drop_algorithm == "wanda":
            drop_fn = self.drop_wanda
        else:
            raise ValueError(f'Invalid drop method {self.sft_config.drop_algorithm}')

        if self.sft_config.selection_algorithm == "sm3":
            self.select_sm3(p, drop_fn)
        elif self.sft_config.selection_algorithm == "gse" and self.sft_config.selection_level == "global":
            logger.info("Selecting with global GSE")
            self.select_gse_global(p, drop_fn)
        elif self.sft_config.selection_algorithm == "rigl" or self.sft_config.selection_algorithm == "gse":
            logger.info("Selecting with RigL")
            self.select_rigl(p, drop_fn)
        else:
            raise ValueError(
                f'Invalid selection method {self.sft_config.selection_algorithm}'
            )

    def gradient_accumulation_hook(self, module_name, candidates=None):

        @torch.no_grad()
        def _gradient_accumulation_hook(grad):
            m = self.model.get_submodule(module_name)
            grad = grad.reshape(-1)
            if module_name in self.reselection_scores:
                candidate_indices, candidate_grads, candidate_grads_sq, samples = self.reselection_scores[module_name]
                new_grads = grad[candidate_indices]
                candidate_grads += new_grads
                candidate_grads_sq.addcmul_(new_grads, new_grads)
                samples += 1
            else:
                if candidates is not None:
                    candidate_indices = candidates
                else:
                    num_candidates = len(m.sft_delta[m.active_adapter].values)
                    _, candidate_indices = torch.topk(
                        torch.abs(grad),
                        num_candidates,
                        largest=True,
                        sorted=False,
                    )
                candidate_grads = grad[candidate_indices]
                self.reselection_scores[module_name] = (
                    candidate_indices.to(m.sft_delta[m.active_adapter].indices.dtype),
                    candidate_grads,
                    candidate_grads * candidate_grads,
                    torch.ones_like(candidate_grads)
                )
                del candidates

        return _gradient_accumulation_hook

    def get_wanda_hook(self, module_name):
        def _wanda_hook(inp):
            inp = inp.detach()
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            tmp = inp.shape[0]
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
            
            inp = inp.type(torch.float32)

            if module_name in self.scaler_row:
                self.scaler_row[module_name] *= self.nsamples[module_name] / (self.nsamples[module_name]+tmp)
                self.nsamples[module_name] += tmp
                self.scaler_row[module_name] += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples[module_name]
            else:
                self.nsamples[module_name] = tmp
                self.scaler_row[module_name] = torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples[module_name]
        return _wanda_hook

    def active_sft_deltas(self):
        for n, m in self.model.named_modules():
            if (
                isinstance(m, Linear) and
                m.active_adapter is not None and
                m.active_adapter in m.sft_delta
            ):
                yield (
                    f'{n}.sft_delta.{m.active_adapter}',
                    m.sft_delta[m.active_adapter]
                )

    def drop_magnitude(self, num_to_reallocate, delta, module_name):
        _, changing_indices = torch.topk(
            torch.abs(delta.values),
            num_to_reallocate,
            largest=False,
            sorted=True,
        )
        return changing_indices


    def drop_wanda(self, num_to_reallocate, delta, module_name):
        
        W = torch.zeros(delta.shape, device=delta.values.device)
        W.view(-1).scatter_reduce_(
            0,
            delta.indices.long(),
            delta.values,
            "sum",
            include_self=False,
        )
        W_metric = torch.abs(W) * torch.sqrt(self.scaler_row[module_name].reshape((1,-1)))

        relevant_scores = W_metric.view(-1)[delta.indices]

        _, changing_indices = torch.topk(
            relevant_scores,
            num_to_reallocate,
            largest=False,
            sorted=True,
        )
        return changing_indices

    @torch.no_grad()
    def select_gse_global(self, change_proportion, drop):
        n_replacements = 0
        total_params = 0

        betas = {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                betas[p] = group['betas']
        
        # If the parameters are selected in globa level compute which indeces are selected first
        if self.sft_config.selection_level == "global":
            grads = None
            # Collect all candidate gradients
            for module_name, (
                candidate_indices,
                candidate_grads,
                candidate_grads_sq,
                candidate_samples
            ) in self.reselection_scores.items():
                if grads is None:
                    grads = torch.abs(candidate_indices)
                else:
                    grads = torch.cat((grads, torch.abs(candidate_indices)))

            num_grads = grads.shape[0]

            # compute number of changing parameters
            num_to_reallocate = int(min(self.num_delta_weights * change_proportion, num_grads))
            
            # get candidates with biggest gradient
            _ ,indices = torch.topk(
                grads,
                num_to_reallocate,
                largest=True,
                sorted=True,
            )

            del grads

            mask = torch.zeros(num_grads, dtype=torch.bool, device=indices.device)
            mask[indices] = True

            # save grwoing candidates for later
            growing_candidates = {}

            for module_name, (
                candidate_indices,
                candidate_grads,
                candidate_grads_sq,
                candidate_samples
            ) in self.reselection_scores.items():
                relevant = mask[:candidate_indices.shape[0]]
                mask = mask[candidate_indices.shape[0]:]
                growing_indices = relevant.nonzero().squeeze() # candidate_indices[relevant]
                growing_scores = torch.abs(candidate_grads)[relevant]
                growing_candidates[module_name] = (growing_scores, growing_indices)
            assert len(mask) == 0

            
            weights = None
            for n, m in self.model.named_modules():
                if (
                    isinstance(m, Linear) and
                    m.active_adapter is not None and
                    m.active_adapter in m.sft_delta
                ):
                    if weights is None:
                        weights = torch.abs(m.sft_delta[m.active_adapter].values)
                    else:
                        weights = torch.cat((weights, torch.abs(m.sft_delta[m.active_adapter].values)))

            # get candidates with smallest magnitude
            _ ,indices = torch.topk(
                weights,
                num_to_reallocate,
                largest=False,
                sorted=True,
            )
            num_weights = weights.shape[0]
            del weights
            mask = torch.zeros(num_weights, dtype=torch.bool, device=indices.device)
            mask[indices] = True

            # save growing candidates for later
            dropping_weights = {}

            for n, m in self.model.named_modules():
                if (
                    isinstance(m, Linear) and
                    m.active_adapter is not None and
                    m.active_adapter in m.sft_delta
                ):
                    delta = m.sft_delta[m.active_adapter]
                    relevant = mask[:delta.values.shape[0]]
                    mask = mask[delta.values.shape[0]:]
                    dropping_indices = relevant.nonzero().squeeze() 
                    dropping_weights[n] = dropping_indices
            assert len(mask) == 0

        for module_name, (
            candidate_indices,
            candidate_grads,
            candidate_grads_sq,
            candidate_samples
        ) in self.reselection_scores.items():
            m = self.model.get_submodule(module_name)
            delta = m.sft_delta[m.active_adapter]
            delta.values.grad = None

            changing_indices = dropping_weights[module_name]
            num_outgoing = changing_indices.shape[0]
            
            # Find the k deltas with smallest absolute values
            outgoing_params = delta.indices[changing_indices]
            # binary mask of weights to drop
            is_outgoing = torch.zeros(
                [delta.dense_numel],
                dtype=torch.bool,
                device=outgoing_params.device,
            )
            is_outgoing[outgoing_params] = True
            # binary mask of currently active weights
            is_current = torch.zeros(
                [delta.dense_numel],
                dtype=torch.bool,
                device=delta.indices.device,
            )
            is_current[delta.indices] = True
            # weights that will stil be active after dropping
            is_remaining = is_current & ~is_outgoing

            # take the top k growth candidates with highest gradient magnitudes
            best_scores, best_candidate_indices = growing_candidates[module_name]

            # don't consider growing any already active candidate
            is_valid_candidate = ~is_remaining[best_candidate_indices]
            
            incoming_params = candidate_indices[best_candidate_indices][is_valid_candidate]

            assert torch.all(incoming_params < delta.dense_numel)

            incoming_grads = candidate_grads[best_candidate_indices][is_valid_candidate]
            incoming_grads_sq = candidate_grads_sq[best_candidate_indices][is_valid_candidate]
            incoming_samples = candidate_samples[best_candidate_indices][is_valid_candidate]
            # binary mask of weights to grow
            is_incoming = torch.zeros(
                [delta.dense_numel],
                dtype=torch.bool,
                device=incoming_params.device,
            )
            is_incoming[incoming_params] = True

            outgoing_is_incoming = is_incoming[outgoing_params]
            changing_indices = changing_indices[~outgoing_is_incoming]
            incoming_is_outgoing = is_outgoing[incoming_params]


            incoming_params = incoming_params[~incoming_is_outgoing]
            incoming_grads = incoming_grads[~incoming_is_outgoing]
            incoming_grads_sq = incoming_grads_sq[~incoming_is_outgoing]
            incoming_samples = incoming_samples[~incoming_is_outgoing]
            assert torch.all(incoming_params < delta.dense_numel)

            # seed the optimizer momenta appropriately
            incoming_grads /= incoming_samples
            incoming_grads_sq /= incoming_samples
            incoming_ages = incoming_samples / self.grad_accumulation_steps
            beta1, beta2 = betas[delta.values]
            # bias counter-correction: these are unbiased estimates of the momenta,
            # so bias them in order that they will be unbiased after Adam's bias
            # correction
            incoming_grads *= (1.0 - beta1 ** incoming_ages)
            incoming_grads_sq *= (1.0 - beta2 ** incoming_ages)

            init_momenta={
                    'age': incoming_ages,
                    'exp_avg': incoming_grads,
                    'exp_avg_sq': incoming_grads_sq,
                }

            prev_parameter = delta.values

            if torch.sum(is_incoming) >  torch.sum(is_outgoing):
                overhead = torch.sum(is_incoming) - torch.sum(is_outgoing)
                prev_delta_size = delta.indices.shape[0]

                changing_indices = torch.cat((changing_indices, torch.arange(delta.indices.shape[0], delta.indices.shape[0]+overhead, device=incoming_params.device)))
                delta.indices = torch.cat((delta.indices, torch.zeros((overhead), dtype=delta.indices.dtype, device=delta.indices.device)))
                delta.values = torch.nn.Parameter(torch.cat((delta.values, torch.zeros((overhead), dtype=delta.values.dtype, device=delta.values.device))))
                

                delta.indices[changing_indices] = incoming_params.to(delta.indices.dtype)
                delta.values[changing_indices] = 0.0
                

                optimizer_resize_param(
                    self.optimizer, 
                    prev_parameter, 
                    changing_indices, 
                    init_momenta, 
                    None, 
                    new_param=delta.values,
                    weight_change=overhead,
                )
                assert torch.all(delta.indices < delta.dense_numel)

                assert prev_delta_size + overhead == delta.values.shape[0]
            elif torch.sum(is_incoming) < torch.sum(is_outgoing):
                overhead = torch.sum(is_outgoing) - torch.sum(is_incoming)
                is_staying = torch.ones(
                    delta.indices.shape,
                    dtype=torch.bool,
                    device=delta.values.device,
                )
                is_staying[changing_indices] = False
                staying = is_staying.nonzero().squeeze() 
                prev_delta_size = delta.indices.shape[0]

                delta.indices = torch.cat((delta.indices[staying], incoming_params.to(delta.indices.dtype)))
                delta.values = torch.nn.Parameter(torch.cat((delta.values[staying], torch.zeros((incoming_params.shape[0]), dtype=delta.values.dtype, device=delta.values.device))))
                optimizer_resize_param(
                    self.optimizer, 
                    prev_parameter, 
                    changing_indices, 
                    init_momenta, 
                    staying, 
                    new_param=delta.values,
                    weight_change=-overhead
                )
                assert prev_delta_size - overhead == delta.values.shape[0]
            else:
                # update delta indices and values
                delta.indices[changing_indices] = incoming_params.to(delta.indices.dtype)
                delta.values[changing_indices] = 0.0
                optimizer_resize_param(
                    self.optimizer, 
                    prev_parameter, 
                    changing_indices, 
                    init_momenta, 
                    staying, 
                    new_param=delta.values,
                    weight_change=0
                )
            
            assert delta.indices.shape == delta.values.shape
            assert torch.unique(delta.indices).shape == delta.indices.shape
            assert torch.all(delta.indices < delta.dense_numel)
            assert delta.values.requires_grad
            n_replacements += len(changing_indices)
            total_params += len(delta.indices)

        # import ipdb; ipdb.set_trace()
        logger.info(
            f'Replacing {n_replacements} ({100*n_replacements/total_params:.4f}%)'
        )

           
    @torch.no_grad()
    def select_rigl(self, change_proportion, drop):
        n_replacements = 0
        total_params = 0

        betas = {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                betas[p] = group['betas']

        for module_name, (
            candidate_indices,
            candidate_grads,
            candidate_grads_sq,
            candidate_samples
        ) in self.reselection_scores.items():
            m = self.model.get_submodule(module_name)
            delta = m.sft_delta[m.active_adapter]
            delta.values.grad = None

            num_to_reallocate = int(len(delta.values) * change_proportion)
            # Find the k deltas with smallest absolute values
            changing_indices = drop(num_to_reallocate, delta, module_name)
            outgoing_params = delta.indices[changing_indices]
            # binary mask of weights to drop
            is_outgoing = torch.zeros(
                [delta.dense_numel],
                dtype=torch.bool,
                device=outgoing_params.device,
            )
            is_outgoing[outgoing_params] = True
            assert torch.sum(is_outgoing) == num_to_reallocate
            # binary mask of currently active weights
            is_current = torch.zeros(
                [delta.dense_numel],
                dtype=torch.bool,
                device=delta.indices.device,
            )
            is_current[delta.indices] = True
            # weights that will stil be active after dropping
            is_remaining = is_current & ~is_outgoing

            # don't consider growing any already active candidate
            is_valid_candidate = ~is_remaining[candidate_indices]
            candidate_indices = candidate_indices[is_valid_candidate]
            candidate_grads = candidate_grads[is_valid_candidate]
            candidate_grads_sq = candidate_grads_sq[is_valid_candidate]
            candidate_samples = candidate_samples[is_valid_candidate]
            candidate_scores = torch.abs(candidate_grads)
            # take the top k growth candidates with highest gradient magnitudes
            best_scores, best_candidate_indices = torch.topk(
                candidate_scores,
                min(num_to_reallocate, len(candidate_grads)),
                largest=True,
                sorted=True,
            )
            incoming_params = candidate_indices[best_candidate_indices]
            incoming_grads = candidate_grads[best_candidate_indices]
            incoming_grads_sq = candidate_grads_sq[best_candidate_indices]
            incoming_samples = candidate_samples[best_candidate_indices]
            # binary mask of weights to grow
            is_incoming = torch.zeros(
                [delta.dense_numel],
                dtype=torch.bool,
                device=incoming_params.device,
            )
            is_incoming[incoming_params] = True

            # filter out weights which have been selected to be dropped and
            # grown simultaneously
            assert torch.sum(is_incoming) == len(best_candidate_indices)
            outgoing_is_incoming = is_incoming[outgoing_params]
            changing_indices = changing_indices[~outgoing_is_incoming]
            incoming_is_outgoing = is_outgoing[incoming_params]
            assert torch.sum(outgoing_is_incoming) == torch.sum(incoming_is_outgoing)
            incoming_params = incoming_params[~incoming_is_outgoing]
            incoming_grads = incoming_grads[~incoming_is_outgoing]
            incoming_grads_sq = incoming_grads_sq[~incoming_is_outgoing]
            incoming_samples = incoming_samples[~incoming_is_outgoing]
            changing_indices = changing_indices[:len(incoming_params)]

            n_replacements += len(changing_indices)
            total_params += len(delta.indices)

            # update delta indices and values
            delta.indices[changing_indices] = incoming_params.to(delta.indices.dtype)
            delta.values[changing_indices] = 0.0

            # seed the optimizer momenta appropriately
            incoming_grads /= incoming_samples
            incoming_grads_sq /= incoming_samples
            incoming_ages = incoming_samples / self.grad_accumulation_steps
            beta1, beta2 = betas[delta.values]
            # bias counter-correction: these are unbiased estimates of the momenta,
            # so bias them in order that they will be unbiased after Adam's bias
            # correction
            incoming_grads *= (1.0 - beta1 ** incoming_ages)
            incoming_grads_sq *= (1.0 - beta2 ** incoming_ages)
            update_optimizer(
                self.optimizer,
                delta.values,
                changing_indices,
                init_momenta={
                    'age': incoming_ages,
                    'exp_avg': incoming_grads,
                    'exp_avg_sq': incoming_grads_sq,
                }
            )

        logger.info(
            f'Replacing {n_replacements} ({100*n_replacements/total_params:.4f}%)'
        )

    @torch.no_grad()
    def select_sm3(self, p, drop):
        n_replacements = 0
        total_params = 0

        for _, delta in self.active_sft_deltas():
            num_to_reallocate = int(len(delta.values) * p)
            changing_indices = drop(num_to_reallocate, delta)

            is_current = torch.zeros(
                [delta.dense_numel],
                dtype=torch.bool,
                device=delta.indices.device,
            )
            is_current[delta.indices] = True
            is_valid_candidate = ~is_current

            optimizer_state = self.optimizer.state[delta.values]
            row_grads_sq = optimizer_state['accumulator_0']
            col_grads_sq = optimizer_state['accumulator_1']
            # take outer product of row and column wise SM3 buffers
            # (here we assume 2D parameter tensors).
            estimated_momenta = torch.outer(row_grads_sq, col_grads_sq)
            estimated_momenta = estimated_momenta.view(-1)[is_valid_candidate]
            candidate_indices = torch.arange(
                0,
                delta.dense_numel,
                device=is_valid_candidate.device,
            )
            candidate_indices = candidate_indices[is_valid_candidate]
            _, best_candidate_indices = torch.topk(
                estimated_momenta,
                num_to_reallocate,
                largest=True,
                sorted=False,
            )
            incoming_params = candidate_indices[best_candidate_indices]

            n_replacements += len(changing_indices)
            total_params += len(delta.indices)

            delta.indices[changing_indices] = incoming_params.to(delta.indices.dtype)
            delta.values[changing_indices] = 0.0

        logger.info(
            f'Replacing {n_replacements} ({100*n_replacements/total_params:.4f}%)'
        )
        logger.info(f"Number of parameters in delta: {total_params}")


class SelectorStepCallback(TrainerCallback):

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        self.trainer._selector.step()


def SftTrainer(_Trainer):
    """
    Wraps a Trainer or subclass thereof for SFT training. The resulting class
    should be constructed with a SftModel as the model and passing sft_config as
    a SftConfig instance.
    """

    class _SftTrainer(_Trainer):

        def __init__(
            self,
            *args,
            sft_config=None,
            **kwargs
        ):
            super().__init__(*args, **kwargs)
            logger.setLevel(self.args.get_process_log_level())

            if sft_config is None:
                raise ValueError('Missing sft_config')
            self.sft_config = sft_config

            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
            else:
                train_dataloader = self.get_train_dataloader()
                len_dataloader = len(train_dataloader)
                num_update_steps_per_epoch = (
                    len_dataloader //
                    self.args.gradient_accumulation_steps
                )
                num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)

            self._selector = SftSelector(
                self.model,
                self.create_optimizer(),
                self.sft_config,
                max_steps,
                self.args.gradient_accumulation_steps,
            )
            self.add_callback(SelectorStepCallback(self))

        def create_optimizer(self):
            if self.optimizer is None:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if p.requires_grad
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                ]

                _, optimizer_kwargs = _Trainer.get_optimizer_cls_and_kwargs(self.args)
                logger.info(f'optimizer_kwargs: {optimizer_kwargs}')

                if self.sft_config.selection_algorithm == "sm3":
                    deltas = {
                        delta.values: delta
                        for _1, _2, delta in self.model.active_deltas()
                    }

                    self.optimizer = SftSM3(
                        optimizer_grouped_parameters,
                        deltas,
                        **optimizer_kwargs
                    )
                else:
                    self.optimizer = SftAdamW(optimizer_grouped_parameters, **optimizer_kwargs)

            return self.optimizer

    return _SftTrainer
