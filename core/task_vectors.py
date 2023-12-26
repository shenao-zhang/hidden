from typing import Dict, List, Optional, Tuple, Union, Iterable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.data.datasets.few_shot_dataset import FewShotDataset
from core.data.tasks.task import Task
from core.models.context_managers.forward_modifiers.hidden_injector import HiddenInjector
from core.models.utils.inference import (
    batch_forward,
    batch_generate,
    decode_predictions,
    get_input_type,
    modified_forward,
    tokenize_datasets,
    traced_forward,
)
from core.models.utils.llm_layers import get_layers
from core.utils.nested import nested_apply


def run_icl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_datasets: List[FewShotDataset],
    include_train: bool = True,
) -> List[str]:
    format_dataset_kwargs = {"include_train": include_train}
    inputs = tokenize_datasets(tokenizer, train_datasets, format_dataset_kwargs=format_dataset_kwargs)
    new_ids = batch_generate(model, tokenizer, inputs=inputs, generate_kwargs={"max_new_tokens": 300})
    predictions = decode_predictions(new_ids, tokenizer)
    print("icl", predictions)
    return predictions


def run_task_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    dev_datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
    multi_context: bool = False,
):
    dev_accuracy_by_layer = task_vector_accuracy_by_layer(
        model,
        tokenizer,
        task,
        dev_datasets,
        layers_to_test=layers_to_test,
        multi_context=multi_context,
    )
    best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))

    task_hiddens = get_task_hiddens(model, tokenizer, test_datasets, multi_context=multi_context)
    predictions = modulated_generate(
        model,
        tokenizer,
        test_datasets,
        task_hiddens=task_hiddens,
        intermediate_layer=best_intermediate_layer,
    )

    return predictions, dev_accuracy_by_layer, task_hiddens

def stack_helper(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_datasets: List[FewShotDataset],
    task_hiddens: torch.Tensor,
    multi_context: bool = False,
    best_intermediate_layer: int = 1,
    train_idx: int = 0
):
    """
    new_ins = []
    new_outs = []
    for idx, dataset in enumerate(test_datasets):
        exclude_samples = [dataset.test_input] + dataset.train_inputs
        new_in = task.sample_inputs(num_examples, exclude=exclude_samples)
        new_ins.append(new_in)
        new_out = [task.calc_output(x) for x in new_in]
        new_outs.append(new_out)
    new_test_datasets = [
        FewShotDataset(
            train_inputs=new_ins[idx],
            train_outputs=new_outs[idx],
            test_input=dataset.test_input,
            test_output=task.calc_output(dataset.test_input),
        )
        for idx, dataset in enumerate(test_datasets)
    ]
    """
    new_task_hiddens = get_task_hiddens(model, tokenizer, train_datasets, multi_context=multi_context,
                                        moderate=True, prev_hiddens=task_hiddens,
                                        prev_intermediate_layer=best_intermediate_layer - 1, train_idx=train_idx) # -1
    stack_predictions = modulated_generate(
        model,
        tokenizer,
        train_datasets,
        task_hiddens=new_task_hiddens,
        intermediate_layer=best_intermediate_layer,
        #      include_train=True
    )
    return new_task_hiddens, stack_predictions

def run_stack_task_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_datasets: List[FewShotDataset],
    multi_context: bool = False,
    best_intermediate_layer: int = 1,
):
    task_hiddens = get_task_hiddens(model, tokenizer, train_datasets, multi_context=multi_context)
    predictions = modulated_generate(
        model,
        tokenizer,
        train_datasets,
        task_hiddens=task_hiddens,
        intermediate_layer=best_intermediate_layer,
    )
    stack_predictions_list = []
    print('0: ', predictions)
    for train_idx in range(10):
        task_hiddens, stack_predictions = stack_helper(model, tokenizer, train_datasets, task_hiddens, multi_context,
                                                       best_intermediate_layer, train_idx + 1)
        stack_predictions_list.append(stack_predictions)
        print(f"{train_idx + 1}: ", stack_predictions)
    return predictions, task_hiddens, stack_predictions_list

def run_overriding_task_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    overriding_datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
):
    dev_accuracy_by_layer = task_vector_accuracy_by_layer(
        model,
        tokenizer,
        task,
        overriding_datasets,
        layers_to_test=layers_to_test,
    )
    best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))

    task_hiddens_datasets = test_datasets if overriding_datasets is None else overriding_datasets
    task_hiddens = get_task_hiddens(model, tokenizer, task_hiddens_datasets)

    predictions = modulated_generate(
        model,
        tokenizer,
        test_datasets,
        task_hiddens=task_hiddens,
        intermediate_layer=best_intermediate_layer,
        include_train=True,
    )

    return predictions, dev_accuracy_by_layer, task_hiddens


def get_multi_context_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
) -> torch.Tensor:
    inputs = tokenize_datasets(tokenizer, datasets)

    outputs, forward_trace = traced_forward(model, inputs=inputs)

    task_hiddens = forward_trace.residual_stream.hidden[:, :, -1, :]

    # for each dataset, average task hiddens from other datasets that did not include the test_input from the current dataset
    mask = torch.ones(len(datasets), len(datasets))
    for i, dataset in enumerate(datasets):
        for j, other_dataset in enumerate(datasets):
            if dataset.test_input in other_dataset.train_inputs or dataset.test_input == other_dataset.test_input:
                mask[i, j] = 0

    task_hiddens = torch.cat([task_hiddens[mask[i].bool()].mean(dim=0).unsqueeze(0) for i in range(len(datasets))])

    task_hiddens = task_hiddens[:, 1:]  # the first one is the embedding layer

    return task_hiddens  # (num_datasets, num_layers, hidden_size)

def get_single_context_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    datasets: List[FewShotDataset],
    num_test_inputs_to_avg: int = 1,  # 2
    train_idx: int = 0
) -> torch.Tensor:
    """
    new_datasets = [
        FewShotDataset(
            train_inputs=dataset.train_inputs,
            train_outputs=dataset.train_outputs,
            test_input=test_input,
            test_output=task.calc_output(test_input),
        )
        for dataset in datasets
        for test_input in task.sample_inputs(num_test_inputs_to_avg, exclude=(dataset.test_input,))
    ]
    """

    inputs = tokenize_datasets(tokenizer, datasets, train_idx)

    # TODO: replace traced forward with a regular forward and rely on huggingface's saved hidden states
    outputs, forward_trace = traced_forward(model, inputs=inputs)
    task_hiddens = forward_trace.residual_stream.hidden[:, :, -1, :]
    _, num_layers, hidden_size = task_hiddens.shape
    task_hiddens = task_hiddens.view(len(datasets), num_test_inputs_to_avg, num_layers, hidden_size).mean(dim=1)

    task_hiddens = task_hiddens[:, 1:]  # the first one is the embedding layer

    return task_hiddens  # (num_datasets, num_layers, hidden_size)

def stack_get_single_context_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    datasets: List[FewShotDataset],
    prev_hiddens,
    prev_intermediate_layer: Union[int, torch.Tensor] = 2,
    num_test_inputs_to_avg: int = 1,  # 2
    train_idx: int = 0
) -> torch.Tensor:
   # for idx, train_in in enumerate(datasets[0].train_inputs):
   #     datasets[0].train_inputs[idx] = ' ' + train_in
    """
    new_datasets = [
        FewShotDataset(
            train_inputs=dataset.train_inputs,
            train_outputs=dataset.train_outputs,
            test_input=test_input,
            test_output=task.calc_output(test_input),
        )
        for dataset in datasets
        for test_input in task.sample_inputs(num_test_inputs_to_avg, exclude=(dataset.test_input,))
    ]
    """
    inputs = tokenize_datasets(tokenizer, datasets, train_idx)

    # Stack hidden states
    if isinstance(prev_intermediate_layer, int):
        prev_intermediate_layer = torch.tensor(prev_intermediate_layer).repeat(len(inputs["input_ids"]))
    injection_positions = 0 * torch.ones_like(prev_intermediate_layer, dtype=torch.long)  # -1
    prev_hiddens = prev_hiddens[torch.arange(len(prev_intermediate_layer)), prev_intermediate_layer]

    forward_modifiers = [
        HiddenInjector(
            model,
            injection_layers=prev_intermediate_layer,
            injection_positions=injection_positions,
            hiddens_to_inject=prev_hiddens,
        )
    ]
    # TODO: replace traced forward with a regular forward and rely on huggingface's saved hidden states
    outputs, forward_trace = traced_forward(model, inputs=inputs, forward_modifiers=forward_modifiers)

    task_hiddens = forward_trace.residual_stream.hidden[:, :, -1, :]
    _, num_layers, hidden_size = task_hiddens.shape
    task_hiddens = task_hiddens.view(len(datasets), num_test_inputs_to_avg, num_layers, hidden_size).mean(dim=1)

    task_hiddens = task_hiddens[:, 1:]  # the first one is the embedding layer

    return task_hiddens  # (num_datasets, num_layers, hidden_size)


def get_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    datasets: List[FewShotDataset],
    multi_context: bool = False,
    moderate: bool = False,
    prev_hiddens=None,
    prev_intermediate_layer: int = 1,
    train_idx: int = 1
) -> torch.Tensor:
    if moderate:
        return stack_get_single_context_task_hiddens(model, tokenizer, datasets, prev_hiddens=prev_hiddens,
                                                     prev_intermediate_layer=prev_intermediate_layer, train_idx=train_idx)
    if not multi_context:
        return get_single_context_task_hiddens(model, tokenizer, datasets) # train_idx = 0


def modulated_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_datasets: List[FewShotDataset],
    task_hiddens: torch.tensor,
    intermediate_layer: Union[int, torch.Tensor],
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    return_task_hiddens: bool = False,
    include_train: bool = False,
) -> List[str]:
    inputs = tokenize_datasets(tokenizer, test_datasets, format_dataset_kwargs={"include_train": include_train})

    first_forward_outputs = modulated_forward(
        model,
        inputs=inputs,
        task_hiddens=task_hiddens,
        intermediate_layer=intermediate_layer,
        past_key_values=past_key_values,
    )
    first_predicted_token_ids = first_forward_outputs.logits[:, -1].argmax(dim=-1).unsqueeze(-1)
    answers = decode_predictions(first_predicted_token_ids, tokenizer)

    if return_task_hiddens:
        return answers, task_hiddens
    return answers


def modulated_forward(
    model: PreTrainedModel,
    inputs: Dict,
    task_hiddens: torch.Tensor,
    intermediate_layer: int,
    batch_size: Optional[int] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
):
    # TODO: move all this to the HiddenInjector class
    if isinstance(intermediate_layer, int):
        intermediate_layer = torch.tensor(intermediate_layer).repeat(len(inputs["input_ids"]))
    injection_positions = -1 * torch.ones_like(intermediate_layer, dtype=torch.long)
    task_hiddens = task_hiddens[torch.arange(len(intermediate_layer)), intermediate_layer]
    forward_modifiers = [
        HiddenInjector(
            model,
            injection_layers=intermediate_layer,
            injection_positions=injection_positions,
            hiddens_to_inject=task_hiddens,
        )
    ]

    if past_key_values is not None:
        inputs[get_input_type(inputs)] = inputs[get_input_type(inputs)][:, -1].unsqueeze(1)

    first_forward_outputs = modified_forward(
        model,
        inputs=inputs,
        forward_kwargs={"past_key_values": past_key_values},
        forward_modifiers=forward_modifiers,
        batch_size=len(inputs["input_ids"]),  # TODO: need to enable batched forward with HiddenInjector
    )

    return first_forward_outputs


def task_vector_accuracy_by_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
    multi_context: bool = False,
    moderate: bool = False,
    prev_hiddens=None,
    prev_intermediate_layer: int = 1
) -> Dict[int, float]:
    if layers_to_test is None:
        num_layers = len(get_layers(model))
        layers_to_test = range(num_layers)

    # Get task hiddens
    task_hiddens = get_task_hiddens(model, tokenizer, datasets, multi_context=multi_context, moderate=moderate,
                                    prev_hiddens=prev_hiddens, prev_intermediate_layer=prev_intermediate_layer)

    # Get input past_key_values
    inputs = tokenize_datasets(tokenizer, datasets, format_dataset_kwargs={"include_train": False})
    outputs = batch_forward(model, inputs=inputs, forward_kwargs={"use_cache": True})
    past_key_values = outputs.past_key_values
    past_key_values = nested_apply(past_key_values, lambda x: x[:, :, :-1])  # remove last token from past_key_values
    inputs["input_ids"] = inputs["input_ids"][:, -1].unsqueeze(1)

    # Find best intermediate layer using dev set
    accuracies = []
    for layer_num in layers_to_test:
        answers = modulated_generate(
            model,
            tokenizer,
            datasets,
            intermediate_layer=layer_num,
            task_hiddens=task_hiddens,
            past_key_values=past_key_values,
        )

        accuracy = calculate_accuracy_on_datasets(task, answers, datasets)
        accuracies.append(accuracy)
    accuracy_by_layer = {layer: accuracy for layer, accuracy in zip(layers_to_test, accuracies)}

    return accuracy_by_layer


def continue_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: Dict,
    first_forward_outputs: CausalLMOutputWithPast,
    test_datasets: List[FewShotDataset],
) -> List[str]:
    """
    Continue generation after the first token. This is currently not supported.
    """
    first_predicted_token_ids = first_forward_outputs.logits[:, -1].argmax(dim=-1).unsqueeze(-1)

    new_input_ids = first_predicted_token_ids
    new_attention_mask = torch.ones_like(new_input_ids)

    full_input_ids = torch.cat([inputs["input_ids"], new_input_ids], dim=-1)
    full_attention_mask = torch.cat([inputs["attention_mask"], new_attention_mask], dim=-1)

    # full_input_ids = new_input_ids
    # full_attention_mask = new_attention_mask

    past_key_values = first_forward_outputs.past_key_values

    max_new_tokens = 1  # Right now we don't support multi-token outputs

    if max_new_tokens > 0:
        output_ids = model.generate(
            **{"input_ids": full_input_ids, "attention_mask": full_attention_mask},
            do_sample=False,
            max_new_tokens=max_new_tokens,
            past_key_values=past_key_values,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        output_ids = full_input_ids

    new_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
    answers = decode_predictions(new_ids, tokenizer)

    return answers
