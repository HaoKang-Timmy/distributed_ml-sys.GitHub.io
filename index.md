## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/timmywanttolearn/distributed_ml-sys.GitHub.io/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

# Betacat's research experience

## 1 Gpipe, source code reading

## 1.1 First part, Class Gpipe, initialization

```python
class GPipe(Module):
```

Usage

```
        model = nn.Sequential(a, b, c, d)
        model = GPipe(model, balance=[1, 1, 1, 1], chunks=8)
        output = model(input)
```

Input

```python
    Keyword Args:
        devices (iterable of devices):
            devices to use (default: all CUDA devices)
        chunks (int):
            number of micro-batches (default: ``1``)
        checkpoint (str):
            when to enable checkpointing, one of ``'always'``,
            ``'except_last'``, or ``'never'`` (default: ``'except_last'``)
        deferred_batch_norm (bool):
            whether to use deferred BatchNorm moving statistics (default:
            :data:`False`, see :ref:`Deferred Batch Normalization` for more
            details)
```

Initialization:1

```python
        if deferred_batch_norm:
            module = DeferredBatchNorm.convert_deferred_batch_norm(module, chunks)
```

```python
    def convert_deferred_batch_norm(cls, module: TModule, chunks: int = 1) -> TModule:
        """Converts a :class:`nn.BatchNorm` or underlying
        :class:`nn.BatchNorm`s into :class:`DeferredBatchNorm`::
            from torchvision.models.resnet import resnet101
            from torchgpipe.batchnorm import DeferredBatchNorm
            model = resnet101()
            model = DeferredBatchNorm.convert_deferred_batch_norm(model)
        """
        if isinstance(module, DeferredBatchNorm) and module.chunks is chunks:
            return cast(TModule, module)

        module_output: nn.Module = module

        if isinstance(module, _BatchNorm) and module.track_running_stats:
            module_output = DeferredBatchNorm(module.num_features,
                                              module.eps,
                                              module.momentum,
                                              module.affine,
                                              chunks)
            if module.affine:
                module_output.register_parameter('weight', module.weight)
                module_output.register_parameter('bias', module.bias)
            module_output.register_buffer('running_mean', module.running_mean)
            module_output.register_buffer('running_var', module.running_var)
            module_output.register_buffer('num_batches_tracked', module.num_batches_tracked)

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_deferred_batch_norm(child, chunks))

        return cast(TModule, module_output)
```

change batchnorm to deferred bn

For defered bn, it compute at each micro-batches

https://github.com/kakaobrain/torchgpipe/blob/a1b4ee25574864e7650e7905a69ce156da9752ec/torchgpipe/batchnorm.py#L45

Initialization:2

```
        try:
            self.partitions, self.balance, self.devices = split_module(module, balance, devices)
```

Initialize:3

```
        self._copy_streams: List[List[AbstractStream]] = []
        self._skip_layout = inspect_skip_layout(self.partitions)
```





## 1.2 Second, Class Gpipe, forward

```
batches = microbatch.scatter(input, self.chunks)
```

```
        # Separate CUDA streams for copy.
        copy_streams = self._ensure_copy_streams()
```

Then it create Pipeline and run it

```
        pipeline = Pipeline(batches,
                            self.partitions,
                            self.devices,
                            copy_streams,
                            self._skip_layout,
                            checkpoint_stop)
        pipeline.run()
```

# 2 Class Pipeline

## 2.1 init

```python
    def __init__(self,
                 batches: List[Batch],
                 partitions: List[nn.Sequential],
                 devices: Optional[List[torch.device]] = None,
                 copy_streams: Optional[List[List[AbstractStream]]] = None,
                 skip_layout: Optional[SkipLayout] = None,
                 checkpoint_stop: int = 0,
                 ) -> None:
        self.batches = batches
        self.partitions = partitions

        if devices is None:
            devices = [torch.device('cpu') for _ in partitions]
        self.devices = devices

        if copy_streams is None:
            copy_streams = [[current_stream(d)] * len(batches) for d in devices]
        self.copy_streams = copy_streams

        if skip_layout is None:
            skip_layout = inspect_skip_layout(partitions)

        self.skip_layout = skip_layout
        self.checkpoint_stop = checkpoint_stop

```

## 2.2 run

```python
    def run(self) -> None:
        """Runs pipeline parallelism.
        It modifies the given batches in place.
        """
        batches = self.batches
        partitions = self.partitions
        devices = self.devices
        skip_layout = self.skip_layout

        m = len(batches)
        n = len(partitions)

        skip_trackers = [SkipTrackerThroughPotals(skip_layout) for _ in batches]

        with spawn_workers(devices) as (in_queues, out_queues):
            for schedule in clock_cycles(m, n):
                self.fence(schedule, skip_trackers)
                self.compute(schedule, skip_trackers, in_queues, out_queues)
```

For `SkipTrackerThroughPotals`, it save skipped tensors and make autograd engine stop trak them

```python
        with spawn_workers(devices) as (in_queues, out_queues):
            for schedule in clock_cycles(m, n):
                self.fence(schedule, skip_trackers)
                self.compute(schedule, skip_trackers, in_queues, out_queues)
```

### 2.2.1 fence

In `fence`,

```python
    def fence(self,
              schedule: List[Tuple[int, int]],
              skip_trackers: List[SkipTrackerThroughPotals],
              ) -> None:
        """Copies micro-batches after computation for the previous
        micro-batches.
        """
        batches = self.batches
        copy_streams = self.copy_streams
        skip_layout = self.skip_layout

        for i, j in schedule:
            # Ensure that batches[i-1] is executed after batches[i] in
            # backpropagation by an explicit dependency.
            if i != 0:
                depend(batches[i-1], batches[i])

            next_stream = copy_streams[j][i]

            for prev_j, ns, name in skip_layout.copy_policy(j):
                prev_stream = copy_streams[prev_j][i]
                skip_trackers[i].copy(batches[i], prev_stream, next_stream, ns, name)

            if j != 0:
                prev_stream = copy_streams[j-1][i]
                copy(batches[i], prev_stream, next_stream)
```

In depend,

```python
def depend(fork_from: Batch, join_to: Batch) -> None:
    fork_from[0], phony = fork(fork_from[0])
    join_to[0] = join(join_to[0], phony)
```

`fork`

```python
def fork(input: Tensor) -> Tuple[Tensor, Tensor]:
    """Branches out from an autograd lane of the given tensor."""
    if torch.is_grad_enabled() and input.requires_grad:
        input, phony = Fork.apply(input)
    else:
        phony = get_phony(input.device, requires_grad=False)

    return input, phony
```

`Fork` class

```python
class Fork(torch.autograd.Function):
    @staticmethod
    def forward(ctx: 'Fork', input: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore
        phony = get_phony(input.device, requires_grad=False)
        return input.detach(), phony.detach()

    @staticmethod
    def backward(ctx: 'Fork', grad_input: Tensor, grad_grad: Tensor) -> Tensor:  # type: ignore
        return grad_output
```

join(TODO why need phony)

```python
def join(input: Tensor, phony: Tensor) -> Tensor:
    """Merges two autograd lanes."""
    if torch.is_grad_enabled() and (input.requires_grad or phony.requires_grad):
        input = Join.apply(input, phony)

    return input
```

Join

```python
class Join(torch.autograd.Function):
    @staticmethod
    def forward(ctx: 'Join', input: Tensor, phony: Tensor) -> Tensor:  # type: ignore
        return input.detach()

    @staticmethod
    def backward(ctx: 'Join', grad_input: Tensor) -> Tuple[Tensor, None]:  # type: ignore
        return grad_input, None
```

`copy`

```python
def copy(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Copy.apply(prev_stream, next_stream, *batch)
```

`Copy`

```python
class Copy(torch.autograd.Function):
    """Copies tensors on specific streams."""
    @staticmethod
    def forward(ctx: Context,  # type: ignore
                prev_stream: AbstractStream,
                next_stream: AbstractStream,
                *input: Tensor,
                ) -> Tensors:
        ctx.prev_stream = prev_stream
        ctx.next_stream = next_stream

        output = []
        output_stream = current_stream(get_device(next_stream))

        with use_stream(prev_stream), use_stream(next_stream):
            for x in input:
                y = x.to(get_device(next_stream))
                output.append(y)

                # 'prev_stream' is not where 'x' has been allocated.
                record_stream(x, prev_stream)
                # 'y' has been allocated on 'next_stream'.
                # It might be used on the current stream captured as 'output_stream'.
                record_stream(y, output_stream)

        return tuple(output)

    @staticmethod
    def backward(ctx: Context,
                 *grad_output: Tensor,
                 ) -> Tuple[Optional[Tensor], ...]:
        prev_stream = ctx.prev_stream
        next_stream = ctx.next_stream

        grad_input: Deque[Tensor] = deque(maxlen=len(grad_output))
        input_stream = current_stream(get_device(prev_stream))

        with use_stream(prev_stream), use_stream(next_stream):
            for x in reversed(grad_output):
                y = x.to(get_device(prev_stream))
                grad_input.appendleft(y)

                # 'next_stream' is not where 'x' has been allocated.
                record_stream(x, next_stream)
                # 'y' has been allocated on 'prev_stream'.
                # It might be used on the current stream captured as 'input_stream'.
                record_stream(y, input_stream)

        grad_streams: Tuple[Optional[Tensor], ...] = (None, None)
        return grad_streams + tuple(grad_input)
```

The most important part is to set original tensor to new device.

TODO: why need grad_streams?

### 2.2.2 compute

```python
        for i, j in schedule:
            batch = batches[i]
            partition = partitions[j]

            # Synchronize with the copied input. ([1] in the diagram)
            if j != 0:
                wait(batch, copy_streams[j][i], streams[j])

            # Determine whether checkpointing or not.
            checkpoint = (i < checkpoint_stop)
            if checkpoint:
                def function(input: TensorOrTensors,
                             partition: nn.Sequential = partition,
                             skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                             ) -> TensorOrTensors:
                    with use_skip_tracker(skip_tracker):
                        return partition(input)

                chk = Checkpointing(function, batch)
                task = Task(streams[j], compute=chk.checkpoint, finalize=chk.recompute)
                del function, chk

            else:
                def compute(batch: Batch = batch,
                            partition: nn.Sequential = partition,
                            skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                            ) -> Batch:
                    with use_skip_tracker(skip_tracker):
                        return batch.call(partition)

                task = Task(streams[j], compute=compute, finalize=None)
                del compute

            # Compute tasks in parallel. ([2] in the diagram)
            in_queues[j].put(task)

        for i, j in schedule:
            ok, payload = out_queues[j].get()

            # Hold the first exception.
            if exc_info is not None:
                continue
            elif not ok:
                exc_info = cast(ExcInfo, payload)
                continue

            task, batch = cast(Tuple[Task, Batch], payload)

            # The copy stream synchronizes to copy the output. ([3] in the
            # diagram)
            if j != n-1:
                wait(batch, streams[j], copy_streams[j][i])

            # Finalize tasks. If checkpointing is enabled, here the
            # recomputation is scheduled at backpropagation. ([4] in the
            # diagram)
            with use_device(devices[j]):
                task.finalize(batch)

            batches[i] = batch

        # Fail at the first exception.
        if exc_info is not None:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
```

 `wait`

```python
def wait(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Wait.apply(prev_stream, next_stream, *batch)
```

`Wait`

```python
class Wait(torch.autograd.Function):
    """Synchronizes a stream to another stream.
    Place it just before you want to start an operation on the next stream,
    provided that all operations on the previous stream are done.
    """
    @staticmethod
    def forward(ctx: Context,  # type: ignore
                prev_stream: AbstractStream,
                next_stream: AbstractStream,
                *input: Tensor,
                ) -> Tensors:
        ctx.prev_stream = prev_stream
        ctx.next_stream = next_stream

        wait_stream(next_stream, prev_stream)

        return tuple(x.detach() for x in input)

    @staticmethod
    def backward(ctx: Context,
                 *grad_input: Tensor,
                 ) -> Tuple[Optional[Tensor], ...]:
        prev_stream = ctx.prev_stream
        next_stream = ctx.next_stream

        wait_stream(prev_stream, next_stream)

        grad_streams: Tuple[Optional[Tensor], ...] = (None, None)
        return grad_streams + grad_input
```

Wait_stream and synchronize analyse

https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html#:~:text=operations%20are%20affected.-,wait_stream,-(stream)



# something about cuda stream

https://pytorch.org/docs/stable/notes/cuda.html#:~:text=CUDA-,streams,-A%20CUDA%20stream
