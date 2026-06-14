# signinum-jpeg-metal

Metal adapter for Signinum JPEG decode and baseline encode paths on macOS.

Supported paths return resident Metal outputs or use Metal kernels for selected
adapter stages. Explicit Metal requests are strict and fail for unsupported
shapes.
