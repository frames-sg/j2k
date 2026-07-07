# j2k-cli

Command-line entry point for J2K.

The current CLI focuses on inspection and conservative smoke-test workflows.
Full compress/decompress CLI parity is not claimed by this package.

Supported commands:

```bash
j2k inspect <file>
j2k transcode <input.jpg> <output.j2k> --htj2k --lossless-53
```

`transcode` writes an HTJ2K codestream from a supported JPEG input and reports
the source geometry, component count, output bytes, and timing/dispatch summary.

## Links

- API docs: <https://docs.rs/j2k-cli>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
