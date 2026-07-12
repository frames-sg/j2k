# j2k-profile

Small profiling and route-summary helper crate for J2K.

Codec and adapter crates use it to emit stable profile rows and route decisions
without each crate reimplementing environment parsing or summary formatting.

## Bounded, fallible ownership

Owned fields, parsed rows, summaries, and formatted output enforce explicit
`ProfileLimits`. Convenience entry points use the default limits and return
`ProfileResult`; callers with tighter service limits can use the corresponding
`*_with_limits` functions or constructors.

```rust
use j2k_profile::{parse_profile_line, ParsedProfileKind, ProfileResult};

fn parse_row(line: &str) -> ProfileResult<()> {
    if let Some(fields) = parse_profile_line(line)? {
        if fields.kind() == ParsedProfileKind::Row {
            let _codec = fields.get("codec");
        }
    }
    Ok(())
}
```

Profile owner graphs are move-only. Optional codec profiling helpers print a
`j2k_profile_error` diagnostic when allocation, input, or formatting fails;
profiling failures do not alter the codec operation's success result.

## Links

- API docs: <https://docs.rs/j2k-profile>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
