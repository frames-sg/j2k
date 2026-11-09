// SPDX-License-Identifier: MIT OR Apache-2.0

//! The encode kernels must be bit-exact with the CPU encoder. The runtime
//! loads their PTX with `cuModuleLoadData` and no JIT options, so ptxas runs
//! with `--fmad=true` and may fuse any un-rounded `mul.f32` into a dependent
//! un-rounded `add.f32`/`sub.f32`; whether that happens would then depend on
//! the driver. These tests fail on such a pair, and on approximate division,
//! square root or reciprocal, in the emitted encode PTX. Arithmetic that must
//! round once uses an explicit `fma.rn.f32`.

/// One finding: the PTX function and the offending instruction.
type Finding = (String, String);

/// Un-rounded `mul.f32` results feeding an un-rounded `add.f32`/`sub.f32`
/// in the same function, and approximate float operations, in `ptx`.
fn rounding_hazards(ptx: &str) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut function = String::new();
    let mut unrounded_products: Vec<String> = Vec::new();
    for line in ptx.lines() {
        let line = line.trim();
        if line.contains(".entry ") || line.contains(".func ") {
            if let Some(head) = line.rfind('(').map(|end| &line[..end]) {
                function = head
                    .split_whitespace()
                    .last()
                    .unwrap_or_default()
                    .to_string();
            }
            unrounded_products.clear();
            continue;
        }
        let mut tokens = line
            .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
            .filter(|token| !token.is_empty());
        let Some(mut opcode) = tokens.next() else {
            continue;
        };
        if opcode.starts_with('@') {
            // A guard predicate precedes the opcode.
            let Some(guarded) = tokens.next() else {
                continue;
            };
            opcode = guarded;
        }
        let parts: Vec<&str> = opcode.split('.').collect();
        if is_approximate(&parts) {
            findings.push((function.clone(), line.to_string()));
            continue;
        }
        if !is_unrounded_f32(&parts) {
            continue;
        }
        let Some(destination) = tokens.next() else {
            continue;
        };
        match parts.first() {
            Some(&"mul") => unrounded_products.push(destination.to_string()),
            Some(&("add" | "sub"))
                if tokens.any(|operand| unrounded_products.iter().any(|p| p == operand)) =>
            {
                findings.push((function.clone(), line.to_string()));
            }
            _ => {}
        }
    }
    findings
}

/// `mul`, `add` or `sub` on `.f32` with no IEEE rounding modifier.
fn is_unrounded_f32(parts: &[&str]) -> bool {
    matches!(parts.first(), Some(&("mul" | "add" | "sub")))
        && parts.last() == Some(&"f32")
        && !parts
            .iter()
            .any(|part| matches!(*part, "rn" | "rz" | "rm" | "rp"))
}

/// `div.approx`, `div.full`, `rcp.approx` or `sqrt.approx`.
fn is_approximate(parts: &[&str]) -> bool {
    matches!(
        parts,
        ["div", "approx" | "full", ..] | ["rcp" | "sqrt", "approx", ..]
    )
}

#[cfg(any(
    all(feature = "cuda-oxide-j2k-encode", j2k_cuda_oxide_j2k_encode_built),
    all(feature = "cuda-oxide-htj2k-encode", j2k_cuda_oxide_htj2k_encode_built)
))]
fn ptx_source(ptx: &[u8]) -> &str {
    let source = ptx.strip_suffix(&[0]).unwrap_or(ptx);
    std::str::from_utf8(source).expect("PTX UTF-8")
}

#[test]
fn rounding_hazard_scan_flags_contractible_pairs_and_approximations() {
    let ptx = "\
.visible .entry contracted(
{
\tmul.f32 \t%r3, %r1, %r2;
\tadd.f32 \t%r4, %r3, %r0;
\tmul.rn.f32 \t%r5, %r1, %r2;
\tadd.f32 \t%r6, %r5, %r0;
\tdiv.approx.f32 \t%r7, %r1, %r2;
\tdiv.rn.f32 \t%r8, %r1, %r2;
\t@%p1 sub.f32 \t%r9, %r0, %r3;
}
.visible .func  (.param .b32 func_retval0) clean(
{
\tadd.f32 \t%r4, %r3, %r0;
\tfma.rn.f32 \t%r9, %r1, %r2, %r0;
\tsub.rn.f32 \t%r10, %r9, %r3;
}
";
    let findings = rounding_hazards(ptx);
    assert_eq!(
        findings,
        [
            (
                "contracted".to_string(),
                "add.f32 \t%r4, %r3, %r0;".to_string()
            ),
            (
                "contracted".to_string(),
                "div.approx.f32 \t%r7, %r1, %r2;".to_string()
            ),
            (
                "contracted".to_string(),
                "@%p1 sub.f32 \t%r9, %r0, %r3;".to_string()
            ),
        ]
    );
}

#[cfg(all(feature = "cuda-oxide-j2k-encode", j2k_cuda_oxide_j2k_encode_built))]
#[test]
fn j2k_encode_ptx_rounds_every_float_operation_explicitly() {
    let findings = rounding_hazards(ptx_source(super::j2k_encode_ptx()));
    assert!(
        findings.is_empty(),
        "cuda_oxide_j2k_encode.ptx has contractible or approximate float operations: {findings:#?}"
    );
}

#[cfg(all(feature = "cuda-oxide-htj2k-encode", j2k_cuda_oxide_htj2k_encode_built))]
#[test]
fn htj2k_encode_ptx_rounds_every_float_operation_explicitly() {
    let findings = rounding_hazards(ptx_source(super::htj2k_encode_ptx()));
    assert!(
        findings.is_empty(),
        "cuda_oxide_htj2k_encode.ptx has contractible or approximate float operations: {findings:#?}"
    );
}
