# Repository lint contract

`cargo xtask repo-lint` enforces repository properties that Rust compilation,
behavior tests, stable-API capture, or a standard external tool cannot express
directly. It is not a second compiler and does not prescribe where an internal
symbol or test must live.

## Accepted policy checks

Repository policy is appropriate for boundaries such as:

- release-manifest completeness and dependency ordering;
- stable and implementation API inventory agreement;
- unsafe-code inventory and reviewed suppression scope;
- workflow permissions, immutable action references, secret isolation, and
  required-gate wiring;
- dependency, corpus-license, generated-source, and public-document contracts;
- accelerator validation that must fail closed when required hardware is
  absent.

Checks should parse the relevant format when a parser exists. Source scanning
is reserved for narrow invariants that do not have a compiler or runtime
representation, and must fail closed on malformed input.

## Rejected policy checks

Do not add repository-lint assertions whose only purpose is to freeze:

- physical file or module ownership;
- source line counts;
- exact test counts or test function names;
- private helper names or textual call shapes;
- formatting already owned by `rustfmt` or lints already owned by Clippy.

Those checks make ordinary refactoring change two implementations without
proving behavior. File size remains a review signal, not a build failure.
Behavior belongs in the owning crate's unit, integration, property,
differential, or hardware tests. Public exposure belongs in the stable-API
inventory. Cheap type invariants should be compiler checked.

## Allocation evidence

Allocation ledgers and `try_reserve` behavior tests remain authoritative for
exact error categories, transactional failure, retained capacity, and
aggregate live-byte formulas.

The unpublished `j2k-alloc-probe` crate supplements those tests at selected
real codec boundaries. It records successful allocation/reallocation calls
and gross requested bytes. Deallocations are diagnostic only and never credit
a budget. It intentionally does not claim retained bytes or peak live bytes.
Measurements are process-global, serial, and must join all worker work before
returning.

Textual allocation policies may be retired only after their unique invariant
is covered by compiler enforcement, direct behavior/property tests, or a
measured boundary. Adding the probe does not by itself justify deleting the
remaining allocation policies.

## Retirement rule

Before removing a policy check:

1. identify every property it claims to protect;
2. map each material property to direct replacement evidence;
3. run the owning crate's behavior suite before deletion;
4. run repository lint, formatting, and Clippy after deletion;
5. keep the old check when the replacement cannot run on the required target.

Historical reports and reviewed API evidence remain provenance records even
when the command surface that produced them is simplified.
