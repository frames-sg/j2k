// SPDX-License-Identifier: MIT OR Apache-2.0

/// Return Metal kernel entry-point names declared as `kernel void`.
pub fn metal_kernel_names(source: &str) -> Vec<String> {
    source.lines().filter_map(metal_kernel_name).collect()
}

/// Return true when `host_source` compiles a named Metal kernel into a pipeline.
pub fn host_compiles_metal_pipeline(host_source: &str, kernel_name: &str) -> bool {
    let quoted = format!("\"{kernel_name}\"");
    host_source.match_indices(&quoted).any(|(index, _)| {
        let context = &host_source[index.saturating_sub(96)..index];
        context.contains("get_function(") || context.contains("pipeline(")
    })
}

/// Return Metal kernels declared by shader sources but not compiled by host setup.
pub fn unwired_metal_kernels<'a>(
    shader_sources: impl IntoIterator<Item = &'a str>,
    host_source: &str,
) -> Vec<String> {
    let mut names = Vec::new();
    for source in shader_sources {
        for name in metal_kernel_names(source) {
            if names.iter().all(|existing| existing != &name) {
                names.push(name);
            }
        }
    }

    names
        .into_iter()
        .filter(|name| !host_compiles_metal_pipeline(host_source, name))
        .collect()
}

fn metal_kernel_name(line: &str) -> Option<String> {
    let rest = line.trim_start().strip_prefix("kernel void ")?;
    rest.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .next()
        .map(ToOwned::to_owned)
}
