pub(crate) fn markdown_header(out: &mut String, columns: &[&str]) {
    markdown_row(out, columns.iter().copied());
    markdown_row(out, columns.iter().map(|_| "---"));
}

pub(crate) fn markdown_row<I, S>(out: &mut String, values: I)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    out.push('|');
    for value in values {
        out.push(' ');
        out.push_str(&escape_table_cell(value.as_ref()));
        out.push_str(" |");
    }
    out.push('\n');
}

pub(crate) fn escape_table_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

pub(crate) fn escape_inline_code(value: &str) -> String {
    value.replace('`', "'").replace('\n', " ")
}

#[cfg(test)]
mod tests {
    use super::{escape_inline_code, escape_table_cell, markdown_header, markdown_row};

    #[test]
    fn table_cells_escape_pipes_and_newlines() {
        assert_eq!(escape_table_cell("a|b\nc"), "a\\|b c");
    }

    #[test]
    fn inline_code_escapes_backticks_and_newlines() {
        assert_eq!(escape_inline_code("a`b\nc"), "a'b c");
    }

    #[test]
    fn markdown_rows_and_headers_are_deterministic() {
        let mut out = String::new();
        markdown_header(&mut out, &["A|B", "C"]);
        markdown_row(&mut out, ["x", "y\nz"]);

        assert_eq!(out, "| A\\|B | C |\n| --- | --- |\n| x | y z |\n");
    }
}
