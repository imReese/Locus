use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PrometheusSample {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    pub value: f64,
}

pub(crate) fn parse_prometheus(
    input: &str,
    max_samples: usize,
) -> Result<Vec<PrometheusSample>, String> {
    let mut samples = Vec::new();
    for (index, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if samples.len() >= max_samples {
            return Err(format!(
                "Prometheus response exceeds the {max_samples}-sample limit"
            ));
        }
        let (metric, remainder) = split_metric_and_value(line)
            .ok_or_else(|| format!("invalid Prometheus sample on line {}", index + 1))?;
        let value_text = remainder
            .split_ascii_whitespace()
            .next()
            .ok_or_else(|| format!("missing Prometheus value on line {}", index + 1))?;
        let value = value_text
            .parse::<f64>()
            .map_err(|_| format!("invalid Prometheus value on line {}", index + 1))?;
        let (name, labels) =
            parse_metric(metric).map_err(|error| format!("{error} on line {}", index + 1))?;
        samples.push(PrometheusSample {
            name,
            labels,
            value,
        });
    }
    Ok(samples)
}

fn split_metric_and_value(line: &str) -> Option<(&str, &str)> {
    let mut quoted = false;
    let mut escaped = false;
    let mut braces = 0_u32;
    for (index, character) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if quoted && character == '\\' {
            escaped = true;
            continue;
        }
        if character == '"' {
            quoted = !quoted;
            continue;
        }
        if !quoted {
            match character {
                '{' => braces = braces.saturating_add(1),
                '}' => braces = braces.saturating_sub(1),
                _ if character.is_ascii_whitespace() && braces == 0 => {
                    let remainder = line[index..].trim_start();
                    return Some((&line[..index], remainder));
                }
                _ => {}
            }
        }
    }
    None
}

fn parse_metric(metric: &str) -> Result<(String, BTreeMap<String, String>), String> {
    let Some(open) = metric.find('{') else {
        validate_metric_name(metric)?;
        return Ok((metric.to_owned(), BTreeMap::new()));
    };
    if !metric.ends_with('}') {
        return Err("unterminated Prometheus label set".to_owned());
    }
    let name = &metric[..open];
    validate_metric_name(name)?;
    let labels = parse_labels(&metric[open + 1..metric.len() - 1])?;
    Ok((name.to_owned(), labels))
}

fn validate_metric_name(name: &str) -> Result<(), String> {
    let mut characters = name.chars();
    let Some(first) = characters.next() else {
        return Err("empty Prometheus metric name".to_owned());
    };
    if !(first.is_ascii_alphabetic() || matches!(first, '_' | ':'))
        || characters
            .any(|character| !(character.is_ascii_alphanumeric() || matches!(character, '_' | ':')))
    {
        return Err(format!("invalid Prometheus metric name {name:?}"));
    }
    Ok(())
}

fn parse_labels(input: &str) -> Result<BTreeMap<String, String>, String> {
    let bytes = input.as_bytes();
    let mut labels = BTreeMap::new();
    let mut cursor = 0_usize;
    while cursor < bytes.len() {
        while cursor < bytes.len() && (bytes[cursor].is_ascii_whitespace() || bytes[cursor] == b',')
        {
            cursor += 1;
        }
        if cursor == bytes.len() {
            break;
        }
        let key_start = cursor;
        while cursor < bytes.len()
            && (bytes[cursor].is_ascii_alphanumeric() || bytes[cursor] == b'_')
        {
            cursor += 1;
        }
        if key_start == cursor {
            return Err("invalid Prometheus label name".to_owned());
        }
        let key = &input[key_start..cursor];
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if bytes.get(cursor) != Some(&b'=') {
            return Err(format!("Prometheus label {key:?} omitted '='"));
        }
        cursor += 1;
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if bytes.get(cursor) != Some(&b'"') {
            return Err(format!("Prometheus label {key:?} is not quoted"));
        }
        cursor += 1;
        let mut value = String::new();
        let mut closed = false;
        while cursor < bytes.len() {
            match bytes[cursor] {
                b'"' => {
                    cursor += 1;
                    closed = true;
                    break;
                }
                b'\\' => {
                    cursor += 1;
                    let escaped = *bytes
                        .get(cursor)
                        .ok_or_else(|| "unterminated Prometheus label escape".to_owned())?;
                    value.push(match escaped {
                        b'n' => '\n',
                        b'\\' => '\\',
                        b'"' => '"',
                        other => char::from(other),
                    });
                    cursor += 1;
                }
                byte if byte.is_ascii() => {
                    value.push(char::from(byte));
                    cursor += 1;
                }
                _ => return Err("Prometheus labels must be UTF-8 ASCII escapes".to_owned()),
            }
        }
        if !closed {
            return Err(format!("unterminated Prometheus label {key:?}"));
        }
        if labels.insert(key.to_owned(), value).is_some() {
            return Err(format!("duplicate Prometheus label {key:?}"));
        }
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor < bytes.len() && bytes[cursor] != b',' {
            return Err("Prometheus labels must be comma separated".to_owned());
        }
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_labels_escapes_and_optional_timestamps() {
        let samples = parse_prometheus(
            "# TYPE queue gauge\nmetric:queue{model_name=\"a\\\"b\",rank=\"0\"} 2 123\nplain 4.5\n",
            10,
        )
        .expect("metrics");
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].name, "metric:queue");
        assert_eq!(samples[0].labels["model_name"], "a\"b");
        assert_eq!(samples[0].value, 2.0);
        assert_eq!(samples[1].value, 4.5);
    }

    #[test]
    fn rejects_malformed_or_unbounded_exposition() {
        assert!(parse_prometheus("metric{label=bad} 1", 10).is_err());
        assert!(parse_prometheus("one 1\ntwo 2", 1).is_err());
    }

    #[test]
    fn preserves_non_finite_values_for_fail_closed_consumers() {
        let samples = parse_prometheus("metric NaN", 1).expect("syntax is valid");
        assert!(samples[0].value.is_nan());
    }
}
