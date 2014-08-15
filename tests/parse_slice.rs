
extern crate ndarray;

use ndarray::Si;

#[cfg(test)]
/// Parse python slice notation into `Si`,
/// including `a:b`, `a:b:c`, `::s`, `1:`
fn parse_slice_str(s: &str) -> Si {
    let mut sp = s.split(':');
    let fst = sp.next();
    let snd = sp.next();
    let step = sp.next();
    assert!(sp.next().is_none());
    assert!(fst.is_some() && snd.is_some());

    let a = match fst.unwrap().trim() {
        "" => 0i,
        s => from_str::<int>(s).unwrap(),
    };
    let b = match snd.unwrap().trim() {
        "" => None,
        s => Some(from_str::<int>(s).unwrap()),
    };
    let c = match step.map(|x| x.trim()) {
        None | Some("") => 1,
        Some(s) => from_str::<int>(s).unwrap(),
    };
    Si(a, b, c)
}


#[test]
fn test_parse()
{
    let slice_strings = ["1:2:3", "::", "1:", "::-1", "::2"];
    for s in slice_strings.iter() {
        println!("Parse {} \t=> {}", *s, parse_slice_str(*s));
    }
}

