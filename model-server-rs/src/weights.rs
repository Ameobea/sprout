use rmpv::Value;
use std::{fs, path::Path};

pub struct Mat {
    pub k: usize,
    pub n: usize,
    pub w: Vec<f32>,
}

pub struct Layer {
    pub w: Mat,
    pub b: Vec<f32>,
}

pub struct Params {
    pub enc1: Layer,     // 12000 x 2048
    pub bott: Layer,     // 2048 x 512
    pub item_up1: Layer, // 512 x 1024
    pub item_up2: Layer, // 1024 x 2048
    pub item_out: Layer, // 2048 x 6000
    pub rat_up1: Layer,
    pub rat_up2: Layer,
    pub rat_out: Layer,
}

fn ext_to_array(v: &Value) -> (Vec<usize>, Vec<f32>) {
    let Value::Ext(1, payload) = v else { panic!("expected msgpack ext(1) ndarray, got {v:?}") };
    let inner = rmpv::decode::read_value(&mut &payload[..]).unwrap();
    let Value::Array(parts) = inner else { panic!("bad ndarray encoding") };
    let shape: Vec<usize> = parts[0]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(parts[1].as_str().unwrap(), "float32");
    let Value::Binary(bytes) = &parts[2] else { panic!("bad ndarray payload") };
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(data.len(), shape.iter().product::<usize>());
    (shape, data)
}

fn get<'a>(map: &'a Value, key: &str) -> &'a Value {
    let Value::Map(entries) = map else { panic!("expected map") };
    &entries
        .iter()
        .find(|(k, _)| k.as_str() == Some(key))
        .unwrap_or_else(|| panic!("missing key {key}"))
        .1
}

fn layer(root: &Value, name: &str, k: usize, n: usize) -> Layer {
    let node = get(root, name);
    let (kshape, kdata) = ext_to_array(get(node, "kernel"));
    let (bshape, bdata) = ext_to_array(get(node, "bias"));
    assert_eq!(kshape, [k, n], "{name} kernel shape");
    assert_eq!(bshape, [n], "{name} bias shape");
    Layer { w: Mat { k, n, w: kdata }, b: bdata }
}

impl Params {
    pub fn load(path: &Path) -> Params {
        let bytes = fs::read(path).unwrap();
        let root = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
        Params {
            enc1: layer(&root, "Dense_0", crate::IN_DIM, crate::HIDDEN),
            bott: layer(&root, "bottleneck", crate::HIDDEN, crate::BOTTLENECK),
            item_up1: layer(&root, "dec_item_up1", crate::BOTTLENECK, crate::DEC_MID),
            item_up2: layer(&root, "dec_item_up2", crate::DEC_MID, crate::HIDDEN),
            item_out: layer(&root, "item_logits", crate::HIDDEN, crate::CORPUS),
            rat_up1: layer(&root, "dec_rating_up1", crate::BOTTLENECK, crate::DEC_MID),
            rat_up2: layer(&root, "dec_rating_up2", crate::DEC_MID, crate::HIDDEN),
            rat_out: layer(&root, "rating_pred", crate::HIDDEN, crate::CORPUS),
        }
    }
}
