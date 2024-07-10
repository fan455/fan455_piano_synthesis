use std::{collections::{BTreeMap, HashMap}, str::FromStr};
use std::hash::Hash;
use std::fmt::Debug;


#[inline]
pub fn convert_hashmap_keys<K, V>( x: &HashMap<String, V> )
-> HashMap<K, V> where K: Eq+Hash+FromStr, <K as FromStr>::Err: Debug, V: Clone {
    let mut y = HashMap::<K, V>::with_capacity(x.len());
    for (key, val) in x.iter() {
        y.insert(key.parse().unwrap(), val.clone());
    }
    y
}

#[inline]
pub fn convert_btreemap_keys<K, V>( x: &BTreeMap<String, V> )
-> BTreeMap<K, V> where K: FromStr+Ord, <K as FromStr>::Err: Debug, V: Clone {
    let mut y = BTreeMap::<K, V>::new();
    for (key, val) in x.iter() {
        y.insert(key.parse().unwrap(), val.clone());
    }
    y
}