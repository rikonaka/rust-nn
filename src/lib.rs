#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

pub struct Node {
    id: u32,
    operation: u8,
    grad: f32,
}

pub impl Node {
    fn init(&self) {
        self.id = global_id;
    }
}