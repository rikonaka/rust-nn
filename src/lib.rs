use std::{fmt, ops::Mul};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

pub struct Node {
    id: u32,
    operation: Operation,
    grad: f32,
    inputs: Vec<Node>,
    value: f32,
}

impl Node {
    fn init(id: u32, operation: Operation, inputs: Vec<Node>) -> Node {
        let new_node = Node {
            id: id,
            operation: operation,
            grad: 0.0,
            inputs: inputs,
            value: 0.0,
        };
        println!("eager exec: {}", &new_node);
        new_node
    }
    /// 将输入统一转换成数值，因为具体的计算只能发生在数值上。
    fn input2values(&self) -> Vec<f32> {
        let mut new_inputs: Vec<f32> = Vec::new();
        for i in &self.inputs {
            new_inputs.push(i.value)
        }
        new_inputs
    }
    fn evaluate(&mut self) {
        let values = self.input2values();
        self.value = match self.operation.compute(values) {
            Ok(o) => o,
            Err(_) => 0.0,
        }
    }
}

// To use the `{}` marker, the trait `fmt::Display` must be implemented
// manually for the type.
impl fmt::Display for Node {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Write strictly the first element into the supplied output
        // stream: `f`. Returns `fmt::Result` which indicates whether the
        // operation succeeded or failed. Note that `write!` uses syntax which
        // is very similar to `println!`.
        // write!(f, "{}", self.0)
        write!(
            f,
            "Node{}: {:?} {} = {}, grad: {:.3}",
            self.id,
            self.input2values(),
            self.operation.name(),
            self.value,
            self.grad
        )
    }
}

/// 所有操作的基类。注意Op本身不包含状态，计算的状态保存在Node中，每次调用Op都会产生一个Node。
enum Operation {
    AddOp,
    SubOp,
}

impl Operation {
    fn name(&self) -> String {
        match self {
            Operation::AddOp => AddOp::name(),
            Operation::SubOp => SubOp::name(),
            _ => AddOp::name(),
        }
    }
    fn compute(&self, values: Vec<f32>) -> Result<f32, ()> {
        match self {
            Operation::AddOp => {
                let ret = match AddOp::compute(values) {
                    Ok(o) => o,
                    Err(_) => 0.0,
                };
                Ok(ret)
            }
            Operation::SubOp => {
                let ret = match SubOp::compute(values) {
                    Ok(o) => o,
                    Err(_) => 0.0,
                };
                Ok(ret)
            }
            _ => Err(()),
        }
    }
    fn gradient<E>(values: Vec<f32>, output_grad: f32) -> Result<Vec<f32>, E> {
        let ret = vec![output_grad, output_grad];
        Ok(ret) // gradient of a and b
    }
    }
}

pub struct AddOp {}
impl AddOp {
    fn name() -> String {
        "add".to_string()
    }
    fn call(id: u32, a: Node, b: Node) -> Node {
        let inputs = vec![a, b];
        let new_node = Node::init(id, Operation::AddOp, inputs);
        new_node
    }
    fn compute(values: Vec<f32>) -> Result<f32, ()> {
        if values.len() == 2 {
            let sum = values[0] + values[1];
            return Ok(sum);
        }
        Err(())
    }
    fn gradient<E>(values: Vec<f32>, output_grad: f32) -> Result<Vec<f32>, E> {
        let ret = vec![output_grad, output_grad];
        Ok(ret) // gradient of a and b
    }
}

pub struct SubOp {}
impl SubOp {
    fn name() -> String {
        "sub".to_string()
    }
    fn call(id: u32, a: Node, b: Node) -> Node {
        let inputs = vec![a, b];
        let new_node = Node::init(id, Operation::SubOp, inputs);
        new_node
    }
    fn compute(values: Vec<f32>) -> Result<f32, ()> {
        if values.len() == 2 {
            let sub = values[0] - values[1];
            return Ok(sub);
        }
        Err(())
    }
    fn gradient<E>(values: Vec<f32>, output_grad: f32) -> Result<Vec<f32>, E> {
        let ret = vec![output_grad, -1.0 * output_grad];
        Ok(ret) // gradient of a and b
    }
}
