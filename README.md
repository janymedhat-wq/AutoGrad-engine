# Computational Graph in C (a baby autograd engine)

So I decided to see what happens if you strip PyTorch down to the bare metal…  
and ended up writing a **tiny automatic differentiation engine in pure C**.  

Every number in this system isn’t just a `double` — it’s a `Value`:  
- it knows its data,  
- it remembers who created it (its parents in the graph),  
- and it carries a little `backward` function so it can whisper gradients upstream when backprop happens.  

Then I wired up some classic ops:  
- `add` (+),  
- `mul` (×),  
- `pow` (^),  
- `exp`,  
- `relu` (because what’s a neural net without a ReLU?).  

When you build expressions, you’re secretly building a **graph**.  
And when you call `backward(f)`, the code walks that graph in reverse (topological sort style),  
kicking gradients back through each node like tiny math dominoes falling over.  

### Example

```c
// f = relu((a * b) + (c^2))
Value *a = new_value(2.0, NULL, 0);
Value *b = new_value(3.0, NULL, 0);
Value *c = new_value(-2.0, NULL, 0);

Value *f = relu(add(mul(a, b), my_pow(c, new_value(2.0, NULL, 0))));
backward(f);
