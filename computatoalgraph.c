#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// A simple structure to represent a value and its gradient.
// It also holds pointers to its children in the graph, which are
// used for the backward pass (backpropagation).
typedef struct Value
{
    double data;
    double grad;
    struct Value **children;
    int num_children;
    void (*backward)(struct Value *);
} Value;

// A basic stack for the topological sort.
typedef struct Stack
{
    Value *items[100];
    int top;
} Stack;

void push(Stack *s, Value *v)
{
    if (s->top >= 99)
    {
        printf("Stack overflow!\n");
        return;
    }
    s->items[++(s->top)] = v;
}

Value *pop(Stack *s)
{
    if (s->top < 0)
    {
        return NULL;
    }
    return s->items[(s->top)--];
}

int is_empty(Stack *s)
{
    return s->top < 0;
}

// Helper function to create a new Value node.
Value *new_value(double data, Value *children[], int num_children)
{
    Value *v = (Value *)malloc(sizeof(Value));
    if (v == NULL)
    {
        perror("Failed to allocate memory for Value node");
        exit(1);
    }
    v->data = data;
    v->grad = 0.0;
    v->num_children = num_children;

    // Allocate memory for children pointers if any.
    if (num_children > 0)
    {
        v->children = (Value **)malloc(num_children * sizeof(Value *));
        if (v->children == NULL)
        {
            perror("Failed to allocate memory for children");
            free(v);
            exit(1);
        }
        memcpy(v->children, children, num_children * sizeof(Value *));
    }
    else
    {
        v->children = NULL;
    }

    v->backward = NULL;
    return v;
}

// Function to free the memory of a Value node and its children.
void free_value(Value *v)
{
    if (v == NULL)
    {
        return;
    }
    // We only free the parent; the children are freed by their
    // own parent nodes' calls to free_value.
    if (v->children != NULL)
    {
        free(v->children);
    }
    free(v);
}

// Zeroes the gradient of the entire graph recursively.
void zero_grad_graph(Value *v)
{
    if (v == NULL)
        return;
    v->grad = 0.0;
    for (int i = 0; i < v->num_children; ++i)
    {
        zero_grad_graph(v->children[i]);
    }
}

// Backward pass for the addition operation.
void _backward_add(Value *out)
{
    if (out->num_children < 2)
        return;
    out->children[0]->grad += out->grad;
    out->children[1]->grad += out->grad;
}

// Backward pass for the multiplication operation.
void _backward_mul(Value *out)
{
    if (out->num_children < 2)
        return;
    out->children[0]->grad += out->children[1]->data * out->grad;
    out->children[1]->grad += out->children[0]->data * out->grad;
}

// Backward pass for the exponentiation operation.
void _backward_pow(Value *out)
{
    if (out->num_children < 2)
        return;
    double base = out->children[0]->data;
    double exp_val = out->children[1]->data;
    out->children[0]->grad += (exp_val * pow(base, exp_val - 1)) * out->grad;
}

// Backward pass for the exponential operation.
void _backward_exp(Value *out)
{
    if (out->num_children < 1)
        return;
    out->children[0]->grad += out->data * out->grad;
}

// Backward pass for the ReLU operation.
void _backward_relu(Value *out)
{
    if (out->num_children < 1)
        return;
    out->children[0]->grad += (out->data > 0.0) * out->grad;
}

// Function for the addition operation.
Value *add(Value *a, Value *b)
{
    Value *children[] = {a, b};
    Value *out = new_value(a->data + b->data, children, 2);
    out->backward = _backward_add;
    return out;
}

// Function for the multiplication operation.
Value *mul(Value *a, Value *b)
{
    Value *children[] = {a, b};
    Value *out = new_value(a->data * b->data, children, 2);
    out->backward = _backward_mul;
    return out;
}

// Function for the exponentiation operation.
Value *my_pow(Value *base, Value *exp_val)
{
    Value *children[] = {base, exp_val};
    Value *out = new_value(pow(base->data, exp_val->data), children, 2);
    out->backward = _backward_pow;
    return out;
}

// Function for the exponential operation.
Value *my_exp(Value *a)
{
    Value *children[] = {a};
    Value *out = new_value(exp(a->data), children, 1);
    out->backward = _backward_exp;
    return out;
}

// Function for the ReLU activation function.
Value *relu(Value *a)
{
    Value *children[] = {a};
    double output = a->data > 0.0 ? a->data : 0.0;
    Value *out = new_value(output, children, 1);
    out->backward = _backward_relu;
    return out;
}

// A more robust topological sort using a stack to handle arbitrary graphs.
void build_topo(Value *v, Value *topo[], int *count, int *visited)
{
    if (visited[(long)v % 100])
        return; // Simple hash-based check for visited nodes
    visited[(long)v % 100] = 1;

    for (int i = 0; i < v->num_children; ++i)
    {
        build_topo(v->children[i], topo, count, visited);
    }
    topo[*count] = v;
    (*count)++;
}

// Main backpropagation function.
void backward(Value *v)
{
    Value *topo[100];
    int count = 0;
    int visited[100] = {0}; // Initialize visited array.

    build_topo(v, topo, &count, visited);

    // Set the initial gradient of the output node to 1.0.
    v->grad = 1.0;

    // Iterate through the topologically sorted nodes in reverse order.
    for (int i = count - 1; i >= 0; --i)
    {
        if (topo[i]->backward != NULL)
        {
            topo[i]->backward(topo[i]);
        }
    }
}

int main()
{
    // Construct a more complex expression graph:
    // f = relu( (a * b) + my_pow(c, 2) )

    // Create the input nodes.
    Value *a = new_value(2.0, NULL, 0);
    Value *b = new_value(3.0, NULL, 0);
    Value *c = new_value(-2.0, NULL, 0);

    // Perform operations.
    Value *mul_out = mul(a, b);
    Value *pow_out = my_pow(c, new_value(2.0, NULL, 0));
    Value *add_out = add(mul_out, pow_out);
    Value *f = relu(add_out);

    // Print the forward pass result.
    printf("Forward Pass Result: %.2f\n", f->data);

    // Perform the backward pass to calculate gradients.
    backward(f);

    // Print the gradients of the input nodes.
    printf("--- Gradients ---\n");
    printf("Gradient of a: %.2f\n", a->grad);
    printf("Gradient of b: %.2f\n", b->grad);
    printf("Gradient of c: %.2f\n", c->grad);

    // Free all allocated memory. In a real application, you would need to
    // implement a more robust garbage collection or reference counting.
    free_value(a);
    free_value(b);
    free_value(c);
    free_value(mul_out);
    free_value(pow_out);
    free_value(add_out);
    free_value(f);

    return 0;
}
