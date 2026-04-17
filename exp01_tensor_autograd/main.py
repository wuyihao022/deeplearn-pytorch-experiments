import torch


def print_section(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def tensor_basics() -> None:
    print_section("1. Tensor Basics")

    vector = torch.tensor([1.0, 2.0, 3.0])
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    random_tensor = torch.rand(4, 3)

    print("vector:\n", vector)
    print("vector shape:", vector.shape)
    print("vector dtype:", vector.dtype)

    print("\nmatrix:\n", matrix)
    print("matrix shape:", matrix.shape)

    print("\nrandom tensor:\n", random_tensor)
    print("random tensor shape:", random_tensor.shape)


def reshape_example() -> None:
    print_section("2. Reshape")

    x = torch.arange(1, 7)
    print("original x:\n", x)
    print("original shape:", x.shape)

    x_reshaped = x.reshape(2, 3)
    print("\nreshaped x:\n", x_reshaped)
    print("reshaped shape:", x_reshaped.shape)


def math_examples() -> None:
    print_section("3. Basic Math")

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    print("a + b =", a + b)
    print("a - b =", a - b)
    print("a * b =", a * b)
    print("a / b =", a / b)


def matrix_multiplication_example() -> None:
    print_section("4. Matrix Multiplication")

    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    print("x shape:", x.shape)
    print("w shape:", w.shape)

    y = x @ w
    print("x @ w:\n", y)
    print("result shape:", y.shape)


def autograd_single_value() -> None:
    print_section("5. Autograd: Single Value")

    x = torch.tensor(2.0, requires_grad=True)
    y = x**2 + 2 * x + 1

    print("x =", x.item())
    print("y = x^2 + 2x + 1 =", y.item())

    y.backward()

    print("dy/dx at x = 2:", x.grad.item())
    print("manual answer should be 2x + 2 = 6")


def autograd_vector_example() -> None:
    print_section("6. Autograd: Vector Example")

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x**2).sum()

    print("x =", x)
    print("y = sum(x^2) =", y.item())

    y.backward()

    print("gradient of y with respect to x:")
    print(x.grad)
    print("manual answer should be [2, 4, 6]")


def main() -> None:
    print("PyTorch version:", torch.__version__)
    tensor_basics()
    reshape_example()
    math_examples()
    matrix_multiplication_example()
    autograd_single_value()
    autograd_vector_example()

    print_section("7. Summary")
    print("You have finished Experiment 01.")
    print("Next step: explain each printed result in your own words.")


if __name__ == "__main__":
    main()
