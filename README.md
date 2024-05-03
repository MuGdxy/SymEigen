# SymEigen

## Introduction

This is a simple Sympy extension to generate Eigen C++ code from the given expression. 
I concentrate on the Vector/Vector derivative, which is the most common operation in the optimization problem.

To use it, you need to install the `sympy` library. You can install it using pip:

```bash
pip install sympy
```

## Install

This library is so simple that you can just copy the `SymEigen.py` file to your project directory.

## Usage

The [QuickStart](./QuickStart.ipynb) can be run using Jupyter Notebook [Highly Recommended]. 
You can use VSCode with Jupyter Notebook extension to run the code, or any other way you like.

The code is also available in the [SymEigen.py](./SymEigen.py) source file. You can just run it in your Python environment.

## Quick View

I paste the code here for your quick view:

Say we are calculating the Energy of a spring. The equation is as follows:
$$
E = \frac{1}{2} k (|\mathbf{x}-\mathbf{y}| - L_0)^2
$$

To compactly write the equation, we can define the following variables, fully 6 Dof:
$$
\mathbf{X} = 
\begin{bmatrix}
\mathbf{x} \\
\mathbf{y} 
\end{bmatrix}
$$
```python
    from SymEigen import *
    # Then we can define such a matrix as follows:
    X = Eigen.Vector('X', 6)
    # Other coefficients are defined as follows:
    k = Eigen.Scalar('k')
    L0 = Eigen.Scalar('L0')
    # It's easy to calculate the Energy as follows:
    X_l = Matrix(X[0:3])
    X_r = Matrix(X[3:6])
    d = X_l - X_r 
    E = k * (sqrt(d.T * d) - L0) / 2
    # We use `VecDiff` to calculate the Vec/Vec derivative, so the Gradient of the Energy is:
    G = VecDiff(E, X)
    # So for the Hessian, we have:
    H = VecDiff(G, X)
    # To generate Eigen Cpp code, we should do the following:
    # 1. Declare a `EigenFunctionGenerator` as a context.
    # 2. Wrap the Input Variable to a `Closure`.
    # 3. Call the `Closure` by inputting the function name and `Expr` (e.g. the `E`, `G`, `H`).
    Gen = EigenFunctionGenerator()
    Closure = Gen.Closure(k, L0, X)
    # First, we generate the Eigen Cpp code for the Energy:
    print(Closure('SpringEnergy', E, 'E'))
    # Then, Gradient:
    print(Closure('SpringGradient', G, 'G'))
    # Finally, Hessian:
    print(Closure('SpringHessian', H, 'H'))
```


And I just paste some output here, because the output is too long:

```cpp
template <typename T>
void SpringGradient(Eigen::Vector<T,6>& G, const T& k, const T& L0, const Eigen::Vector<T,6>& X)
{
/*******************************************************************
Function generated by SymEigen.py
Author: MuGdxy
GitHub: https://github.com/MuGdxy
E-Mail: lxy819469559@gmail.com
********************************************************************
LaTeX expression:
//tex:$$G = \left[\begin{matrix}\frac{k \left(X(0) - X(3)\right)}{2 \sqrt{\left(X(0) - X(3)\right)^{2} + \left(X(1) - X(4)\right)^{2} + \left(X(2) - X(5)\right)^{2}}}\\\frac{k \left(X(1) - X(4)\right)}{2 \sqrt{\left(X(0) - X(3)\right)^{2} + \left(X(1) - X(4)\right)^{2} + \left(X(2) - X(5)\right)^{2}}}\\\frac{k \left(X(2) - X(5)\right)}{2 \sqrt{\left(X(0) - X(3)\right)^{2} + \left(X(1) - X(4)\right)^{2} + \left(X(2) - X(5)\right)^{2}}}\\\frac{k \left(- X(0) + X(3)\right)}{2 \sqrt{\left(X(0) - X(3)\right)^{2} + \left(X(1) - X(4)\right)^{2} + \left(X(2) - X(5)\right)^{2}}}\\\frac{k \left(- X(1) + X(4)\right)}{2 \sqrt{\left(X(0) - X(3)\right)^{2} + \left(X(1) - X(4)\right)^{2} + \left(X(2) - X(5)\right)^{2}}}\\\frac{k \left(- X(2) + X(5)\right)}{2 \sqrt{\left(X(0) - X(3)\right)^{2} + \left(X(1) - X(4)\right)^{2} + \left(X(2) - X(5)\right)^{2}}}\end{matrix}\right]$$

Symbol Name Mapping:
k:
    -> {}
    -> Matrix([[k]])
Symbol Name Mapping:
L0:
    -> {}
    -> Matrix([[L0]])
Symbol Name Mapping:
X:
    -> {}
    -> Matrix([[X(0)], [X(1)], [X(2)], [X(3)], [X(4)], [X(5)]])
*******************************************************************/
/* Sub Exprs */
auto x0 = X(0) - X(3);
auto x1 = X(1) - X(4);
auto x2 = X(2) - X(5);
auto x3 = (1.0/2.0)*k/sqrt((x0 * x0) + (x1 * x1) + (x2 * x2));
/* Simplified Expr */
G(0) = x0*x3;
G(1) = x1*x3;
G(2) = x2*x3;
G(3) = -x0*x3;
G(4) = -x1*x3;
G(5) = -x2*x3;
}
```

## Further

`SymEigen` concentrates on the Vector/Vector derivative. If you want to differentiate between Matrix/Matrix, Matrix/Vectors or other operations, the recommended way is to use the `Vectorize` function to convert the Matrix to a Vector, then use the `VecDiff` function to calculate the derivative.

For example, in Continuum Mechanics, we have the Deformation Gradient `F`, which is a 3 by 3 matrix, we can convert it to a 9 by 1 vector, then use the `VecDiff` function to calculate the derivative.

```python
F = Eigen.Matrix('F', 3, 3)
VecF = F.Vectorize('VecF')
C = F.T * F - eye(3)
VecC = C.Vectorize('VecC')
G = VecDiff(VecC, VecF)
```

The output will be:
$$
\left[\begin{matrix}2 F(0,0) & 2 F(1,0) & 2 F(2,0) & 0 & 0 & 0 & 0 & 0 & 0\\F(0,1) & F(1,1) & F(2,1) & F(0,0) & F(1,0) & F(2,0) & 0 & 0 & 0\\F(0,2) & F(1,2) & F(2,2) & 0 & 0 & 0 & F(0,0) & F(1,0) & F(2,0)\\F(0,1) & F(1,1) & F(2,1) & F(0,0) & F(1,0) & F(2,0) & 0 & 0 & 0\\0 & 0 & 0 & 2 F(0,1) & 2 F(1,1) & 2 F(2,1) & 0 & 0 & 0\\0 & 0 & 0 & F(0,2) & F(1,2) & F(2,2) & F(0,1) & F(1,1) & F(2,1)\\F(0,2) & F(1,2) & F(2,2) & 0 & 0 & 0 & F(0,0) & F(1,0) & F(2,0)\\0 & 0 & 0 & F(0,2) & F(1,2) & F(2,2) & F(0,1) & F(1,1) & F(2,1)\\0 & 0 & 0 & 0 & 0 & 0 & 2 F(0,2) & 2 F(1,2) & 2 F(2,2)\end{matrix}\right]
$$

SymEigen does the Element Name Mapping for you, which means, SymEigen doesn't care about the layout of the elements, the only thing it cares about is the unique element name. All differentiations are performed element-wise. The layout only affects the generated code. And you can freely substitute the `VecF` as `F`, when you generate the Eigen C++ code.

```python
Gen = EigenFunctionGenerator()
Closure = Gen.Closure(VecF) # Take VecF as Input
print(Closure('dVecCdVecF', dVecCdVecF))

Closure = Gen.Closure(F) # Take F as Input
print(Closure('dVecCdF', dVecCdVecF))
```
The output function declarations:
```cpp
template <typename T>
void dVecCdVecF(Eigen::Matrix<T,9,9>& R, const Eigen::Vector<T,9>& VecF);
template <typename T>
void dVecCdF(Eigen::Matrix<T,9,9>& R, const Eigen::Matrix<T,3,3>& F);
```
The `R` is the same, but the input is different. You can use the `VecF` or `F` as you like. But I recommend using the `VecF` as the input, because it's clearer to understand and use the code. Because the `Vectorize` operation is well-defined.