is_male = True
if is_male:
    print("You are a male.")
else:
    print("You are not a male.")

### If Statements & Comparisons

def max_num(num1, num2, num3):
    """

    :param num1: param num2:
    :param num3: param num2:
    :param num2: 

    """
    if num1 >= num2 and num1 >= num3:
        return num1
    elif num2 >= num1 and num2 >= num3:
        return num2
    else:
        return num3

print(max_num(3, 4, 5))

### Building a Better Calculator

num1 = float(input("Enter first number: "))
op = input("Enter operator: ")
num2 = float(input("Enter second number: "))

if op == "+":
    print(num1 + num2)
elif op == "-":
    print(num1 - num2)
elif op == "*":
    print(num1 * num2)
elif op == "/":
    print(num1 / num2)
else:
    print("Invalid operator")


