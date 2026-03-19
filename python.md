# Getting Started

Alright let's start with python

install python

after installing python in macos you need to create a virtual environment for same

```sh
    python3 -m venv myenv
    source path/myenv/bin/activate.fish
    python3 -m pip install xyz  #to install packages
    python3 ai.py   #to interpret the file
```

### Variables & Data Types

In Python, variables are used to store data. You don’t need to declare a variable type — Python infers it automatically.

```python
name = "Nalin"
age = 20
height = 5.9
is_student = True
```

### Working With Strings

Strings are sequences of characters enclosed in quotes.

```python
greeting = "Hello"
name = "World"
print(greeting + " " + name)
print(greeting.lower())
print(greeting.upper())
print(len(greeting))
```

### Working With Numbers

Python supports integers, floats, and complex numbers.

```python
num = 5
print(num + 3)
print(num * 2)
print(num ** 2)  # exponentiation
```

### Getting Input From Users

```python
name = input("Enter your name: ")
print("Hello " + name)
```

### Building a Basic Calculator

```python
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))
print(num1 + num2)
```

### Mad Libs Game

```python
color = input("Enter a color: ")
plural_noun = input("Enter a plural noun: ")
celebrity = input("Enter a celebrity: ")

print("Roses are " + color)
print(plural_noun + " are blue")
print("I love " + celebrity)
```

### Lists

they are similar to vector in cpp i guess
they store the data of same type in them

```python
friends = ["Kevin", "Karen", "Jim"]
print(friends[0])
print(friends[-1])
friends[1] = "Mike"
```

### List Functions

```python
lucky_numbers = [4, 8, 15, 16, 23, 42]
friends = ["Kevin", "Karen", "Jim", "Oscar", "Toby"]
friends.extend(lucky_numbers)
friends.append("Creed") #puts at last
friends.insert(1, "Kelly")  #puts at specified index given
friends.remove("Jim")   #remove something from list
friends.pop()   #basically gets rid of last element of list
print(friends)

#sort the list
lucky_numbers.sort()
luckey_numbers.index(4) #returns 0
```

well you can slice a list via following: `[start:end+1:step]`

```python
x [1:4]     Items 1 to 3
X [1:6:2]   Items 1, 3, 5
x [3: ]     Items 3 to end
x [: 5]     Items 0 to 4
x [-1]      Last item
x [-3:]     Last 3 items
x [: -2]    All except last 2 items
```

All operations from Sequences, plus:

- constructors:

- `del list1[2]`
  delete item from list1

- `list1.append(item)`
  appends an item to list1

- `list1.extend(sequence1)`
  appends sequence1 to list1

- `list1.insert(index, item)`
  inserts item at index

- `list1.pop()`
  pops last item

- `list1.remove(item)`
  removes first instance of item

- `list1.reverse()`
  reverses list order

- `list1.sort()`
  sorts list in place

### Tuples

type of data structure which helps to store different values but have few differences
u see they are declared with `()` to store the data

- Support all operations for Sequences
- Immutable, but member objects may be mutable
- If the contents of a list shouldn't change, use a tuple to prevent items from accidently being added, changed or deleted
- Tuples are more efficient than lists due to Python's implementation

```python
coordinates = (4, 5)
print(coordinates[0])
```

they are immutable by default

a list can have tuples as it's arguments

### Functions

coillection of code that performs some task

```python
def say_hi(name):
    print("Hello " + name)

say_hi("Mike")
say_hi("Steve")
```

### Return Statement

return something info from a function

```python
def cube(num):
    return num ** 3

print(cube(3))
```

### If Statements

check for a cindition if true of not

```python
is_male = True
if is_male:
    print("You are a male.")
else:
    print("You are not a male.")
```

### If Statements & Comparisons

checks for condition if true or not, but this one basically helps you span out things

```python
def max_num(num1, num2, num3):
    if num1 >= num2 and num1 >= num3:
        return num1
    elif num2 >= num1 and num2 >= num3:
        return num2
    else:
        return num3

print(max_num(3, 4, 5))
```

### Building a Better Calculator

```python
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
```

### Dictionaries

allows to store data in key value pairs
Unordered Map sorta

```python
monthConversions = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
}

print(monthConversions["Mar"])
```

---

| Description                                           | Code                           |
| ----------------------------------------------------- | ------------------------------ |
| Add or change item in dict x                          | ['beef"] = 25.2                |
| Remove item from dict x                               | del x['beef"]                  |
| Get length of dict x                                  | len (x)                        |
| Check membership in x(only looks in keys, not values) | `item in x` or `item not in x` |
| Delete all items from dict x                          | x.clear ()                     |
| Delete dict x                                         | del x                          |

### While Loop

basically run till the defining condition as long it is true

```python
i = 1
while i <= 5:
    print(i)
    i += 1

print("Done")
```

### Building a Guessing Game

basically prompt user to enter a guess, if guess is not equal to secret_word then we want him to take another guess
use a while loop

```python
secret_word = "giraffe"
guess = ""
while guess != secret_word:
    guess = input("Enter guess: ")

print("You win!")
```

### For Loops

basically something to remove code repetition

```python
for letter in "Python":
    print(letter)
```

### Exponent Function

```python
def raise_to_power(base_num, pow_num):
    result = 1
    for index in range(pow_num):
        result *= base_num
    return result

print(raise_to_power(2, 3))
```

### 2D Lists & Nested Loops

```python
number_grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [0]
]

for row in number_grid:
    for col in row:
        print(col)
```

### Building a Translator

```python
def translate(phrase):
    translation = ""
    for letter in phrase:
        if letter.lower() in "aeiou":
            translation += "g"
        else:
            translation += letter
    return translation

print(translate(input("Enter a phrase: ")))
```

### Comments

```python
# This is a single-line comment
```

### Try / Except

basically used to do exception handling

```python
try:
    number = int(input("Enter a number: "))
    print(number)
except:
    print("Invalid input")
```

### Reading Files

read the files with `open()` and `read()` function

```python
file = open("employees.txt", "r")
print(file.read())
file.close()
```

### Writing to Files

```python
file = open("employees.txt", "a")
file.write("\nToby - HR")
file.close()
```

### Modules & Pip

```bash
pip install requests
```

```python
import requests
response = requests.get("https://api.github.com")
print(response.status_code)
```

### Classes & Objects

classes is like a library that wraps around a object

object is similar to a real world entity

```python
class Student:
    def __init__(self, name, major, gpa, is_on_probation):
        self.name = name
        self.major = major
        self.gpa = gpa
        self.is_on_probation = is_on_probation
```

### Building a Multiple Choice Quiz

```python
class Question:
    def __init__(self, prompt, answer):
        self.prompt = prompt
        self.answer = answer
```

### Object Functions

function that can be used indie a class

```python
class Student:
    def __init__(self, name, gpa):
        self.name = name
        self.gpa = gpa

    def on_honor_roll(self):
        return self.gpa >= 3.5
```

### Inheritance

```python
class Chef:
    def make_chicken(self):
        print("The chef makes chicken")

class ChineseChef(Chef):
    def make_fried_rice(self):
        print("The chef makes fried rice")
```

# Homework

- [Implement matrix operations from scratch](./23.matrixOpr.py)
- [Build basic statistical analysis tool](./27.statistic-analytic.py)
- [Create data visualization dashboard](./28.data-visualisation.py)