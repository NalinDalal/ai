friends = ["Kevin", "Karen", "Jim"]
print(friends[0])
print(friends[-1])
friends[1] = "Mike"

lucky_numbers = [4, 8, 15, 16, 23, 42]
friends_1 = ["Kevin", "Karen", "Jim", "Oscar", "Toby"]
friends_1.extend(lucky_numbers)     # # extend() adds all elements of another list to the end of the current list

# append() adds a single element to the end of the list (like push in other languages)
friends_1.append("Creed")

# insert(index, value) inserts an element at a specific position
# Here, "Kelly" will be inserted at index 1 (second position)
friends_1.insert(1, "Kelly")

# remove(value) removes the first occurrence of the specified value
friends_1.remove("Jim")

print(friends_1)

#find index of something
print(friends_1.index("Kevin"))

#find count of something
print(friends_1.count("Kevin"))
