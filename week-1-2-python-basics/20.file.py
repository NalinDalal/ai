file = open("employees.txt", "r")
# r- read the file
# w- write the file
# a- append to the file
print(file.read())  #just read the file
print(file.readlines()) #read the file line by line; 1 line at a time
file.close()


file = open("employees.txt", "a")
file.write("\nToby - HR")   #add new info
file.close()

